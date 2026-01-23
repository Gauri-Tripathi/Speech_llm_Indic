import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from omni_speech.constants import IGNORE_INDEX


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def _uniform_assignment(src_lens, tgt_lens):
    tgt_indices = torch.arange(torch.max(tgt_lens)).expand(len(tgt_lens), -1).to(tgt_lens.device)
    ratio = tgt_lens / src_lens
    index_t = (tgt_indices / ratio.view(-1, 1)).long()
    return index_t


class SimpleTransformerLayer(nn.Module):
    """Simple transformer decoder layer without RoPE dependency."""
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Check if flash attention is available
        self.use_flash_attn = False
        try:
            from torch.nn.functional import scaled_dot_product_attention
            self.use_flash_attn = True
        except ImportError:
            pass
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Check input for NaN
        if torch.isnan(hidden_states).any():
            hidden_states = torch.where(torch.isnan(hidden_states), 
                                        torch.zeros_like(hidden_states), 
                                        hidden_states)
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Clamp after layer norm
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Clamp QKV to prevent explosion
        q = torch.clamp(q, min=-100, max=100)
        k = torch.clamp(k, min=-100, max=100)
        v = torch.clamp(v, min=-100, max=100)
        
        if self.use_flash_attn and seq_len > 256:
            # Use Flash Attention (much more memory efficient)
            # Convert 4D attention mask to the format expected by scaled_dot_product_attention
            if attention_mask is not None:
                # Replace -inf with large negative number (flash attn doesn't like true -inf)
                attention_mask = torch.clamp(attention_mask, min=-1e4)
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attention_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False  # We provide explicit mask
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=True
                )
        else:
            # Standard attention for short sequences
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Clamp attention scores before adding mask
            attn_weights = torch.clamp(attn_weights, min=-100, max=100)
            
            if attention_mask is not None:
                # Replace -inf with large negative number
                attention_mask = torch.clamp(attention_mask, min=-1e4)
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            
            # Check for NaN in attention weights
            if torch.isnan(attn_weights).any():
                attn_weights = torch.where(torch.isnan(attn_weights),
                                          torch.zeros_like(attn_weights),
                                          attn_weights)
            
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Check attn_output for NaN
        if torch.isnan(attn_output).any():
            attn_output = torch.where(torch.isnan(attn_output),
                                     torch.zeros_like(attn_output),
                                     attn_output)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_output = torch.clamp(attn_output, min=-100, max=100)
        
        hidden_states = residual + attn_output
        
        # Check for NaN after attention
        if torch.isnan(hidden_states).any():
            hidden_states = torch.where(torch.isnan(hidden_states), residual, hidden_states)
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Compute FFN with numerical stability
        gate = self.gate_proj(hidden_states)
        gate = torch.clamp(gate, min=-50, max=50)  # Prevent overflow in silu
        up = self.up_proj(hidden_states)
        up = torch.clamp(up, min=-1e3, max=1e3)
        
        hidden_states = self.down_proj(F.silu(gate) * up)
        hidden_states = residual + hidden_states
        
        # Clamp to prevent explosion (tighter bound)
        hidden_states = torch.clamp(hidden_states, min=-1e3, max=1e3)
        
        # Final NaN check
        if torch.isnan(hidden_states).any():
            hidden_states = torch.where(torch.isnan(hidden_states), 
                                        torch.zeros_like(hidden_states), 
                                        hidden_states)
        
        return (hidden_states,)


class SpeechGeneratorCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers, n_dims, n_heads, n_inter_dims = list(map(int, config.ctc_decoder_config[1:-1].split(",")))
        
        self.upsample_factor = config.ctc_upsample_factor
        self.input_proj = nn.Linear(config.hidden_size, n_dims)
        self.layers = nn.ModuleList(
            [SimpleTransformerLayer(n_dims, n_heads, n_inter_dims) for _ in range(n_layers)]
        )
        self.unit_vocab_size = config.unit_vocab_size
        self.output_proj = nn.Linear(n_dims, config.unit_vocab_size + 1)
        
        # Initialize weights with smaller scale to prevent instability
        self._init_weights()
    
    def _init_weights(self):
        # Use smaller initialization scale
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def upsample(self, reps, tgt_units=None):
        src_lens = torch.LongTensor([len(rep) for rep in reps]).to(reps[0].device)
        up_lens = src_lens * self.upsample_factor
        if tgt_units is not None:
            tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
            up_lens = torch.max(up_lens, tgt_lens)
        reps = torch.nn.utils.rnn.pad_sequence(reps, batch_first=True)
        padding_mask = lengths_to_padding_mask(up_lens)
        mapped_inputs = _uniform_assignment(src_lens, up_lens).masked_fill(
            padding_mask, 0
        )
        copied_reps = torch.gather(
            reps,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), reps.size(-1)
            ),
        )
        copied_reps = copied_reps.masked_fill(padding_mask.unsqueeze(-1), 0)
        position_ids = torch.arange(0, max(up_lens)).unsqueeze(0).expand(len(reps), -1).to(device=copied_reps.device)
        return copied_reps, ~padding_mask, position_ids
    
    def forward(self, tgt_reps, labels, tgt_units):
        tgt_label_reps = []
        for tgt_rep, label in zip(tgt_reps, labels):
            rep = tgt_rep[label != IGNORE_INDEX]
            # Check for NaN in LLM hidden states
            if torch.isnan(rep).any():
                rep = torch.where(torch.isnan(rep), torch.zeros_like(rep), rep)
            # Clamp LLM hidden states
            rep = torch.clamp(rep, min=-100, max=100)
            tgt_label_reps.append(rep)
        
        hidden_states, attention_mask_2d, position_ids = self.upsample(tgt_label_reps, tgt_units)
        hidden_states = self.input_proj(hidden_states)
        
        # Clamp after input projection
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)
        
        # Create 4D causal + padding mask
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Causal mask: upper triangular with -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Padding mask: -inf for padded positions
        padding_mask = (~attention_mask_2d).unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(-1, 1, seq_len, -1)
        padding_mask = padding_mask.to(dtype=hidden_states.dtype) * float('-inf')
        
        attention_mask = causal_mask + padding_mask
        
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
        
        # Check for NaN in hidden states before output projection
        if torch.isnan(hidden_states).any():
            print("WARNING: NaN in hidden states before output_proj! Replacing with zeros.")
            hidden_states = torch.where(torch.isnan(hidden_states), 
                                        torch.zeros_like(hidden_states), 
                                        hidden_states)
        
        ctc_logits = self.output_proj(hidden_states)
        
        # Clamp logits to prevent overflow in softmax
        ctc_logits = torch.clamp(ctc_logits, min=-100, max=100)
        
        # Replace any remaining NaN with 0
        if torch.isnan(ctc_logits).any():
            print("WARNING: NaN in ctc_logits! Replacing with zeros.")
            ctc_logits = torch.where(torch.isnan(ctc_logits), 
                                     torch.zeros_like(ctc_logits), 
                                     ctc_logits)
        
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_lens = attention_mask_2d.long().sum(dim=-1)
        ctc_tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = tgt_units.masked_select(ctc_tgt_mask)
        
        # DEBUG: Check for common issues
        debug_mode = not hasattr(self, '_debug_count') or self._debug_count < 5
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        
        if debug_mode:
            self._debug_count += 1
            print(f"\n=== CTC Debug (call {self._debug_count}) ===")
            print(f"  ctc_lens: {ctc_lens.tolist()}")
            print(f"  tgt_lens: {ctc_tgt_lens.tolist()}")
            print(f"  unit_vocab_size: {self.unit_vocab_size}")
            print(f"  tgt_units range: [{ctc_tgt_flat.min().item()}, {ctc_tgt_flat.max().item()}]")
            print(f"  lprobs has nan: {torch.isnan(ctc_lprobs).any().item()}")
            print(f"  lprobs has inf: {torch.isinf(ctc_lprobs).any().item()}")
            print(f"  hidden_states has nan: {torch.isnan(hidden_states).any().item()}")
        
        # Check 1: Target units must be in valid range [0, unit_vocab_size)
        if ctc_tgt_flat.max() >= self.unit_vocab_size:
            print(f"ERROR: Target units ({ctc_tgt_flat.max().item()}) >= vocab size ({self.unit_vocab_size})!")
            print(f"  Fix: Increase --unit_vocab_size to at least {ctc_tgt_flat.max().item() + 1}")
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        
        if ctc_tgt_flat.min() < 0:
            print(f"ERROR: Target units contain negative values: {ctc_tgt_flat.min().item()}")
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        
        # Check 2: Log probs should not have NaN
        if torch.isnan(ctc_lprobs).any():
            print("ERROR: Log probabilities contain NaN!")
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        
        # Check 3: Validate CTC requirements: input_length >= target_length
        valid_mask = ctc_lens >= ctc_tgt_lens
        
        if not valid_mask.all():
            print(f"WARNING: CTC input shorter than target! ctc_lens={ctc_lens.tolist()}, tgt_lens={ctc_tgt_lens.tolist()}")
            # Pad the ctc_lens to be at least tgt_lens
            ctc_lens = torch.max(ctc_lens, ctc_tgt_lens)
        
        # Check 4: Empty targets
        if ctc_tgt_lens.sum() == 0:
            print("WARNING: No target units in batch!")
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
        
        try:
            ctc_loss = F.ctc_loss(
                ctc_lprobs.transpose(0, 1),
                ctc_tgt_flat,
                ctc_lens,
                ctc_tgt_lens,
                reduction="sum",
                zero_infinity=True,
                blank=self.unit_vocab_size
            )
            
            if debug_mode:
                print(f"  raw ctc_loss: {ctc_loss.item()}")
            
            # Check for NaN/Inf
            if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                print(f"WARNING: CTC loss is {ctc_loss.item()}!")
                print(f"  This usually means input_len < target_len or vocab mismatch")
                return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
                
            ctc_loss = ctc_loss / ctc_tgt_lens.sum().clamp(min=1).item()
            
            if debug_mode:
                print(f"  normalized ctc_loss: {ctc_loss.item()}")
            
        except Exception as e:
            print(f"CTC loss error: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
            
        return ctc_loss
    
    def predict(self, tgt_reps):
        hidden_states, attention_mask_2d, position_ids = self.upsample([tgt_reps])
        hidden_states = self.input_proj(hidden_states)
        
        batch_size, seq_len = hidden_states.shape[:2]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        padding_mask = (~attention_mask_2d).unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(-1, 1, seq_len, -1)
        padding_mask = padding_mask.to(dtype=hidden_states.dtype) * float('-inf')
        
        attention_mask = causal_mask + padding_mask
        
        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
            
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_pred = ctc_lprobs.argmax(dim=-1).masked_fill_(~attention_mask_2d, self.unit_vocab_size)
        return ctc_pred
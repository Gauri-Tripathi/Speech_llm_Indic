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
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        hidden_states = residual + attn_output
        
        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states
        
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
            tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])
        hidden_states, attention_mask_2d, position_ids = self.upsample(tgt_label_reps, tgt_units)
        hidden_states = self.input_proj(hidden_states)
        
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
            
        ctc_logits = self.output_proj(hidden_states)
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1, dtype=torch.float32)
        ctc_lens = attention_mask_2d.long().sum(dim=-1)
        ctc_tgt_lens = tgt_units.ne(IGNORE_INDEX).long().sum(dim=-1)
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = tgt_units.masked_select(ctc_tgt_mask)
        ctc_loss = F.ctc_loss(
            ctc_lprobs.transpose(0, 1),
            ctc_tgt_flat,
            ctc_lens,
            ctc_tgt_lens,
            reduction="sum",
            zero_infinity=True,
            blank=self.unit_vocab_size
        )
        ctc_loss /= ctc_tgt_lens.sum().item()
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
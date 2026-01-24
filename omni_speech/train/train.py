# Training script for LLaMA-Omni Hindi Speech LLM
# Adapted from LLaVA training pipeline

import os
import copy
import json
import torch
import pathlib
import transformers
import whisper

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from torch.utils.data import Dataset

from omni_speech.constants import IGNORE_INDEX, DEFAULT_SPEECH_TOKEN
from omni_speech.model import OmniSpeechLlamaForCausalLM, OmniSpeech2SLlamaForCausalLM
from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
from omni_speech import conversation as conversation_lib
from omni_speech.utils import safe_save_model_for_hf_trainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-3B-Instruct")
    version: Optional[str] = field(default="llama_3")
    freeze_backbone: bool = field(default=False)
    tune_speech_projector: bool = field(default=False)
    tune_speech_encoder: bool = field(default=False)
    tune_speech_generator_only: bool = field(default=False)
    speech_encoder_type: Optional[str] = field(default="whisper")
    speech_encoder: Optional[str] = field(default=None)
    pretrain_speech_projector: Optional[str] = field(default=None)
    speech_projector_type: Optional[str] = field(default='linear')
    speech_generator_type: Optional[str] = field(default='ctc')
    ctc_decoder_config: str = field(default="(2,4096,32,11008)")
    ctc_upsample_factor: int = field(default=25)
    ctc_loss_weight: float = field(default=1.0)
    unit_vocab_size: int = field(default=5000)
    speech_encoder_ds_rate: int = field(default=5)
    speech_encoder_hidden_size: int = field(default=1280)
    s2s: bool = field(default=True, metadata={"help": "Whether to use speech-to-speech model"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    is_multimodal: bool = field(default=True)
    input_type: str = field(default="mel")
    speech_normalize: bool = field(default=False)
    mel_size: int = field(default=128)
    has_tgt_units: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    freeze_speech_projector: bool = field(default=False)
    model_max_length: int = field(default=2048)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    speech_projector_lr: Optional[float] = field(default=None)
    group_by_modality_length: bool = field(default=False)
    # Important: Don't remove custom columns like speech, speech_length, tgt_units
    remove_unused_columns: bool = field(default=False)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with lazy loading."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        print(f"Loading data from {data_path}")
        from datasets import load_from_disk
        self.dataset = load_from_disk(data_path)
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.dataset[i]
        
        try:
            # Load speech
            speech_file = item["speech"]
            speech = whisper.load_audio(speech_file)
            
            if self.data_args.input_type == "raw":
                speech = torch.from_numpy(speech)
                if self.data_args.speech_normalize:
                    speech = torch.nn.functional.layer_norm(speech, speech.shape)
            elif self.data_args.input_type == "mel":
                speech = whisper.pad_or_trim(speech)
                speech = whisper.log_mel_spectrogram(speech, n_mels=self.data_args.mel_size).permute(1, 0)
            
            # Process conversations
            sources = copy.deepcopy([item["conversations"]])
            sources = preprocess_multimodal(sources, self.data_args)
            data_dict = preprocess(sources, self.tokenizer, has_speech=True)
            
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                               labels=data_dict["labels"][0])

            data_dict["speech"] = speech
            data_dict["speech_length"] = torch.LongTensor([speech.shape[0]])
            
            # Handle target units for speech-to-speech
            if self.data_args.has_tgt_units and "tgt_units" in item:
                tgt_units_str = item["tgt_units"]
                tgt_units = [int(x) for x in tgt_units_str.split()]
                data_dict["tgt_units"] = torch.LongTensor(tgt_units)
            
            return data_dict
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            raise


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    has_tgt_units: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # Handle speech tensors
        speeches = [instance["speech"] for instance in instances]
        speech_lengths = torch.cat([instance["speech_length"] for instance in instances])
        
        # Pad speeches to same length
        max_len = max(s.shape[0] for s in speeches)
        if speeches[0].dim() == 1:
            # Raw audio
            speeches_padded = torch.zeros(len(speeches), max_len)
        else:
            # Mel spectrogram
            speeches_padded = torch.zeros(len(speeches), max_len, speeches[0].shape[1])
        
        for i, s in enumerate(speeches):
            speeches_padded[i, :s.shape[0]] = s
        
        batch["speech"] = speeches_padded
        batch["speech_lengths"] = speech_lengths

        # Handle target units - check all instances have tgt_units
        if self.has_tgt_units:
            # Check which instances have tgt_units
            instances_with_units = [inst for inst in instances if "tgt_units" in inst]
            
            if len(instances_with_units) == 0:
                # No instances have tgt_units, skip
                print("WARNING: has_tgt_units=True but no instances have tgt_units")
            elif len(instances_with_units) != len(instances):
                # Some instances missing tgt_units - this is a data issue
                print(f"WARNING: {len(instances) - len(instances_with_units)}/{len(instances)} instances missing tgt_units")
                # Still process the ones that have it
                tgt_units = [instance.get("tgt_units", torch.LongTensor([0])) for instance in instances]
                max_unit_len = max(u.shape[0] for u in tgt_units)
                tgt_units_padded = torch.full((len(tgt_units), max_unit_len), IGNORE_INDEX, dtype=torch.long)
                for i, u in enumerate(tgt_units):
                    if "tgt_units" in instances[i]:
                        tgt_units_padded[i, :u.shape[0]] = u
                    # else: leave as IGNORE_INDEX
                batch["tgt_units"] = tgt_units_padded
            else:
                # All instances have tgt_units
                tgt_units = [instance["tgt_units"] for instance in instances]
                max_unit_len = max(u.shape[0] for u in tgt_units)
                tgt_units_padded = torch.full((len(tgt_units), max_unit_len), IGNORE_INDEX, dtype=torch.long)
                for i, u in enumerate(tgt_units):
                    tgt_units_padded[i, :u.shape[0]] = u
                batch["tgt_units"] = tgt_units_padded

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        has_tgt_units=data_args.has_tgt_units
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    data_args.is_multimodal = True

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine model class
    model_cls = OmniSpeech2SLlamaForCausalLM if model_args.s2s else OmniSpeechLlamaForCausalLM

    # Load model
    print(f"Loading model from {model_args.model_name_or_path}")
    
    # Use bf16 if available, else fp16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using compute dtype: {compute_dtype}")
    
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if training_args.bf16 else None,
    )

    # Initialize speech modules
    print("Initializing speech modules...")
    model.get_model().initialize_speech_modules(model_args)
    
    # Initialize speech generator for s2s
    if model_args.s2s:
        print("Initializing speech generator...")
        model.initialize_speech_generator(model_args)

    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Freeze components as needed
    # Handle freeze_backbone first
    if model_args.freeze_backbone:
        print("Freezing LLM backbone...")
        model.model.requires_grad_(False)

    # Handle freeze_speech_projector
    if training_args.freeze_speech_projector:
        print("Freezing speech projector...")
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = False

    # Handle tune_speech_projector (train projector only, but keep speech generator trainable for s2s)
    if model_args.tune_speech_projector:
        print("Training speech projector (and speech generator if s2s)...")
        # Freeze LLM backbone
        model.model.requires_grad_(False)
        # Freeze speech encoder (it's already frozen by default, but be explicit)
        for p in model.get_model().get_speech_encoder().parameters():
            p.requires_grad = False
        # Unfreeze speech projector
        for p in model.get_model().speech_projector.parameters():
            p.requires_grad = True
        # Keep speech generator trainable for s2s
        if model_args.s2s and hasattr(model, 'speech_generator'):
            for p in model.speech_generator.parameters():
                p.requires_grad = True

    # Handle tune_speech_generator_only (train only speech generator)
    if model_args.tune_speech_generator_only and model_args.s2s:
        print("Training speech generator only...")
        model.requires_grad_(False)
        for p in model.speech_generator.parameters():
            p.requires_grad = True

    # Print detailed trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Log trainable parameter groups for debugging
    print("\nTrainable parameter groups:")
    param_groups = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            group = name.split('.')[0] if '.' in name else name
            if group not in param_groups:
                param_groups[group] = 0
            param_groups[group] += param.numel()
    for group, count in param_groups.items():
        print(f"  {group}: {count:,} params")

    # Make data module
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

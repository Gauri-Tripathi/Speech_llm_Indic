# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):
        speech_encoder_path = model_config.speech_encoder
        
        # Check if it's a HuggingFace model directory
        if os.path.isdir(speech_encoder_path) and os.path.exists(os.path.join(speech_encoder_path, "config.json")):
            return cls._load_hf_whisper(speech_encoder_path)
        else:
            return cls._load_openai_whisper(speech_encoder_path)
    
    @classmethod
    def _load_hf_whisper(cls, model_path):
        """Load Whisper encoder from HuggingFace format."""
        from transformers import WhisperModel
        
        print(f"Loading Whisper encoder from HuggingFace format: {model_path}")
        whisper_model = WhisperModel.from_pretrained(model_path)
        encoder = whisper_model.encoder
        
        # Freeze encoder weights by default
        for param in encoder.parameters():
            param.requires_grad = False
            
        return encoder
    
    @classmethod
    def _load_openai_whisper(cls, model_name_or_path):
        """Load Whisper encoder from OpenAI format."""
        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)

        import whisper
        print(f"Loading Whisper encoder from OpenAI format: {model_name_or_path}")
        encoder = whisper.load_model(name=model_name_or_path, device='cpu').encoder
        replace_layer_norm(encoder)
        return encoder
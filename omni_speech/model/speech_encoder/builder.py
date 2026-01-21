from .speech_encoder import WhisperWrappedEncoder


def build_speech_encoder(config, freeze_encoder=None):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    
    # Determine freeze state: if not explicitly passed, check config
    if freeze_encoder is None:
        # Default to freezing unless tune_speech_encoder is True
        tune_speech_encoder = getattr(config, 'tune_speech_encoder', False)
        freeze_encoder = not tune_speech_encoder
    
    if "whisper" in speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config, freeze_encoder=freeze_encoder)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')


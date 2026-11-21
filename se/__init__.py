"""Deep-learning speech enhancement for the 2nd challenge (16 kHz mono).

    se.config      experiment configs (model + training settings, JSON)
    se.models      model registry (MODELS) and build_model
    se.losses      training losses (LOSSES)
    se.data        paired train/validation data with a waveform cache
    se.checkpoint  self-contained weight files (model name + settings + weights)
    se.inference   enhance one waveform, write it like the input
    se.metrics     PESQ-WB and SNR exactly as the course notebook scores them
"""
SAMPLE_RATE = 16_000

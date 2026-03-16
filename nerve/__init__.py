"""
NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks.

A unified eigenspectral framework for understanding how feed-forward networks 
in large language models organize and regulate information flow in 
high-dimensional latent space.

Core metrics (Section 2.2):
  - Spectral Entropy (SE): variance uniformity
  - Participation Ratio (PR): effective latent dimensionality  
  - Eigenvalue Early Enrichment (EEE): top-heaviness
  - Jensen-Shannon Divergence (JS): distributional shift

Quick start (inference-time analysis on any pretrained model):
    from nerve import NerVEAnalyzer
    analyzer = NerVEAnalyzer(model)         # auto-detects architecture
    results = analyzer.analyze(input_ids)   # one forward pass
    analyzer.print_summary(results)         # display metrics

Quick start (training-time monitoring):
    from nerve import FFNEigenMetricsCallback, register_ffn_hooks

    callback = FFNEigenMetricsCallback(log_steps=200)
    trainer.add_callback(callback)
    register_ffn_hooks(model, callback)

Reference: NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)
Project page: https://nerve-eigenspectrum.github.io
"""

# Core metrics (the main contribution)
from .metrics import (
    compute_covariance,
    compute_sorted_eigs,
    normalize_eigs,
    compute_spectral_entropy,
    compute_participation_ratio,
    compute_eee,
    compute_js,
)

# Training-time monitoring
from .callback import (
    FFNEigenMetricsCallback,
    FFNEigenMetricsHelper,
    register_ffn_hooks,
)

# Inference-time analysis (generic, works with any HuggingFace model)
from .analyzer import NerVEAnalyzer, NerVEResult, LayerMetrics

# Custom trainer (for reproducing paper experiments)
from .trainer import MyTrainer

__version__ = "0.1.0"

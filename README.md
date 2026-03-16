# NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks

<p align="center">
  <a href="https://nerve-eigenspectrum.github.io/"><b>Project Page</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2603.06922"><b>Paper</b></a> &nbsp;|&nbsp;
  <a href="https://colab.research.google.com/github/nerve-eigenspectrum/NerVE/blob/main/notebooks/demo_pretrained.ipynb"><b>Demo Notebook</b></a>
</p>


<p align="center">
  <em>To appear at ICLR 2026</em>
</p>

**NerVE** is a unified eigenspectral framework for understanding how feed-forward networks (FFNs) in large language models organize and regulate information flow in high-dimensional latent space. Despite FFNs dominating the parameter budget, their high-dimensional dynamics remain poorly understood. NerVE addresses this gap through lightweight, memory-efficient tracking of eigenspectrum dynamics via four complementary metrics.

**Core finding:** FFN nonlinearities do not merely rescale activations — they actively reinject variance across eigenmodes, reawakening previously inactive directions in high-dimensional latent space. Moreover, optimizer geometry strongly modulates the extent of this variance reinjection.

## Metrics

NerVE tracks FFN eigenspectrum dynamics through four scale-invariant, distribution-aware metrics:

| Metric | What it captures | Range |
|---|---|---|
| **Spectral Entropy (SE)** | Variance uniformity / dispersion | [0, ln D] |
| **Participation Ratio (PR)** | Effective latent dimensionality | [1, D] |
| **Eigenvalue Early Enrichment (EEE)** | Top-heaviness of the spectrum | [0, 1) |
| **Jensen-Shannon Divergence (JS)** | Distributional shift (pre → post activation) | [0, ln 2] |

## Installation

```bash
pip install nerve-spectral
```

Or install from source:

```bash
git clone https://github.com/nerve-eigenspectrum/NerVE.git
cd NerVE
pip install -e .
```

### Requirements
```bash
pip install -r requirements.txt
```

- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.20 (for analyzer and training callback)

## Quick Start

### Analyze any pretrained model (5 lines)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nerve import NerVEAnalyzer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

inputs = tokenizer("The role of feed-forward networks in transformers", return_tensors="pt")
analyzer = NerVEAnalyzer(model)  # auto-detects architecture
results = analyzer.analyze(inputs["input_ids"])
analyzer.print_summary(results)
```

Auto-detection supports: GPT-2, Pythia, GPT-J, LLaMA, Mistral, Gemma, Phi, Qwen2, and more. To add a new architecture:

```python
NerVEAnalyzer.register_architecture(
    "my_model",
    block_path="backbone.layers",
    up="feed_forward.w1",
    down="feed_forward.w2",
    gated=True,
)
```

### Monitor during training (with HuggingFace Trainer)

```python
from nerve import FFNEigenMetricsCallback, register_ffn_hooks

callback = FFNEigenMetricsCallback(
    log_steps=200,          # compute every 200 steps
    device='cuda',
    output_dir="eigen_metrics_logs",
    num_layers=12,
    do_sampling=False,      # full-batch covariance (recommended)
)

trainer.add_callback(callback)
register_ffn_hooks(model, callback)
trainer.train()
```

### Use metrics directly

```python
import torch
from nerve import compute_covariance, compute_sorted_eigs, normalize_eigs
from nerve import compute_spectral_entropy, compute_participation_ratio, compute_eee, compute_js

# Given activations tensor X of shape [N, D]
cov = compute_covariance(X)
eigenvalues = compute_sorted_eigs(cov)
eigenvalues_norm = normalize_eigs(eigenvalues)

se = compute_spectral_entropy(eigenvalues_norm)    # Spectral Entropy
pr = compute_participation_ratio(eigenvalues)       # Participation Ratio
eee = compute_eee(eigenvalues)                      # Eigenvalue Early Enrichment

# JS divergence between pre and post activation spectra
js = compute_js(eigenvalues_norm_pre, eigenvalues_norm_post)
```

## Key Findings

| Experimental Axis | Key Finding |
|---|---|
| Activation (GELU vs ReLU) | Similar trend, distinct dynamics; GELU explores broader subspace |
| Norm-free models | GELU exhibits spectral inertia; ReLU compensates for missing LayerNorms |
| FFN weight geometry | Performance tracks sustained spectral flattening |
| Norm placement (Pre/Post/Mix) | PreLN: best return-on-width; PostLN: diminishing spectral returns |
| Positional encoding | RoPE prevents mid-to-deep spectral collapse |
| Optimizer (AdamW vs Muon) | Repair vs refinement; performance follows mid-layer capacity |
| Non-transformer (MLP-Mixer) | Core findings generalize beyond transformer architecture |

## Repository Structure

```
NerVE/
├── nerve/
│   ├── metrics.py          # Core metrics: SE, PR, EEE, JS (Section 2.2)
│   ├── analyzer.py         # Inference-time analysis for any pretrained model
│   ├── callback.py         # Training-time monitoring callback (Algorithm 1)
│   ├── trainer.py          # Custom HuggingFace Trainer
│   └── models/
│       └── gpt2.py         # GPT-2 architectural variants (Sections 3.1-3.4)
├── scripts/
│   └── run_clm.py          # Training entry point
├── configs/                # Hydra configuration files
└── notebooks/              # Demo notebooks (coming soon)
```

## Reproducing Paper Experiments

All experiments are controlled via Hydra config overrides:

```bash
# Baseline GPT-2 with GELU (Section 3.1)
python scripts/run_clm.py model.activation_function=gelu

# ReLU variant
python scripts/run_clm.py model.activation_function=relu

# Norm-free models (Section 3.2)
python scripts/run_clm.py model.norm_type=free model.norm_position=free

# FFN weight geometry (Section 3.3)
python scripts/run_clm.py model.norm_type=free model.ffn_norm_type=spectral

# LayerNorm placement sweep (Section 3.4)
python scripts/run_clm.py model.norm_position=post    # PostLN
python scripts/run_clm.py model.norm_position=pre     # PreLN (default)
python scripts/run_clm.py model.norm_position=mixed model.post_ln_layers=4  # MixLN

# FFN width sweep (Section 3.4)
python scripts/run_clm.py model.mlp_width_mult=1   # D = 1d
python scripts/run_clm.py model.mlp_width_mult=4   # D = 4d (default)
python scripts/run_clm.py model.mlp_width_mult=8   # D = 8d
```

## Citation

If you find NerVE useful in your research, please cite:

```bibtex
@inproceedings{jha2026nerve,
    title={NerVE: Nonlinear Eigenspectrum Dynamics in {LLM} Feed-Forward Networks},
    author={Nandan Kumar Jha and Brandon Reagen},
    booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
    year={2026},    
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Our GPT-2 implementation builds on the [Simplified Transformers](https://github.com/bobby-he/simplified_transformers) codebase.

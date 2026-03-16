"""
NerVE: Generic Inference-time Eigenspectrum Analyzer.

Applies the NerVE framework to any pretrained HuggingFace model without
requiring the training callback. Supports multiple model families out of
the box, and is easily extensible to new architectures.

Usage:
    from nerve.analyzer import NerVEAnalyzer

    analyzer = NerVEAnalyzer(model)                    # auto-detects architecture
    results = analyzer.analyze(input_ids)              # run analysis
    analyzer.print_summary(results)                    # display results

    # Or with explicit model type:
    analyzer = NerVEAnalyzer(model, model_type="pythia")

    # Analyze multiple Pythia checkpoints (training dynamics):
    for step in [0, 1000, 10000, 143000]:
        model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/pythia-410m", revision=f"step{step}")
        results = NerVEAnalyzer(model).analyze(input_ids)

Reference: NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks (ICLR 2026)
"""

import torch
import gc
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .metrics import (
    compute_covariance,
    compute_sorted_eigs,
    normalize_eigs,
    compute_spectral_entropy,
    compute_participation_ratio,
    compute_eee,
    compute_js,
)


# ============================================================
#  Model Architecture Registry
# ============================================================
#
# Each entry maps a model family to its FFN module paths.
#
#   block_path: attribute path from model root to the list of transformer blocks
#   up:         attribute path from a block to the up-projection (or gate-projection for gated FFNs)
#   down:       attribute path from a block to the down-projection
#   gated:      whether the FFN uses a gating mechanism (SwiGLU, GeGLU, etc.)
#
# For NON-GATED architectures (e.g., GPT-2, Pythia):
#   PreAct  = output of up-projection (before activation)
#   PostAct = input to down-projection (after activation)
#
# For GATED architectures (e.g., LLaMA, Mistral):
#   PreAct  = output of gate-projection (before activation, i.e., W_gate * x)
#   PostAct = input to down-projection (after gating, i.e., sigma(W_gate * x) * (W_up * x))

MODEL_REGISTRY = {
    # GPT-2 family
    "gpt2": {
        "block_path": "transformer.h",
        "up": "mlp.c_fc",
        "down": "mlp.c_proj",
        "gated": False,
    },
    # GPT-NeoX / Pythia family
    "gpt_neox": {
        "block_path": "gpt_neox.layers",
        "up": "mlp.dense_h_to_4h",
        "down": "mlp.dense_4h_to_h",
        "gated": False,
    },
    # GPT-J family
    "gptj": {
        "block_path": "transformer.h",
        "up": "mlp.fc_in",
        "down": "mlp.fc_out",
        "gated": False,
    },
    # LLaMA family (gated SwiGLU)
    "llama": {
        "block_path": "model.layers",
        "up": "mlp.gate_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    # Mistral family (same architecture as LLaMA)
    "mistral": {
        "block_path": "model.layers",
        "up": "mlp.gate_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    # Gemma family (gated)
    "gemma": {
        "block_path": "model.layers",
        "up": "mlp.gate_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    "gemma2": {
        "block_path": "model.layers",
        "up": "mlp.gate_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    # Phi family
    "phi": {
        "block_path": "model.layers",
        "up": "mlp.fc1",
        "down": "mlp.fc2",
        "gated": False,
    },
    "phi3": {
        "block_path": "model.layers",
        "up": "mlp.gate_up_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    # Qwen2 family (gated)
    "qwen2": {
        "block_path": "model.layers",
        "up": "mlp.gate_proj",
        "down": "mlp.down_proj",
        "gated": True,
    },
    # Custom GPT-2 variants (from NerVE paper experiments)
    "nerve_gpt2": {
        "block_path": "transformer.h",
        "up": "mlp.c_fc",
        "down": "mlp.c_proj",
        "gated": False,
    },
}


def _resolve_attr(obj, attr_path: str):
    """Resolve a dotted attribute path like 'model.layers' on an object."""
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj


def auto_detect_model_type(model) -> Optional[str]:
    """
    Auto-detect model family from the model's config.

    Uses config.model_type which is set by HuggingFace for all standard models.
    Falls back to None if detection fails.

    Args:
        model: a HuggingFace model

    Returns:
        model type string (key into MODEL_REGISTRY), or None
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    model_type = getattr(config, "model_type", None)
    if model_type and model_type in MODEL_REGISTRY:
        return model_type

    # Fallback: check class name patterns
    class_name = model.__class__.__name__.lower()
    for key in MODEL_REGISTRY:
        if key in class_name:
            return key

    return None


# ============================================================
#  Analysis Result
# ============================================================

@dataclass
class LayerMetrics:
    """Eigenspectrum metrics for a single FFN layer."""
    layer_idx: int
    se_pre: float
    se_post: float
    pr_pre: float
    pr_post: float
    eee_pre: float
    eee_post: float
    js: float

    # Derived quantities (Section B.3)
    @property
    def pr_gain(self) -> float:
        """Post/Pre participation ratio gain."""
        return self.pr_post / max(self.pr_pre, 1e-12)

    @property
    def eee_diff(self) -> float:
        """EEE difference (Post - Pre). More negative = stronger flattening."""
        return self.eee_post - self.eee_pre


@dataclass
class NerVEResult:
    """Complete NerVE analysis result for a model."""
    model_type: str
    num_layers: int
    ffn_dim: int
    num_tokens: int
    layers: Dict[int, LayerMetrics] = field(default_factory=dict)

    def get_metric(self, metric_name: str) -> List[float]:
        """
        Get a specific metric across all layers.

        Args:
            metric_name: one of 'se_pre', 'se_post', 'pr_pre', 'pr_post',
                         'eee_pre', 'eee_post', 'js', 'pr_gain', 'eee_diff'

        Returns:
            list of values, one per layer in order
        """
        return [getattr(self.layers[i], metric_name) for i in sorted(self.layers.keys())]

    def summary(self) -> Dict[str, float]:
        """Mean of each metric across all layers."""
        metrics = ['se_pre', 'se_post', 'pr_pre', 'pr_post',
                   'eee_pre', 'eee_post', 'js', 'pr_gain', 'eee_diff']
        result = {}
        for m in metrics:
            vals = self.get_metric(m)
            result[m + '_mean'] = sum(vals) / len(vals)
        return result


# ============================================================
#  NerVE Analyzer
# ============================================================

class NerVEAnalyzer:
    """
    Generic inference-time eigenspectrum analyzer.

    Attaches temporary hooks to any HuggingFace model's FFN layers,
    runs a forward pass, computes NerVE metrics, and cleans up.

    Args:
        model: a HuggingFace model (e.g., from AutoModelForCausalLM.from_pretrained)
        model_type: model family string (e.g., 'gpt2', 'gpt_neox', 'llama').
                    If None, auto-detected from model.config.model_type.
        device: device for eigendecomposition. If None, uses model's device.

    Example:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
        analyzer = NerVEAnalyzer(model)
        results = analyzer.analyze(input_ids)
        analyzer.print_summary(results)
    """

    def __init__(self, model, model_type: Optional[str] = None, device: Optional[str] = None):
        self.model = model
        self.model.eval()

        # Auto-detect or validate model type
        if model_type is None:
            model_type = auto_detect_model_type(model)
        if model_type is None:
            raise ValueError(
                "Could not auto-detect model type. Please specify model_type explicitly.\n"
                f"Supported types: {list(MODEL_REGISTRY.keys())}\n"
                "Or register a new architecture with NerVEAnalyzer.register_architecture()"
            )
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'.\n"
                f"Supported types: {list(MODEL_REGISTRY.keys())}\n"
                "Register new architectures with NerVEAnalyzer.register_architecture()"
            )

        self.model_type = model_type
        self.arch_config = MODEL_REGISTRY[model_type]

        # Resolve the list of transformer blocks
        self.blocks = list(_resolve_attr(model, self.arch_config["block_path"]))
        self.num_layers = len(self.blocks)

        # Determine compute device
        if device is not None:
            self.compute_device = device
        else:
            try:
                self.compute_device = str(next(model.parameters()).device)
            except StopIteration:
                self.compute_device = "cpu"

    @classmethod
    def register_architecture(cls, name: str, block_path: str, up: str, down: str, gated: bool = False):
        """
        Register a new model architecture for NerVE analysis.

        Args:
            name: identifier for this architecture
            block_path: dotted path from model root to the list of transformer blocks
            up: dotted path from a block to the up-projection (or gate-projection)
            down: dotted path from a block to the down-projection
            gated: whether the FFN uses gating (SwiGLU, GeGLU, etc.)

        Example:
            # Adding support for a custom model
            NerVEAnalyzer.register_architecture(
                "my_model",
                block_path="backbone.layers",
                up="feed_forward.w1",
                down="feed_forward.w2",
                gated=True,
            )
        """
        MODEL_REGISTRY[name] = {
            "block_path": block_path,
            "up": up,
            "down": down,
            "gated": gated,
        }

    def analyze(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> NerVEResult:
        """
        Run NerVE eigenspectrum analysis on the model.

        Attaches temporary hooks, runs one forward pass, computes all four
        metrics (SE, PR, EEE, JS) for every FFN layer, and cleans up.

        Args:
            input_ids: [B, S] tensor of token IDs
            attention_mask: optional [B, S] attention mask

        Returns:
            NerVEResult with per-layer and summary metrics
        """
        # Storage for activations: layer_idx -> tensor
        pre_acts: Dict[int, torch.Tensor] = {}
        post_acts: Dict[int, torch.Tensor] = {}
        hook_handles = []

        up_path = self.arch_config["up"]
        down_path = self.arch_config["down"]

        # --- Attach hooks ---
        for layer_idx, block in enumerate(self.blocks):
            up_module = _resolve_attr(block, up_path)
            down_module = _resolve_attr(block, down_path)

            def make_up_hook(idx):
                def hook(mod, inp, out):
                    # out is PreAct: output of up/gate projection before activation
                    pre_acts[idx] = out.detach()
                return hook

            def make_down_hook(idx):
                def hook(mod, inp):
                    # inp[0] is PostAct: input to down projection (after activation/gating)
                    post_acts[idx] = inp[0].detach()
                return hook

            h1 = up_module.register_forward_hook(make_up_hook(layer_idx))
            h2 = down_module.register_forward_pre_hook(make_down_hook(layer_idx))
            hook_handles.extend([h1, h2])

        # --- Forward pass ---
        try:
            with torch.no_grad():
                kwargs = {"input_ids": input_ids}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                self.model(**kwargs)
        finally:
            # Always remove hooks, even if forward pass fails
            for h in hook_handles:
                h.remove()

        # --- Compute metrics per layer (Algorithm 1: sequential with cleanup) ---
        ffn_dim = None
        layer_results = {}

        for layer_idx in sorted(pre_acts.keys()):
            if layer_idx not in post_acts:
                continue

            pre_tensor = pre_acts[layer_idx]
            post_tensor = post_acts[layer_idx]

            # Flatten [B, S, D] -> [N, D]  (Section 2.1: discard sequence order)
            if pre_tensor.dim() == 3:
                D = pre_tensor.shape[-1]
                pre_tensor = pre_tensor.reshape(-1, D)
                post_tensor = post_tensor.reshape(-1, D)
            else:
                D = pre_tensor.shape[-1]

            if ffn_dim is None:
                ffn_dim = D

            num_tokens = pre_tensor.shape[0]

            # Move to compute device
            pre_tensor = pre_tensor.float().to(self.compute_device)
            post_tensor = post_tensor.float().to(self.compute_device)

            # Covariance -> eigenvalues -> metrics (for pre-activation)
            cov_pre = compute_covariance(pre_tensor)
            lam_pre = compute_sorted_eigs(cov_pre)
            lam_norm_pre = normalize_eigs(lam_pre)
            se_pre = compute_spectral_entropy(lam_norm_pre).item()
            pr_pre = compute_participation_ratio(lam_pre).item()
            eee_pre = compute_eee(lam_pre).item()
            del cov_pre

            # Same for post-activation
            cov_post = compute_covariance(post_tensor)
            lam_post = compute_sorted_eigs(cov_post)
            lam_norm_post = normalize_eigs(lam_post)
            se_post = compute_spectral_entropy(lam_norm_post).item()
            pr_post = compute_participation_ratio(lam_post).item()
            eee_post = compute_eee(lam_post).item()
            del cov_post

            # JS divergence between pre and post spectra
            js_val = compute_js(lam_norm_pre, lam_norm_post).item()

            layer_results[layer_idx] = LayerMetrics(
                layer_idx=layer_idx,
                se_pre=se_pre, se_post=se_post,
                pr_pre=pr_pre, pr_post=pr_post,
                eee_pre=eee_pre, eee_post=eee_post,
                js=js_val,
            )

            # Cleanup this layer before moving to next (Algorithm 1, line 5)
            del pre_acts[layer_idx], post_acts[layer_idx]
            del pre_tensor, post_tensor, lam_pre, lam_post, lam_norm_pre, lam_norm_post
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return NerVEResult(
            model_type=self.model_type,
            num_layers=self.num_layers,
            ffn_dim=ffn_dim or 0,
            num_tokens=num_tokens if layer_results else 0,
            layers=layer_results,
        )

    @staticmethod
    def print_summary(result: NerVEResult):
        """
        Print a formatted summary of NerVE analysis results.

        Args:
            result: NerVEResult from analyze()
        """
        print(f"\n{'='*80}")
        print(f"NerVE Eigenspectrum Analysis")
        print(f"{'='*80}")
        print(f"Model type: {result.model_type} | Layers: {result.num_layers} | "
              f"FFN dim: {result.ffn_dim} | Tokens: {result.num_tokens}")
        print(f"{'='*80}")
        print(f"{'Layer':>5} | {'SE_pre':>7} {'SE_post':>7} | {'PR_pre':>8} {'PR_post':>8} "
              f"{'PR_gain':>8} | {'EEE_pre':>7} {'EEE_post':>8} {'dEEE':>7} | {'JS':>7}")
        print(f"{'-'*80}")

        for idx in sorted(result.layers.keys()):
            m = result.layers[idx]
            print(f"{m.layer_idx:>5} | {m.se_pre:>7.3f} {m.se_post:>7.3f} | "
                  f"{m.pr_pre:>8.1f} {m.pr_post:>8.1f} {m.pr_gain:>8.1f}x | "
                  f"{m.eee_pre:>7.3f} {m.eee_post:>8.3f} {m.eee_diff:>7.3f} | "
                  f"{m.js:>7.4f}")

        print(f"{'-'*80}")
        s = result.summary()
        print(f"{'Mean':>5} | {s['se_pre_mean']:>7.3f} {s['se_post_mean']:>7.3f} | "
              f"{s['pr_pre_mean']:>8.1f} {s['pr_post_mean']:>8.1f} {s['pr_gain_mean']:>8.1f}x | "
              f"{s['eee_pre_mean']:>7.3f} {s['eee_post_mean']:>8.3f} {s['eee_diff_mean']:>7.3f} | "
              f"{s['js_mean']:>7.4f}")
        print(f"{'='*80}")

        # Highlight the core finding
        if s['se_post_mean'] > s['se_pre_mean']:
            print(f"\nCore finding: Post-activation SE > Pre-activation SE across layers")
            print(f"  -> FFN nonlinearity is reinjecting variance (SE_post - SE_pre = "
                  f"{s['se_post_mean'] - s['se_pre_mean']:.3f})")
        if s['eee_diff_mean'] < 0:
            print(f"  -> Eigenspectrum is being flattened (mean dEEE = {s['eee_diff_mean']:.3f})")
        print()

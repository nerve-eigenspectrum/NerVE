"""
Custom HuggingFace Trainer for NerVE experiments.

Modifications over the standard Trainer:
  - Separate weight decay handling for gain parameters (norm-free models, Section 3.2)
  - Post-optimizer weight re-normalization for hyperspherical modes (Section 3.3)
"""

import torch
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names


class MyTrainer(Trainer):
    def create_optimizer(self):
        """
        Identical to standard HF AdamW optimizer, but with no weight decay for gain parameters.
        This is needed for norm-free models (Section 3.2) where gain parameters replace LayerNorm.
        """
        opt_model = self.model
        self.logged_steps = set()
        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, [torch.nn.LayerNorm]
            )
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            gain_parameters = [name for name in decay_parameters if "gain" in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in gain_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in gain_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Standard HF compute_loss."""
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        """
        After each optimizer step, re-normalize FFN weights for hyperspherical modes.
        This projects weights back onto the unit hypersphere (Section 3.3).
        """
        super().optimizer_step(*args, **kwargs)

        ffn_norm_type = getattr(self.model.config, "ffn_norm_type", "none")
        if ffn_norm_type in ["hyperspherical", "ngpt"]:
            for layer in self.model.transformer.h:
                if hasattr(layer.mlp, 'post_update_step'):
                    layer.mlp.post_update_step()

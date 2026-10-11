"""bitsandbytes: 4-bit NF4 or 8-bit weights, no calibration needed."""

from __future__ import annotations

from ocq.quantizers.base import TransformersQuantizer, free_memory
from ocq.types import OptimizationPlan, QuantAlgorithm


class BnbQuantizer(TransformersQuantizer):
    algorithm = QuantAlgorithm.BNB
    summary = "4 or 8 bit on an NVIDIA GPU, no calibration. Loads straight into transformers."
    install_hint = "uv sync --extra bnb"
    needs = ("bitsandbytes", "transformers", "accelerate")
    needs_cuda = True

    def quantize(self, plan: OptimizationPlan) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if plan.quant_config.bits == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"Quantizing with {plan.quant_config.summary()} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, quantization_config=config, device_map="cuda"
        )
        model.save_pretrained(self.artifact_dir)
        AutoTokenizer.from_pretrained(self.model_dir).save_pretrained(self.artifact_dir)

        del model
        free_memory()

"""Calibration based quantization with llm-compressor. AWQ and GPTQ only differ in the recipe."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace

from ocq.quantizers.base import CALIBRATION_DATASET, TransformersQuantizer, free_memory
from ocq.types import BenchmarkResult, OcqError, OptimizationPlan


class OneShotQuantizer(TransformersQuantizer):
    needs = ("llmcompressor",)
    symmetric = True

    def __init__(self) -> None:
        super().__init__()
        self.calibration = None

    @abstractmethod
    def recipe(self, scheme) -> list:
        """The llm-compressor modifiers that apply this algorithm."""

    def calibrate(self, plan: OptimizationPlan) -> None:
        from datasets import Dataset, load_dataset
        from transformers import AutoTokenizer

        samples = plan.quant_config.calibration_samples
        length = plan.quant_config.calibration_seq_len

        print(f"Loading {samples} calibration samples of {length} tokens ...", flush=True)
        rows = load_dataset(*CALIBRATION_DATASET, split="train")
        text = "\n\n".join(row for row in rows["text"] if row.strip())

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        tokens = tokenizer(text[: samples * length * 8], add_special_tokens=False).input_ids
        chunks = [tokens[i : i + length] for i in range(0, len(tokens) - length + 1, length)]
        chunks = chunks[:samples]
        if not chunks:
            raise OcqError("Calibration text is shorter than one sample. Lower --seq-len.")

        self.calibration = Dataset.from_dict(
            {"input_ids": chunks, "attention_mask": [[1] * length] * len(chunks)}
        )

    def quantize(self, plan: OptimizationPlan) -> None:
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
        from llmcompressor import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = plan.quant_config
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype="auto")
        group_size = group_size_that_fits(model, config.group_size)
        if group_size != config.group_size:
            print(
                f"Group size {config.group_size} does not divide this model's layers, "
                f"using {group_size}.",
                flush=True,
            )
            config = replace(config, group_size=group_size)
            plan.quant_config = config

        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=config.bits,
                type="int",
                symmetric=self.symmetric,
                strategy="group",
                group_size=group_size,
            ),
        )

        print(f"Quantizing with {config.summary()} (this can take a while) ...", flush=True)
        oneshot(
            model=model,
            dataset=self.calibration,
            recipe=self.recipe(scheme),
            max_seq_length=config.calibration_seq_len,
            num_calibration_samples=len(self.calibration),
        )

        model.save_pretrained(self.artifact_dir, save_compressed=True)
        AutoTokenizer.from_pretrained(self.model_dir).save_pretrained(self.artifact_dir)

        del model
        free_memory()

    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        result = super().benchmark(plan, artifact_path)
        result.notes.append(
            "transformers also unpacks these weights back to 16-bit, vLLM runs them packed."
        )
        return result


def group_size_that_fits(model, wanted: int) -> int:
    import torch

    widths = {
        module.in_features
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name
    }
    for size in (wanted, 128, 64, 32, 16, 8):
        if size <= wanted and all(width % size == 0 for width in widths):
            return size
    raise OcqError(f"No group size up to {wanted} divides this model's layer widths.")

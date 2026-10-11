"""What the transformers based backends (AWQ, GPTQ, BNB) have in common."""

from __future__ import annotations

import gc
import importlib.util
import os
import shutil
import time
from pathlib import Path
from typing import ClassVar

from ocq.quantizers.protocol import BENCHMARK_PROMPT, BENCHMARK_TOKENS, Quantizer
from ocq.types import BenchmarkResult, OptimizationPlan

CALIBRATION_DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1")

NOT_NEEDED_FOR_QUANTIZATION = [
    "*.onnx",
    "onnx/*",
    "openvino/*",
    "coreml/*",
    "runs/*",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.tflite",
]


# PyTorch and llm-compressor compile some GPU kernels the first time they run on Linux.
# On a PC without a C compiler that crashes, so tell both to use their prebuilt kernels.
if not any(shutil.which(compiler) for compiler in ("cc", "gcc", "clang")):
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    os.environ.setdefault("CT_ENFORCE_EAGER", "1")


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def download_model(model_id: str) -> Path:
    from huggingface_hub import list_repo_files, snapshot_download

    skip = list(NOT_NEEDED_FOR_QUANTIZATION)
    if any(name.endswith(".safetensors") for name in list_repo_files(model_id)):
        skip += ["*.bin", "*.pt", "*.pth"]

    print(f"Downloading {model_id} ...", flush=True)
    return Path(snapshot_download(model_id, ignore_patterns=skip))


def free_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TransformersQuantizer(Quantizer):
    needs: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.artifact_dir: Path | None = None

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(module) for module in cls.needs)

    def prepare(self, plan: OptimizationPlan) -> None:
        self.model_dir = download_model(plan.model_id)
        config = plan.quant_config
        name = f"{model_slug(plan.model_id)}-{config.algorithm.value}-{config.bits}bit"
        self.artifact_dir = plan.output_dir / "artifact" / name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def calibrate(self, plan: OptimizationPlan) -> None:
        return

    def save(self, plan: OptimizationPlan) -> str:
        return str(self.artifact_dir)

    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        free_memory()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(artifact_path)
        model = AutoModelForCausalLM.from_pretrained(artifact_path, device_map=device)

        prompt = BENCHMARK_PROMPT
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": BENCHMARK_PROMPT}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        model.generate(**inputs, max_new_tokens=4, do_sample=False)

        timer = TokenTimer()
        output = model.generate(
            **inputs, max_new_tokens=BENCHMARK_TOKENS, do_sample=False, streamer=timer
        )
        text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

        del model
        free_memory()

        return BenchmarkResult(
            tokens_per_second=timer.tokens_per_second(),
            time_to_first_token_s=timer.time_to_first_token(),
            method="transformers generate",
            device=device,
            sample_output=text,
            notes=[
                f"{timer.generated_tokens()} tokens generated from a {prompt_length} token prompt.",
                "transformers adds Python overhead per token, a serving runtime is faster.",
            ],
        )


class TokenTimer:
    """Passed to generate() as the streamer: it gets the prompt first, then every new token."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def put(self, value) -> None:
        self.times.append(time.perf_counter())

    def end(self) -> None:
        return

    def generated_tokens(self) -> int:
        return max(len(self.times) - 1, 0)

    def time_to_first_token(self) -> float | None:
        if len(self.times) < 2:
            return None
        return self.times[1] - self.times[0]

    def tokens_per_second(self) -> float | None:
        if len(self.times) < 3:
            return None
        return (len(self.times) - 2) / (self.times[-1] - self.times[1])

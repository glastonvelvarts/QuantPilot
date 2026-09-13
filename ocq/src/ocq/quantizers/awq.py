"""AutoAWQ backend — install with pip install ocq[awq]."""

from __future__ import annotations

from pathlib import Path

from ocq.quantizers.protocol import Quantizer
from ocq.types import (
    BenchmarkResult,
    OptimizationPlan,
    QuantAlgorithm,
)

# AutoAWQ's stable, documented path is 4-bit (w_bit=4). Other bit widths
# exist in some forks but aren't reliably supported — validate() rejects
# anything else rather than silently attempting it.
_SUPPORTED_BITS = {4}

# AutoAWQ's own default calibration set when you don't pass calib_data.
# Pulling it explicitly (rather than letting quantize() default to it) lets
# OCQ control sample count / determinism via plan.quant_config.
_DEFAULT_CALIB_DATASET = "mit-han-lab/pile-val-backup"


class AwqQuantizer(Quantizer):
    """
    AWQ backend powered by AutoAWQ.

    Pipeline:

        HuggingFace model
            ↓
        AutoAWQForCausalLM.from_pretrained
            ↓
        Calibration samples (pileval subset, size from plan.quant_config)
            ↓
        model.quantize(...)
            ↓
        Quantized AWQ safetensors

    Inherits Quantizer (protocol.py) directly, NOT StubQuantizer — see the
    note in quantizers/gguf.py for why: StubQuantizer.run() hardcodes
    success=False regardless of what quantize()/save()/benchmark() actually
    did, which would silently discard a real result here.
    """

    algorithm = QuantAlgorithm.AWQ
    package_extra = "awq"
    pip_hint = "uv sync --extra awq"

    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.calib_data: list[str] | None = None
        self.quantized_dir: Path | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """
        Check whether the `awq` package is importable. Deliberately does
        NOT check for a GPU here — availability is about "is the package
        installed", not "can this actually run right now"; the GPU
        requirement is a validate()-time concern instead, so the failure
        message can be specific about which check failed.
        """
        try:
            import awq  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, plan: OptimizationPlan) -> tuple[bool, str]:
        if not self.is_available():
            return False, f"AWQ backend not installed. Run: {self.pip_hint}"

        if not plan.model_id:
            return False, "Model ID is required."

        if plan.quant_config.bits not in _SUPPORTED_BITS:
            return (
                False,
                f"AWQ backend only supports {sorted(_SUPPORTED_BITS)}-bit, "
                f"got {plan.quant_config.bits}-bit. Pick a different "
                "algorithm (GPTQ/GGUF) for other bit widths.",
            )

        if not plan.hardware.has_gpu:
            return (
                False,
                "AWQ quantization requires a CUDA GPU — AutoAWQ has no "
                "CPU quantization path. Use the BNB backend instead.",
            )

        return True, "AWQ backend ready."

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self, plan: OptimizationPlan) -> None:
        """
        Download the HuggingFace model into the OCQ workspace.

        huggingface_hub import is deliberately kept inside this method
        (not at module top) so importing AwqQuantizer to just check
        is_available() never requires huggingface_hub to be installed.
        """
        from huggingface_hub import snapshot_download

        workspace = plan.output_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        model_dir = workspace / "hf_model"

        if model_dir.exists() and any(model_dir.iterdir()):
            self.model_dir = model_dir
            return

        print(f"Downloading {plan.model_id} ...")
        self.model_dir = Path(
            snapshot_download(
                repo_id=plan.model_id,
                local_dir=str(model_dir),
            )
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, plan: OptimizationPlan) -> None:
        """
        Pull calibration text up front rather than letting AutoAWQ fetch
        its own default set inside quantize() — this way sample count and
        sequence length come from plan.quant_config (set by the
        Optimization Engine based on model size), not a hardcoded default.
        """
        from datasets import load_dataset

        n_samples = plan.quant_config.calibration_samples
        seq_len = plan.quant_config.calibration_seq_len

        print(
            f"Loading {n_samples} calibration samples from "
            f"{_DEFAULT_CALIB_DATASET} ..."
        )

        dataset = load_dataset(_DEFAULT_CALIB_DATASET, split="validation")

        texts: list[str] = []
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if len(text.split()) < 32:  # skip near-empty rows
                continue
            # Rough chars-per-token cap so calibration forward passes stay
            # near the target sequence length without a full tokenizer pass
            # here — AutoAWQ re-tokenizes and truncates internally anyway.
            texts.append(text[: seq_len * 4])
            if len(texts) >= n_samples:
                break

        if not texts:
            raise RuntimeError(
                f"Calibration dataset {_DEFAULT_CALIB_DATASET!r} yielded "
                "no usable samples."
            )

        self.calib_data = texts

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(self, plan: OptimizationPlan) -> None:
        if self.model_dir is None:
            raise RuntimeError("prepare() must run before quantize().")
        if self.calib_data is None:
            raise RuntimeError("calibrate() must run before quantize().")

        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        quant_config = {
            "zero_point": True,
            "q_group_size": plan.quant_config.group_size or 128,
            "w_bit": plan.quant_config.bits,
            "version": "GEMM",
        }

        print(f"Loading {self.model_dir} for AWQ quantization ...")
        model = AutoAWQForCausalLM.from_pretrained(
            str(self.model_dir), safetensors=True, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), trust_remote_code=True
        )

        print(f"Quantizing with config: {quant_config} (this can take a while) ...")
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=self.calib_data,
        )

        quant_dir = (
            plan.output_dir / "artifact" / f"{self._model_slug(plan.model_id)}-awq"
        )
        quant_dir.mkdir(parents=True, exist_ok=True)

        model.save_quantized(str(quant_dir))
        tokenizer.save_pretrained(str(quant_dir))

        self.quantized_dir = quant_dir

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, plan: OptimizationPlan) -> str:
        if self.quantized_dir is None:
            raise RuntimeError("Quantized AWQ artifact does not exist.")

        if not self.quantized_dir.exists():
            raise RuntimeError(
                f"Quantized AWQ directory missing: {self.quantized_dir}"
            )

        return str(self.quantized_dir)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, plan: OptimizationPlan, artifact_path: str) -> BenchmarkResult:
        """
        Benchmarking will eventually load the artifact under vLLM/
        transformers and measure real throughput. For now we verify the
        artifact exists and expose the path.
        """
        artifact = Path(artifact_path)

        if not artifact.exists():
            return BenchmarkResult(
                tokens_per_second=None,
                method="awq-bench",
                notes=["AWQ artifact does not exist."],
            )

        return BenchmarkResult(
            tokens_per_second=None,
            method="awq-bench",
            notes=[
                "AWQ quantization completed.",
                "Runtime benchmark not implemented yet.",
                f"Artifact: {artifact}",
            ],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_slug(model_id: str) -> str:
        return model_id.replace("/", "--")
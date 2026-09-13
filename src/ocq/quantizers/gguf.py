"""GGUF quantization backend using llama.cpp."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ocq.quantizers.protocol import Quantizer
from ocq.types import (
    BenchmarkResult,
    OptimizationPlan,
    QuantAlgorithm,
)


class GGUFQuantizer(Quantizer):
    """
    GGUF backend powered by llama.cpp.

    Pipeline:

        HuggingFace model
            ↓
        convert_hf_to_gguf.py
            ↓
        FP16/BF16 GGUF
            ↓
        llama-quantize
            ↓
        Quantized GGUF

    Inherits Quantizer (protocol.py) directly, NOT StubQuantizer —
    StubQuantizer.run() hardcodes success=False with an "not implemented
    yet" message, which is correct for backends that are only an
    availability check, but would silently discard a REAL quantize() result
    here. Quantizer.run() actually reports what happened.
    """

    algorithm = QuantAlgorithm.GGUF

    package_extra = "gguf"
    pip_hint = (
        "Install/build llama.cpp and make sure "
        "`llama-quantize` is available on PATH."
    )

    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.fp_gguf: Path | None = None
        self.quantized_gguf: Path | None = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """
        Check whether llama.cpp's quantization binary is available.
        """

        return (
            shutil.which("llama-quantize") is not None
            or shutil.which("llama-quantize.exe") is not None
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        plan: OptimizationPlan,
    ) -> tuple[bool, str]:
        """
        Validate llama.cpp and the model configuration.
        """

        if not self.is_available():
            return False, f"GGUF backend not ready. {self.pip_hint}"

        quantize_bin = self._find_quantize_binary()

        if quantize_bin is None:
            return (
                False,
                "llama-quantize not found. "
                "Build llama.cpp and add its bin directory to PATH.",
            )

        if not plan.model_id:
            return False, "Model ID is required."

        if plan.quant_config.bits not in {2, 3, 4, 5, 6, 8}:
            return (
                False,
                f"Unsupported GGUF bit width: {plan.quant_config.bits}",
            )

        return (
            True,
            f"GGUF backend ready: {quantize_bin}",
        )

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self, plan: OptimizationPlan) -> None:
        """
        Download the HuggingFace model into the OCQ workspace.

        huggingface_hub import is deliberately kept inside this method
        (not at module top) so importing GGUFQuantizer to just check
        is_available() never requires huggingface_hub to be installed.
        """
        from huggingface_hub import snapshot_download

        workspace = plan.output_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        model_dir = workspace / "hf_model"

        if model_dir.exists() and any(model_dir.iterdir()):
            self.model_dir = model_dir
            return

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
        GGUF quantization through llama.cpp does not use the same
        calibration workflow as AWQ/GPTQ.

        The HF → GGUF conversion and llama-quantize stages are
        sufficient for this backend.

        Kept as a no-op because the Quantizer interface expects
        every backend to expose a calibration stage.
        """

        return

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize(self, plan: OptimizationPlan) -> None:
        """
        Convert the HuggingFace model to GGUF and then quantize it.
        """

        if self.model_dir is None:
            raise RuntimeError("prepare() must run before quantize().")

        llama_cpp_dir = self._find_llama_cpp_dir()

        converter = llama_cpp_dir / "convert_hf_to_gguf.py"

        if not converter.exists():
            raise RuntimeError(
                f"GGUF converter not found: {converter}"
            )

        workspace = plan.output_dir / "workspace"
        fp_dir = workspace / "converted"

        fp_dir.mkdir(parents=True, exist_ok=True)

        fp_gguf = fp_dir / "model-f16.gguf"

        # --------------------------------------------------------------
        # Step 1: HF → GGUF
        # --------------------------------------------------------------

        if not fp_gguf.exists():
            convert_cmd = [
                os.environ.get("PYTHON", "python"),
                str(converter),
                str(self.model_dir),
                "--outfile",
                str(fp_gguf),
                "--outtype",
                "f16",
            ]

            self._run_command(
                convert_cmd,
                cwd=llama_cpp_dir,
            )

        self.fp_gguf = fp_gguf

        # --------------------------------------------------------------
        # Step 2: GGUF → quantized GGUF
        # --------------------------------------------------------------

        quant_type = self._quant_type(plan)

        output_dir = plan.output_dir / "artifact"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_dir
            / f"{self._model_slug(plan.model_id)}-{quant_type}.gguf"
        )

        quantize_bin = self._find_quantize_binary()

        if quantize_bin is None:
            raise RuntimeError(
                "llama-quantize binary not found."
            )

        quantize_cmd = [
            quantize_bin,
            str(fp_gguf),
            str(output_file),
            quant_type,
        ]

        self._run_command(quantize_cmd)

        self.quantized_gguf = output_file

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, plan: OptimizationPlan) -> str:
        """
        Return the generated GGUF artifact path.
        """

        if self.quantized_gguf is None:
            raise RuntimeError(
                "Quantized GGUF artifact does not exist."
            )

        if not self.quantized_gguf.exists():
            raise RuntimeError(
                f"Quantized GGUF file missing: {self.quantized_gguf}"
            )

        return str(self.quantized_gguf)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        plan: OptimizationPlan,
        artifact_path: str,
    ) -> BenchmarkResult:
        """
        Benchmarking will eventually invoke llama.cpp's inference
        runtime.

        For now we verify the artifact exists and expose the path.
        """

        artifact = Path(artifact_path)

        if not artifact.exists():
            return BenchmarkResult(
                tokens_per_second=None,
                method="gguf-bench",
                notes=[
                    "GGUF artifact does not exist.",
                ],
            )

        return BenchmarkResult(
            tokens_per_second=None,
            method="gguf-bench",
            notes=[
                "GGUF quantization completed.",
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

    @staticmethod
    def _quant_type(plan: OptimizationPlan) -> str:
        """
        Convert OCQ's requested bit width into a GGUF quantization type.
        """

        bits = plan.quant_config.bits

        mapping = {
            2: "Q2_K",
            3: "Q3_K_M",
            4: "Q4_K_M",
            5: "Q5_K_M",
            6: "Q6_K",
            8: "Q8_0",
        }

        try:
            return mapping[bits]
        except KeyError as exc:
            raise ValueError(
                f"No GGUF quantization mapping for {bits}-bit."
            ) from exc

    @staticmethod
    def _find_quantize_binary() -> str | None:
        """
        Locate llama-quantize executable.
        """

        for name in (
            "llama-quantize",
            "llama-quantize.exe",
        ):
            path = shutil.which(name)

            if path:
                return path

        return None

    @staticmethod
    def _find_llama_cpp_dir() -> Path:
        """
        Locate llama.cpp source directory.

        Preferred:

            OCQ_LLAMA_CPP=/path/to/llama.cpp
        """

        env_path = os.environ.get("OCQ_LLAMA_CPP")

        if env_path:
            path = Path(env_path).expanduser().resolve()

            if path.exists():
                return path

            raise RuntimeError(
                f"OCQ_LLAMA_CPP does not exist: {path}"
            )

        raise RuntimeError(
            "llama.cpp source directory not configured. "
            "Set OCQ_LLAMA_CPP=/path/to/llama.cpp"
        )

    @staticmethod
    def _run_command(
        command: list[str],
        *,
        cwd: Path | None = None,
    ) -> None:
        """
        Execute a subprocess and surface useful errors.
        """

        print(
            "Running:",
            " ".join(command),
        )

        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Executable not found: {command[0]}"
            ) from exc

        if completed.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code "
                f"{completed.returncode}: "
                f"{' '.join(command)}"
            )
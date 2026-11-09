"""Load models from a local HuggingFace-format folder.

Works with anything that saves a standard transformers layout: Unsloth merged exports,
Axolotl `output_dir`, TRL `save_pretrained`, vanilla PyTorch training checkpoints, etc.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ocq.model_size import Tensor
from ocq.types import OcqError


def parse_model_ref(ref: str) -> tuple[str, Path | None]:
    """Return (display_id, local_dir). HuggingFace ids pass through with local_dir=None."""
    if ref.startswith("ollama:"):
        from ocq.discovery import ollama_blob  # discovery imports this module

        name = ref.removeprefix("ollama:")
        return f"ollama/{name}", ollama_blob(name)
    path = Path(ref).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        resolved = path.resolve()
        return f"local/{resolved.name}", resolved
    if path.is_dir() and gguf_files(path) and not (path / "config.json").is_file():
        resolved = path.resolve()
        return f"local/{resolved.name}", resolved
    if path.is_dir():
        problem = validate_hf_folder(path)
        if problem:
            raise OcqError(problem)
        resolved = path.resolve()
        return f"local/{resolved.name}", resolved

    looks_local = ref.startswith((".", "/", "\\", "~")) or (len(ref) > 1 and ref[1] == ":")
    if looks_local:
        raise OcqError(f"Model folder not found: {ref}")
    return ref, None


def validate_hf_folder(path: Path) -> str | None:
    if (path / "adapter_config.json").is_file() and not (path / "config.json").is_file():
        return (
            f"{path} holds only a LoRA adapter, not a full model. Merge it into its base model "
            "first (PEFT merge_and_unload() or Unsloth save_pretrained_merged()), then save that."
        )
    if not (path / "config.json").is_file():
        return (
            f"{path} is not a HuggingFace-style model folder (missing config.json). "
            "Save the full model with save_pretrained()."
        )
    files = weight_files(path)
    missing = [file.name for file in files if not file.is_file()]
    if missing:
        try:
            index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            size = f", {index['metadata']['total_size'] / 1024**3:.2f} GB in all"
        except (OSError, ValueError, KeyError, TypeError):
            size = ""
        shown = ", ".join(missing[:3]) + (
            f" and {len(missing) - 3} more" if len(missing) > 3 else ""
        )
        return (
            f"{path} is missing {len(missing)} of its {len(files)} weight files ({shown}{size}, "
            "as model.safetensors.index.json lists them): only the config and the tokenizer "
            "are there. Copy the weight files into the folder."
        )
    if not files:
        if list(path.glob("*.bin")):
            return (
                f"{path} only has PyTorch .bin weights. Re-save it as safetensors: load it with "
                "from_pretrained() and call save_pretrained() again."
            )
        return f"{path} has no *.safetensors weight files."
    return None


def gguf_files(folder: Path) -> list[Path]:
    """The GGUF model files of a folder. mmproj files are vision projectors, not models."""
    return sorted(f for f in folder.glob("*.gguf") if not f.name.lower().startswith("mmproj"))


def weight_files(folder: Path) -> list[Path]:
    """The files transformers would load, not every safetensors file lying in the folder."""
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError) as exc:
            raise OcqError(f"Could not read {index}") from exc
        return [folder / name for name in sorted(names)]

    single = folder / "model.safetensors"
    if single.is_file():
        return [single]
    return sorted(f for f in folder.glob("*.safetensors") if not f.name.startswith("adapter"))


def has_tokenizer(folder: Path) -> bool:
    names = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")
    return any((folder / name).is_file() for name in names)


NO_TOKENIZER = (
    "{folder} has the model weights but no tokenizer files. Save the tokenizer into the same "
    "folder: tokenizer.save_pretrained(folder)."
)


def load_local_config(folder: Path) -> dict:
    try:
        return json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OcqError(f"Could not read config.json in {folder}") from exc


def read_safetensors_header(file: Path) -> dict:
    """A safetensors file starts with the header length (8 bytes) and then a JSON header."""
    try:
        with open(file, "rb") as handle:
            length = int.from_bytes(handle.read(8), "little")
            if not 0 < length < 500_000_000:
                raise ValueError("bad header length")
            header = json.loads(handle.read(length))
    except (OSError, ValueError) as exc:
        raise OcqError(f"Could not read safetensors file {file}") from exc
    header.pop("__metadata__", None)
    return header


# What QuantPilot writes next to a model it made: the repo it came from and the method.
PROVENANCE = "quantpilot.json"


def provenance(folder: Path) -> dict:
    try:
        made = json.loads((folder / PROVENANCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return made if isinstance(made, dict) else {}


def compressed_method(folder: Path) -> str:
    """Which method made a compressed-tensors folder: llm-compressor saves AWQ, GPTQ and NVFP4
    alike. From QuantPilot's record, else llm-compressor's recipe.yaml, else the name
    QuantPilot gives the folder (<model>-awq-4bit, -gptq-8bit, -nvfp4). "" when unknown."""
    method = str(provenance(folder).get("method", "")).lower()
    if method in ("awq", "gptq", "nvfp4"):
        return method
    try:
        recipe = (folder / "recipe.yaml").read_text(encoding="utf-8")
    except OSError:
        recipe = ""
    for modifier, method in (("AWQModifier", "awq"), ("GPTQModifier", "gptq")):
        if modifier in recipe:
            return method
    if "NVFP4" in recipe:
        return "nvfp4"
    named = re.search(r"-(awq|gptq)-\d+bit$|-(nvfp4)(?:a16)?$", folder.name.lower())
    return next((group for group in named.groups() if group), "") if named else ""


def fetch_local_model_files(folder: Path) -> tuple[dict, list[Tensor], dict[str, str]]:
    """config.json, every tensor's shape, and every tensor's stored type."""
    from ocq.model_analyzer import MADE_WITH, quantization_config

    config = load_local_config(folder)
    settings = quantization_config(config)
    if str(settings.get("quant_method", "")) == "compressed-tensors":
        method = compressed_method(folder)
        if method:
            settings[MADE_WITH] = method
    tensors: list[Tensor] = []
    dtypes: dict[str, str] = {}

    for file in weight_files(folder):
        for name, info in read_safetensors_header(file).items():
            tensors.append((name, tuple(info["shape"])))
            dtypes[name] = info["dtype"]

    if not tensors:
        raise OcqError(f"No tensors found in {folder}")
    return config, tensors, dtypes


def slug_local(model_id: str) -> str:
    """Filesystem-safe slug for artifact names."""
    cleaned = model_id.replace("\\", "/").strip("/")
    return re.sub(r"[^\w.\-]+", "--", cleaned)


def cached_hf_model(repo: str) -> Path | None:
    """The folder of a Hugging Face repo in the Hugging Face cache, when its weights and
    tokenizer are all there (often only config.json is: QuantPilot reads it to size a model)."""
    from huggingface_hub import try_to_load_from_cache

    from ocq.discovery import hf_cache_dirs  # discovery imports this module

    config = None
    for cache in hf_cache_dirs():  # read now: the library reads its setting once, on import
        try:
            config = try_to_load_from_cache(repo, "config.json", cache_dir=cache)
        except Exception:  # not a repo id
            return None
        if isinstance(config, str):
            break
    if not isinstance(config, str):
        return None
    folder = Path(config).parent
    try:
        files = weight_files(folder)
    except OcqError:
        return None
    if not files or not all(file.is_file() for file in files) or not has_tokenizer(folder):
        return None
    return folder


@dataclass(frozen=True)
class ModelCheck:
    """What a folder (or file) chosen as a model of your own holds, for people to read."""

    ok: bool  # it can be added: QuantPilot can read, check, quantize and benchmark it
    label: str  # what it is: "LlamaForCausalLM, 0.27 GB" or "2 GGUF files, 5.1 GB"
    lines: tuple[tuple[bool, str], ...]  # each thing checked: (fine, what)
    folder: Path | None = None  # what to add to the model folders


def check_model(path: Path) -> ModelCheck:
    """Whether path is a model QuantPilot can work with: a folder saved with save_pretrained
    (config.json, safetensors weights, a tokenizer), a folder of GGUF files, or a .gguf file.
    Says what is missing, and what to do about a LoRA adapter or .bin weights."""
    path = path.expanduser()
    if not path.exists():
        return ModelCheck(False, "", ((False, "Nothing is there"),))
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            return ModelCheck(False, "", ((False, "Choose a model folder or a .gguf file"),))
        size = path.stat().st_size / 1024**3
        return ModelCheck(True, f"GGUF file, {size:.2f} GB", ((True, "A GGUF file"),), path.parent)
    ggufs = gguf_files(path)
    if ggufs and not (path / "config.json").is_file():
        size = sum(f.stat().st_size for f in ggufs) / 1024**3
        what = f"{len(ggufs)} GGUF file{'s' * (len(ggufs) > 1)}, {size:.2f} GB"
        return ModelCheck(True, what, ((True, what),), path)
    if (path / "adapter_config.json").is_file() and not (path / "config.json").is_file():
        return ModelCheck(False, "", (
            (False, "A LoRA adapter, not a whole model"),
            (False, "Merge it into its base model first (PEFT merge_and_unload(), or Unsloth "
                    "save_pretrained_merged()), then save that"),
        ))  # fmt: skip
    lines: list[tuple[bool, str]] = []
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        kind = ", ".join(config.get("architectures") or []) or config.get("model_type", "")
        lines.append((True, f"config.json ({kind or 'a model'})"))
    except (OSError, ValueError):
        lines.append((False, "No readable config.json: save the model with save_pretrained()"))
        kind = ""
    try:
        files = weight_files(path)
    except OcqError:
        files = []
    present = [f for f in files if f.is_file()]
    size = sum(f.stat().st_size for f in present) / 1024**3
    if files and len(present) == len(files):
        count = f"{len(files)} file{'s' * (len(files) > 1)}"
        lines.append((True, f"Weights: {count}, {size:.2f} GB"))
    elif files:
        lines.append(
            (False, f"Weights: {len(files) - len(present)} of {len(files)} files are missing")
        )
    elif list(path.glob("*.bin")):
        lines.append((False, "Only .bin weights: load it and save_pretrained() it again "
                             "(safetensors)"))  # fmt: skip
    else:
        lines.append((False, "No weights (*.safetensors)"))
    if has_tokenizer(path):
        lines.append((True, "Tokenizer"))
    else:
        lines.append((False, "No tokenizer: tokenizer.save_pretrained() into the same folder"))
    ok = all(fine for fine, _ in lines)
    return ModelCheck(
        ok, f"{kind}, {size:.2f} GB" if ok else "", tuple(lines), path if ok else None
    )

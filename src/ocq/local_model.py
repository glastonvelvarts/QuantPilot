"""Load models from a local HuggingFace-format folder.

Works with anything that saves a standard transformers layout: Unsloth merged exports,
Axolotl `output_dir`, TRL `save_pretrained`, vanilla PyTorch training checkpoints, etc.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ocq.model_size import Tensor
from ocq.types import OcqError


def parse_model_ref(ref: str) -> tuple[str, Path | None]:
    """Return (display_id, local_dir). HuggingFace ids pass through with local_dir=None."""
    path = Path(ref).expanduser()
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
    if not weight_files(path):
        if list(path.glob("*.bin")):
            return (
                f"{path} only has PyTorch .bin weights. Re-save it as safetensors: load it with "
                "from_pretrained() and call save_pretrained() again."
            )
        return f"{path} has no *.safetensors weight files."
    return None


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


def fetch_local_model_files(folder: Path) -> tuple[dict, list[Tensor], dict[str, int]]:
    """config.json, every tensor's shape, and parameter counts per stored type."""
    config = load_local_config(folder)
    tensors: list[Tensor] = []
    parameters: dict[str, int] = {}

    for file in weight_files(folder):
        for name, info in read_safetensors_header(file).items():
            shape = tuple(info["shape"])
            tensors.append((name, shape))
            parameters[info["dtype"]] = parameters.get(info["dtype"], 0) + math.prod(shape)

    if not tensors:
        raise OcqError(f"No tensors found in {folder}")
    return config, tensors, parameters


def local_hf_meta(parameters: dict[str, int]) -> dict:
    """Same shape as the HuggingFace API answer, so the rest of the code treats both alike."""
    return {
        "safetensors": {"parameters": parameters, "total": sum(parameters.values())},
        "pipeline_tag": "local",
    }


def slug_local(model_id: str) -> str:
    """Filesystem-safe slug for artifact names."""
    cleaned = model_id.replace("\\", "/").strip("/")
    return re.sub(r"[^\w.\-]+", "--", cleaned)

"""Find the models already on this computer.

Where each app keeps them:
  Ollama          ~/.ollama/models (or $OLLAMA_MODELS): a manifest per name:tag lists blobs by
                  sha256; the model layer is a GGUF file. Newer versions also write
                  manifests-v2/, and a manifest can be a list with one child per runner
                  (llama.cpp GGUF, MLX safetensors).
  LM Studio       ~/.lmstudio/models/<publisher>/<model>/ (moved with ~/.lmstudio-home-pointer
                  or the downloadsFolder setting; older versions ~/.cache/lm-studio/models)
  Hugging Face    ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<commit>/, also
                  where llama.cpp's -hf downloads go; often holds only config.json, fetched
                  to size a model without downloading it
  llama.cpp       its older download cache: plain GGUF files
Nothing here loads a model or uses the network: every entry is read from files on disk.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from ocq.gguf_file import SPLIT_PART, read_local
from ocq.local_model import gguf_files, weight_files
from ocq.types import OcqError

OLLAMA = "Ollama"
LM_STUDIO = "LM Studio"
HF_CACHE = "Hugging Face cache"
LLAMA_CPP_CACHE = "llama.cpp cache"
FOLDER = "folder"
SOURCES = (OLLAMA, LM_STUDIO, HF_CACHE, LLAMA_CPP_CACHE)

GGUF = "GGUF"
SAFETENSORS = "safetensors"


@dataclass(frozen=True)
class LocalModel:
    label: str  # how the app names it
    ref: str  # what to pass to ocq inspect / feasibility
    source: str
    path: Path  # the file or folder holding the weights
    format: str
    quantization: str
    size_bytes: int  # weights on disk
    complete: bool  # every weight file is there
    note: str = ""


# Ollama ------------------------------------------------------------------------------------

OLLAMA_PUBLIC_HOSTS = ("registry.ollama.ai", "ollama.com")
MANIFEST_LIST = "application/vnd.ollama.manifest.list.v2+json"
MODEL_LAYER = "application/vnd.ollama.image.model"
TENSOR_LAYER = "application/vnd.ollama.image.tensor"
CONFIG_LAYER = "application/vnd.ollama.image.json"
EXTRA_LAYERS = {
    "application/vnd.ollama.image.projector": "vision projector",
    "application/vnd.ollama.image.draft": "draft model",
    "application/vnd.ollama.image.adapter": "LoRA adapter",
}
DIGEST = re.compile(r"^sha256[:-][0-9a-fA-F]{64}$")
# llama.cpp runs GGUF; QuantPilot sizes GGUF for it, so that child of a list comes first.
GGUF_RUNNERS = ("llamacpp", "ggml")
# Linux installs that run Ollama as a service keep models in the service user's home.
OLLAMA_SERVICE_DIR = Path("/usr/share/ollama/.ollama/models")


def ollama_dirs() -> list[Path]:
    configured = os.environ.get("OLLAMA_MODELS")
    dirs = [Path(configured).expanduser()] if configured else [Path.home() / ".ollama" / "models"]
    dirs.append(OLLAMA_SERVICE_DIR)
    return existing(dirs)


def ollama_name(host: str, namespace: str, model: str, tag: str) -> str:
    if host.lower() in OLLAMA_PUBLIC_HOSTS:
        return f"{model}:{tag}" if namespace == "library" else f"{namespace}/{model}:{tag}"
    return f"{host}/{namespace}/{model}:{tag}"


def ollama_manifests(root: Path) -> dict[str, Path]:
    """Every name:tag with its manifest file. manifests-v2 first: Ollama reads it first too."""
    found: dict[str, Path] = {}
    for tree in ("manifests-v2", "manifests"):
        base = root / tree
        if not base.is_dir():
            continue
        for file in sorted(base.glob("*/*/*/*")):
            if file.is_file():
                host, namespace, model, tag = file.relative_to(base).parts
                host = host.replace("%3A", ":").replace("%3a", ":")
                found.setdefault(ollama_name(host, namespace, model, tag), file)
    return found


def blob_path(root: Path, digest: str) -> Path | None:
    if not DIGEST.match(digest or ""):
        return None
    return root / "blobs" / digest.replace(":", "-")


def blob_ok(root: Path, layer: dict) -> bool:
    blob = blob_path(root, layer.get("digest", ""))
    return blob is not None and blob.is_file() and blob.stat().st_size == layer.get("size")


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise OcqError(f"Could not read {path}") from exc
    if not isinstance(data, dict):
        raise OcqError(f"{path} is not a manifest")
    return data


def runnable_manifest(root: Path, data: dict) -> dict | None:
    """A plain manifest as it is; from a list, the downloaded child, GGUF first."""
    if data.get("mediaType") != MANIFEST_LIST:
        return data
    children = sorted(
        data.get("manifests", []), key=lambda child: child.get("runner") not in GGUF_RUNNERS
    )
    for child in children:
        if child.get("layers"):
            return child
        blob = blob_path(root, child.get("digest", ""))
        if blob is not None and blob.is_file():
            return read_json(blob)
    return None


def ollama_model(root: Path, name: str, manifest_file: Path) -> LocalModel:
    manifest = runnable_manifest(root, read_json(manifest_file))
    if manifest is None:
        return LocalModel(
            name, f"ollama:{name}", OLLAMA, manifest_file, "?", "", 0, False, "no variant on disk"
        )
    layers = manifest.get("layers", [])
    model = [layer for layer in layers if layer.get("mediaType") == MODEL_LAYER]
    tensors = [layer for layer in layers if layer.get("mediaType") == TENSOR_LAYER]
    weights = model or tensors
    kinds = {layer.get("mediaType") for layer in layers}
    extras = sorted(EXTRA_LAYERS[kind] for kind in kinds if kind in EXTRA_LAYERS)
    note = f"also {', '.join(extras)}" if extras else ""
    size = sum(int(layer.get("size", 0)) for layer in weights)
    complete = bool(weights) and all(blob_ok(root, layer) for layer in weights)
    if model:
        blob = blob_path(root, model[0].get("digest", "")) or manifest_file
        quant = gguf_quantization(blob) if complete else ""
        return LocalModel(name, f"ollama:{name}", OLLAMA, blob, GGUF, quant, size, complete, note)
    if tensors:
        mlx = "MLX " if manifest.get("runner") == "mlx" else ""
        return LocalModel(
            name,
            f"ollama:{name}",
            OLLAMA,
            manifest_file,
            f"{mlx}{SAFETENSORS}",
            "",
            size,
            complete,
            note or "QuantPilot cannot read this layout yet",
        )
    return LocalModel(
        name, f"ollama:{name}", OLLAMA, manifest_file, "?", "", 0, False, "no weights"
    )


def ollama_models(roots: list[Path] | None = None) -> list[LocalModel]:
    models: list[LocalModel] = []
    for root in ollama_dirs() if roots is None else roots:
        for name, manifest_file in ollama_manifests(root).items():
            models.append(ollama_model(root, name, manifest_file))
    return models


def ollama_blob(name: str) -> Path:
    """The GGUF file behind an Ollama name, for example qwen3:0.6b or qwen3 (tag latest)."""
    wanted = name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"
    for model in ollama_models():
        if model.label == wanted:
            if model.format != GGUF:
                raise OcqError(f"Ollama's {wanted} is {model.format}; QuantPilot reads GGUF.")
            if not model.complete:
                raise OcqError(f"Ollama's {wanted} is not fully downloaded: ollama pull {wanted}")
            return model.path
    raise OcqError(f"Ollama has no model {wanted}. ocq models lists the ones on this PC.")


# LM Studio ---------------------------------------------------------------------------------


def lmstudio_dirs() -> list[Path]:
    home = Path.home() / ".lmstudio"
    pointer = Path.home() / ".lmstudio-home-pointer"
    if pointer.is_file():
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            home = Path(text).expanduser()
    dirs = []
    try:
        folder = json.loads((home / "settings.json").read_text(encoding="utf-8")).get(
            "downloadsFolder"
        )
        if isinstance(folder, str) and folder:
            dirs.append(Path(folder).expanduser())
    except (OSError, ValueError, AttributeError):
        pass
    dirs += [home / "models", Path.home() / ".cache" / "lm-studio" / "models"]
    return existing(dirs)


def lmstudio_models(roots: list[Path] | None = None) -> list[LocalModel]:
    models: list[LocalModel] = []
    for root in lmstudio_dirs() if roots is None else roots:
        for folder in sorted(p for p in root.glob("*/*") if p.is_dir()):
            label = f"{folder.parent.name}/{folder.name}"
            models += folder_models(folder, label, LM_STUDIO)
    return models


# Hugging Face cache and llama.cpp ----------------------------------------------------------


def hf_cache_dirs() -> list[Path]:
    """huggingface_hub's own order; llama.cpp's -hf downloads use the same cache."""
    env = os.environ
    if env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE"):
        dirs = [Path(env.get("HF_HUB_CACHE") or env["HUGGINGFACE_HUB_CACHE"]).expanduser()]
    elif env.get("HF_HOME"):
        dirs = [Path(env["HF_HOME"]).expanduser() / "hub"]
    elif env.get("XDG_CACHE_HOME"):
        dirs = [Path(env["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"]
    else:
        dirs = [Path.home() / ".cache" / "huggingface" / "hub"]
    if env.get("LLAMA_CACHE"):
        dirs.append(Path(env["LLAMA_CACHE"]).expanduser())
    return existing(dirs)


def snapshot_dir(repo: Path) -> Path | None:
    """The revision refs/main points to, else the newest snapshot."""
    try:
        commit = (repo / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    if commit and (repo / "snapshots" / commit).is_dir():
        return repo / "snapshots" / commit
    snapshots = [p for p in (repo / "snapshots").glob("*") if p.is_dir()]
    return max(snapshots, key=lambda p: p.stat().st_mtime) if snapshots else None


def hf_cache_models(roots: list[Path] | None = None) -> list[LocalModel]:
    models: list[LocalModel] = []
    for root in hf_cache_dirs() if roots is None else roots:
        for repo in sorted(root.glob("models--*")):
            snapshot = snapshot_dir(repo)
            if snapshot is None:
                continue
            repo_id = repo.name.removeprefix("models--").replace("--", "/", 1)
            found = folder_models(snapshot, repo_id, HF_CACHE, ref=repo_id)
            if any(repo.glob("blobs/*.incomplete")):
                found = [m if m.complete else replace(m, note="download unfinished") for m in found]
            models += found
    return models


def llama_cpp_cache_dirs() -> list[Path]:
    """Before llama.cpp moved to the Hugging Face cache, -hf saved plain GGUF files here."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return existing([base / "llama.cpp"])


def llama_cpp_cache_models(roots: list[Path] | None = None) -> list[LocalModel]:
    models: list[LocalModel] = []
    for root in llama_cpp_cache_dirs() if roots is None else roots:
        models += gguf_models(root, "", LLAMA_CPP_CACHE)
    return models


# Any folder --------------------------------------------------------------------------------

MAX_DEPTH = 4


def folder_tree_models(root: Path) -> list[LocalModel]:
    """Every model folder and GGUF file under root, a few levels deep."""
    models: list[LocalModel] = []
    pending = [(root, 0)]
    while pending:
        folder, depth = pending.pop()
        label = str(folder.relative_to(root)) if folder != root else folder.name
        found = folder_models(folder, label, FOLDER)
        models += found
        if depth < MAX_DEPTH:
            pending += [
                (child, depth + 1)
                for child in sorted(folder.iterdir(), reverse=True)
                if child.is_dir() and not child.name.startswith(".")
            ]
    return models


def folder_models(
    folder: Path, label: str, source: str, ref: str | None = None
) -> list[LocalModel]:
    """The models one folder holds: its GGUF files, or a transformers-style checkpoint."""
    if gguf_files(folder):
        return gguf_models(folder, label, source)
    if not (folder / "config.json").is_file():
        return []
    try:
        files = weight_files(folder)
    except OcqError:
        files = []
    present = [f for f in files if f.is_file()]
    complete = bool(files) and len(present) == len(files)
    note = "" if complete else "only config.json" if not present else "some weight files missing"
    return [
        LocalModel(
            label,
            ref or tilde(folder),
            source,
            folder,
            SAFETENSORS,
            safetensors_quantization(folder),
            sum(f.stat().st_size for f in present),
            complete,
            note,
        )
    ]


def gguf_models(folder: Path, label: str, source: str) -> list[LocalModel]:
    """One entry per GGUF variant; the parts of a split file add up to one."""
    groups: dict[str, list[Path]] = {}
    for file in gguf_files(folder):
        groups.setdefault(SPLIT_PART.sub(".gguf", file.name), []).append(file)
    models = []
    for name, files in sorted(groups.items()):
        files.sort()
        first = files[0]
        split = SPLIT_PART.search(first.name)
        complete = not split or len(files) == int(split.group(2))
        quant = gguf_quantization(first)
        models.append(
            LocalModel(
                f"{label}/{name}" if label else name,
                tilde(first),
                source,
                first,
                GGUF,
                quant,
                sum(f.stat().st_size for f in files),
                complete and quant != "unreadable",
                "" if complete else f"{len(files)} of {int(split.group(2))} parts",
            )
        )
    return models


# Reading what a model is -------------------------------------------------------------------


def gguf_quantization(path: Path) -> str:
    """The file type its header names, for example Q4_K_M."""
    try:
        return read_local(path).file_type or "?"
    except OcqError:
        return "unreadable"


def safetensors_quantization(folder: Path) -> str:
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "?"
    quant = config.get("quantization_config") or {}
    if isinstance(quant, dict) and quant:
        method = quant.get("quant_method") or quant.get("quant_algo") or "quantized"
        bits = quant.get("bits")
        return f"{method} {bits}-bit" if bits else str(method)
    if isinstance(config.get("quantization"), dict):  # MLX
        return f"mlx {config['quantization'].get('bits', '?')}-bit"
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return str(config.get("torch_dtype") or config.get("dtype") or text.get("torch_dtype") or "?")


# All of them -------------------------------------------------------------------------------


def find_models(extra_dirs: list[Path] | None = None) -> list[LocalModel]:
    models = ollama_models() + lmstudio_models() + hf_cache_models() + llama_cpp_cache_models()
    for folder in extra_dirs or []:
        if not folder.is_dir():
            raise OcqError(f"Folder not found: {folder}")
        models += folder_tree_models(folder.expanduser().resolve())
    return models


def existing(dirs: list[Path]) -> list[Path]:
    unique: list[Path] = []
    for folder in dirs:
        if folder.is_dir() and folder.resolve() not in [d.resolve() for d in unique]:
            unique.append(folder)
    return unique


def tilde(path: Path) -> str:
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def searched_locations(extra_dirs: list[Path] | None = None) -> list[tuple[str, Path]]:
    """Where find_models looked, for telling the user."""
    places = [(OLLAMA, d) for d in ollama_dirs()]
    places += [(LM_STUDIO, d) for d in lmstudio_dirs()]
    places += [(HF_CACHE, d) for d in hf_cache_dirs()]
    places += [(LLAMA_CPP_CACHE, d) for d in llama_cpp_cache_dirs()]
    return places + [(FOLDER, d) for d in extra_dirs or []]

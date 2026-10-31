"""Finding the models already on this PC. The Ollama manifests are real (registry.ollama.ai,
saved 2026-10-31): smollm2:135m, and qwen3.5:0.8b, a manifest list with a llama.cpp child and an
MLX child. The model blobs here are tiny GGUF files padded to the size the manifest names; the
real smollm2:135m blob was checked by hand with ocq verify (llama.cpp's memory matched exactly)."""

import gzip
import json
import os
from pathlib import Path

import pytest
from conftest import DATA
from test_gguf_file import gguf_bytes
from typer.testing import CliRunner

from ocq import discovery
from ocq.cli import app
from ocq.model_analyzer import resolve_model
from ocq.types import OcqError

OLLAMA_DATA = DATA / "ollama"


@pytest.fixture(autouse=True)
def empty_home(tmp_path, monkeypatch):
    """A PC with no models, unless a test puts some there."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in (
        "OLLAMA_MODELS",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "XDG_CACHE_HOME",
        "LLAMA_CACHE",
        "LOCALAPPDATA",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(discovery, "OLLAMA_SERVICE_DIR", tmp_path / "no-service")
    return home


def tiny_gguf(file_type: int = 7) -> bytes:
    """A valid two-layer GGUF (file type 7 is Q8_0)."""
    q8 = 8
    tensors = [("token_embd.weight", (64, 100), q8, 64 * 100 // 32 * 34)]
    for layer in range(2):
        tensors.append((f"blk.{layer}.attn_k.weight", (64, 32), q8, 64 * 32 // 32 * 34))
    settings = {
        "general.architecture": "llama",
        "general.file_type": file_type,
        "llama.block_count": 2,
        "llama.embedding_length": 64,
        "llama.attention.head_count": 2,
        "llama.attention.head_count_kv": 1,
        "llama.context_length": 2048,
    }
    return gguf_bytes(settings, tensors)


def write_blob(root: Path, digest: str, data: bytes = b"", size: int | None = None) -> Path:
    blob = root / "blobs" / digest.replace(":", "-")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    if size is not None:
        os.truncate(blob, size)  # zeros after the header, as many bytes as the manifest says
    return blob


def ollama_store(root: Path, tree: str, name: str, manifest: bytes) -> Path:
    model, tag = name.split(":")
    host = "registry.ollama.ai" if tree == "manifests" else "ollama.com"
    path = root / tree / host / "library" / model / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(manifest)
    return path


def smollm2_store(root: Path) -> dict:
    raw = (OLLAMA_DATA / "smollm2-135m.json").read_bytes()
    manifest = json.loads(raw)
    ollama_store(root, "manifests", "smollm2:135m", raw)
    for layer in [manifest["config"], *manifest["layers"]]:
        if layer["mediaType"] == discovery.MODEL_LAYER:
            write_blob(root, layer["digest"], tiny_gguf(), layer["size"])
        else:
            write_blob(root, layer["digest"], b"x", layer["size"])
    return manifest


def test_an_ollama_model_is_found_from_its_manifest(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    smollm2_store(root)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))

    (model,) = discovery.ollama_models()
    assert (model.label, model.ref, model.source) == (
        "smollm2:135m",
        "ollama:smollm2:135m",
        "Ollama",
    )
    assert (model.format, model.quantization) == ("GGUF", "Q8_0")  # from the blob's own header
    assert model.size_bytes == 270_885_952  # the model layer, as the manifest says
    assert model.complete
    assert model.path.name.startswith("sha256-f535f83ec568")


def test_ollama_names_are_read_like_any_gguf_file(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    smollm2_store(root)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))

    snap = resolve_model("ollama:smollm2:135m", None)
    assert snap.model_id == "ollama/smollm2:135m"
    assert snap.facts.quantization.value == "GGUF Q8_0"
    assert snap.facts.parameters.value == 64 * 100 + 2 * 64 * 32
    with pytest.raises(OcqError, match="no model qwen3:latest"):
        resolve_model("ollama:qwen3", None)


def test_an_unfinished_ollama_download_is_not_used(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    manifest = smollm2_store(root)
    model_layer = next(
        lay for lay in manifest["layers"] if lay["mediaType"] == discovery.MODEL_LAYER
    )
    write_blob(root, model_layer["digest"], tiny_gguf())  # shorter than the manifest says
    monkeypatch.setenv("OLLAMA_MODELS", str(root))

    (model,) = discovery.ollama_models()
    assert not model.complete
    with pytest.raises(OcqError, match="not fully downloaded"):
        discovery.ollama_blob("smollm2:135m")


def test_a_manifest_list_gives_the_llama_cpp_child(tmp_path):
    root = tmp_path / "ollama"
    ollama_store(
        root, "manifests-v2", "qwen3.5:0.8b", (OLLAMA_DATA / "qwen3.5-0.8b-list.json").read_bytes()
    )
    child = (OLLAMA_DATA / "qwen3.5-0.8b-llamacpp.json").read_bytes()
    mlx = gzip.decompress((OLLAMA_DATA / "qwen3.5-0.8b-mlx.json.gz").read_bytes())
    listing = json.loads((OLLAMA_DATA / "qwen3.5-0.8b-list.json").read_bytes())
    digests = {c["runner"]: c["digest"] for c in listing["manifests"]}
    write_blob(root, digests["llamacpp"], child)
    write_blob(root, digests["mlx"], mlx)
    model_layer = next(
        lay for lay in json.loads(child)["layers"] if lay["mediaType"] == discovery.MODEL_LAYER
    )
    write_blob(root, model_layer["digest"], tiny_gguf(), model_layer["size"])

    (model,) = discovery.ollama_models([root])
    assert model.label == "qwen3.5:0.8b"  # manifests-v2 names the public host ollama.com
    assert (model.format, model.size_bytes, model.complete) == ("GGUF", 811_843_520, True)
    assert model.note == "also draft model, vision projector"


def test_an_mlx_only_ollama_model_is_listed_but_not_read(tmp_path):
    root = tmp_path / "ollama"
    ollama_store(
        root, "manifests-v2", "qwen3.5:0.8b", (OLLAMA_DATA / "qwen3.5-0.8b-list.json").read_bytes()
    )
    mlx = gzip.decompress((OLLAMA_DATA / "qwen3.5-0.8b-mlx.json.gz").read_bytes())
    listing = json.loads((OLLAMA_DATA / "qwen3.5-0.8b-list.json").read_bytes())
    write_blob(root, next(c["digest"] for c in listing["manifests"] if c["runner"] == "mlx"), mlx)

    (model,) = discovery.ollama_models([root])
    assert model.format == "MLX safetensors"
    assert model.size_bytes == sum(
        lay["size"]
        for lay in json.loads(mlx)["layers"]
        if lay["mediaType"] == discovery.TENSOR_LAYER
    )
    assert not model.complete  # the tensor blobs are not there
    assert "cannot read" in model.note


@pytest.mark.parametrize(
    ("parts", "name"),
    [
        (("registry.ollama.ai", "library", "qwen3", "0.6b"), "qwen3:0.6b"),
        (("ollama.com", "library", "qwen3", "latest"), "qwen3:latest"),
        (("registry.ollama.ai", "someone", "tuned", "q4"), "someone/tuned:q4"),
        (("localhost:5000", "team", "model", "v1"), "localhost:5000/team/model:v1"),
    ],
)
def test_ollama_names(parts, name):
    assert discovery.ollama_name(*parts) == name


def test_a_digest_cannot_point_outside_the_blobs_folder(tmp_path):
    assert discovery.blob_path(tmp_path, "sha256:../../etc/passwd") is None


def hf_repo(cache: Path, repo_id: str, files: dict[str, bytes | None]) -> Path:
    """A repo laid out like huggingface_hub does: snapshot files link to blobs.
    None makes a link whose blob is missing."""
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (repo / "blobs").mkdir()
    for i, (name, data) in enumerate(files.items()):
        blob = repo / "blobs" / f"blob{i}"
        if data is not None:
            blob.write_bytes(data)
        (snapshot / name).symlink_to(blob)
    return repo


def test_hugging_face_cache_tells_complete_models_from_config_only(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    config = json.dumps({"model_type": "llama", "torch_dtype": "bfloat16"}).encode()
    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    hf_repo(cache, "org/whole", {"config.json": config, "model.safetensors": b"w" * 100})
    hf_repo(cache, "org/settings-only", {"config.json": config})
    hf_repo(
        cache,
        "org/half",
        {
            "config.json": config,
            "model.safetensors.index.json": index.encode(),
            "model-1.safetensors": b"w" * 10,
            "model-2.safetensors": None,
        },
    )
    hf_repo(cache, "org/model-GGUF", {"model-Q4_K_M.gguf": tiny_gguf(15)})
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # the cache is HF_HOME/hub

    found = {m.label: m for m in discovery.hf_cache_models()}
    assert found["org/whole"].complete and found["org/whole"].size_bytes == 100
    assert found["org/whole"].ref == "org/whole"
    assert found["org/whole"].quantization == "bfloat16"
    assert (found["org/settings-only"].complete, found["org/settings-only"].note) == (
        False,
        "only config.json",
    )
    assert found["org/half"].note == "some weight files missing"
    gguf = found["org/model-GGUF/model-Q4_K_M.gguf"]  # where llama.cpp -hf downloads go
    assert (gguf.format, gguf.quantization, gguf.complete) == ("GGUF", "Q4_K_M", True)


def test_lm_studio_models_follow_its_home_pointer_and_download_folder(empty_home, tmp_path):
    lm_home = tmp_path / "lmstudio-moved"
    downloads = tmp_path / "big-disk"
    (empty_home / ".lmstudio-home-pointer").write_text(str(lm_home))
    lm_home.mkdir()
    (lm_home / "settings.json").write_text(json.dumps({"downloadsFolder": str(downloads)}))
    folder = downloads / "lmstudio-community" / "Qwen3-8B-GGUF"
    folder.mkdir(parents=True)
    (folder / "Qwen3-8B-Q4_K_M.gguf").write_bytes(tiny_gguf(15))
    (folder / "mmproj-Qwen3-8B-F16.gguf").write_bytes(tiny_gguf(1))  # vision part, not a model
    big = downloads / "unsloth" / "Big-GGUF"
    big.mkdir(parents=True)
    (big / "Big-Q8_0-00001-of-00002.gguf").write_bytes(tiny_gguf())

    found = {m.label: m for m in discovery.lmstudio_models()}
    assert set(found) == {
        "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        "unsloth/Big-GGUF/Big-Q8_0.gguf",
    }
    assert found["unsloth/Big-GGUF/Big-Q8_0.gguf"].note == "1 of 2 parts"
    assert not found["unsloth/Big-GGUF/Big-Q8_0.gguf"].complete


def test_any_folder_is_searched_a_few_levels_deep(tmp_path):
    deep = tmp_path / "models" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "x-Q8_0.gguf").write_bytes(tiny_gguf())
    hidden = tmp_path / "models" / ".trash"
    hidden.mkdir()
    (hidden / "y-Q8_0.gguf").write_bytes(tiny_gguf())

    found = discovery.find_models([tmp_path / "models"])
    assert [m.label for m in found] == ["a/b/x-Q8_0.gguf"]
    with pytest.raises(OcqError, match="Folder not found"):
        discovery.find_models([tmp_path / "nowhere"])


def test_ocq_models_lists_what_runs_and_hides_partial_downloads(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    smollm2_store(root)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    config = json.dumps({"model_type": "llama"}).encode()
    hf_repo(tmp_path / "hub", "org/settings-only", {"config.json": config})
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    result = CliRunner().invoke(app, ["models"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "ollama:smollm2:135m" in result.output
    assert "org/settings-only" not in result.output
    assert "1 partial download(s) hidden" in result.output
    assert "Not found on this PC: LM Studio, llama.cpp cache" in result.output

    result = CliRunner().invoke(app, ["models", "--all"], env={"COLUMNS": "200"})
    assert "org/settings-only" in result.output and "only config.json" in result.output


def test_ocq_models_rates_each_model_as_stored_offline(tmp_path, monkeypatch):
    from conftest import MLX_QWEN, QWEN, RTX_3090, nvidia_pc, real_model, write_safetensors

    from ocq import cli

    folders = tmp_path / "models"
    # An MLX model as LM Studio saves it: config.json and the safetensors, real headers.
    _, config, tensors, dtypes = real_model(MLX_QWEN)
    mlx = folders / "mlx-community" / "Qwen2.5-0.5B-Instruct-4bit"
    mlx.mkdir(parents=True)
    (mlx / "config.json").write_text(json.dumps(config))
    write_safetensors(mlx / "model.safetensors", tensors, dtypes)
    (folders / "tiny-Q8_0.gguf").write_bytes(tiny_gguf())
    (folders / "broken-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 8)  # a cut-off download
    damaged = folders / "damaged"
    damaged.mkdir()
    (damaged / "config.json").write_text(json.dumps(config))
    (damaged / "model.safetensors").write_bytes(b"not a safetensors file")
    # A complete download in the Hugging Face cache, real config and headers.
    _, qwen_config, qwen_tensors, qwen_dtypes = real_model(QWEN)
    write_safetensors(tmp_path / "weights", qwen_tensors, qwen_dtypes)
    hf_repo(
        tmp_path / "hub",
        QWEN,
        {
            "config.json": json.dumps(qwen_config).encode(),
            "model.safetensors": (tmp_path / "weights").read_bytes(),
        },
    )
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    network = []
    monkeypatch.setattr("ocq.model_analyzer.fetch_hf_model", lambda *a: network.append(a) or {})

    def rows(hardware, *more) -> dict[str, str]:
        monkeypatch.setattr(cli, "detect_hardware", lambda _bin: hardware)
        args = ["models", "--dir", str(folders), *more]
        result = CliRunner().invoke(cli.app, args, env={"COLUMNS": "250"})
        assert result.exit_code == 0, result.output
        return {
            line.split("│")[1].strip(): line for line in result.output.splitlines() if "│" in line
        }

    on_nvidia = rows(nvidia_pc(RTX_3090))
    tiny = on_nvidia["models/tiny-Q8_0.gguf"]
    assert "Perfect" in tiny and "gpu" in tiny
    assert "│ ?" in on_nvidia["damaged"]  # listed with a question mark, the others still rated
    assert "models/broken-Q4_K_M.gguf" not in on_nvidia  # cut off: a partial download
    assert "header unreadable" in str(rows(nvidia_pc(RTX_3090), "--all"))
    assert "needs a Mac" in on_nvidia["mlx-community/Qwen2.5-0.5B-Instruct-4bit"]
    assert "Perfect" in on_nvidia[QWEN]
    assert not network  # every fit comes from the files on disk, not the Hub

    mac = nvidia_pc(RTX_3090)
    from dataclasses import replace

    mac = replace(mac, backend="Metal", unified_memory=True)
    assert "Perfect" in rows(mac)["mlx-community/Qwen2.5-0.5B-Instruct-4bit"]

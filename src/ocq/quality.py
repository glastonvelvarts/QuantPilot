"""Measured quality: perplexity on WikiText-2, the text llama.cpp's own numbers are measured on.

Perplexity is how surprised the model is by real text it has not been trained to repeat: lower
is better. llama.cpp's method, which both runtimes here follow: the text is cut into chunks of
512 tokens, and each chunk's second half is scored, with its first half as the context.

    fixed  the first 40 chunks (20,480 tokens): a few minutes for a 7B model, enough to rank the
           formats of one model, which all read exactly the same tokens
    full   every chunk of the test text (about 290,000 tokens): comparable to published numbers

Every result is saved in quality.json, next to the speed history.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ocq.types import LLAMA_CPP, TRANSFORMERS, HardwareProfile, OcqError

DATASETS = Path.home() / ".cache" / "ocq" / "datasets"
STORE = Path.home() / ".cache" / "ocq" / "quality.json"
# llama.cpp's scripts/get-wikitext-2.sh downloads this copy.
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
DATASET = "WikiText-2 test"
CHUNK = 512


@dataclass(frozen=True)
class Slice:
    name: str
    chunks: int | None  # None: all of them


FIXED = Slice("fixed", 40)
FULL = Slice("full", None)


@dataclass(frozen=True)
class Perplexity:
    value: float
    error: float  # one standard error, as llama.cpp gives it
    chunks: int
    runtime: str
    build: str = ""  # llama.cpp's build; transformers' version for the transformers runtime

    @property
    def tokens_scored(self) -> int:
        return self.chunks * (CHUNK - 1 - CHUNK // 2)


def wikitext() -> Path:
    """The test text, downloaded once (4.7 MB) and checked against llama.cpp's copy."""
    text = DATASETS / "wikitext-2-raw" / "wiki.test.raw"
    if text.is_file():
        return text
    from ocq.quantizers.llama_cpp import download

    archive = download(WIKITEXT_URL, DATASETS / "wikitext-2-raw-v1.zip")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != WIKITEXT_SHA256:
        archive.unlink(missing_ok=True)
        raise OcqError(
            "The WikiText-2 download is not the file llama.cpp measures with (its checksum "
            "differs). Try again."
        )
    with zipfile.ZipFile(archive) as bundle:
        bundle.extract("wikitext-2-raw/wiki.test.raw", DATASETS)
    archive.unlink()
    return text


FINAL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
OVER = re.compile(r"perplexity over (\d+) chunks")


def llama_perplexity(
    tool: Path,
    model: Path,
    text: Path,
    piece: Slice,
    *,
    place: list[str],
    kv_cache: str,
    env: dict[str, str],
    log: Path,
    build: str = "",
) -> Perplexity:
    """llama-perplexity over the slice, the model placed as for its speed (bench.placement)."""
    from ocq.quantizers.gguf import run_tool  # gguf imports bench, which imports this

    cache_flags = [] if kv_cache == "f16" else ["-ctk", kv_cache, "-ctv", kv_cache]
    out, err = run_tool(
        [
            str(tool),
            "-m",
            str(model),
            "-f",
            str(text),
            "-c",
            str(CHUNK),
            *(["--chunks", str(piece.chunks)] if piece.chunks else []),
            *place,
            *cache_flags,
        ],
        log=log,
        env=env,
    )
    return read_llama_output(out + "\n" + err, build)


def read_llama_output(output: str, build: str = "") -> Perplexity:
    final, over = FINAL.search(output), OVER.search(output)
    if not final or not over:
        raise OcqError("llama-perplexity gave no final estimate.")
    return Perplexity(
        float(final.group(1)), float(final.group(2)), int(over.group(1)), LLAMA_CPP, build
    )


def score_chunks(tokens: list[int], piece: Slice, bos: int | None, chunk_logprobs) -> Perplexity:
    """llama.cpp's perplexity, for any runtime: chunks of CHUNK tokens (the first one replaced
    by the BOS token for models that start with one), each scored on its second half.
    chunk_logprobs(chunk) gives, for every position, the log-probability of the next token."""
    available = len(tokens) // CHUNK
    chunks = min(available, piece.chunks) if piece.chunks else available
    if chunks == 0:
        raise OcqError(f"The text has fewer than {CHUNK} tokens.")
    first = CHUNK // 2
    losses: list[float] = []
    for index in range(chunks):
        chunk = list(tokens[index * CHUNK : (index + 1) * CHUNK])
        if bos is not None:
            chunk[0] = bos
        logprobs = chunk_logprobs(chunk)
        losses += [-logprobs[j] for j in range(first, CHUNK - 1)]
    mean = sum(losses) / len(losses)
    spread = sum(loss * loss for loss in losses) / len(losses) - mean * mean
    error = math.sqrt(max(spread, 0.0) / (len(losses) - 1)) if len(losses) > 1 else 0.0
    value = math.exp(mean)
    return Perplexity(value, value * error, chunks, TRANSFORMERS)


def transformers_perplexity(model_dir: Path, text: Path, piece: Slice) -> Perplexity:
    """The same measurement in transformers: the original model in 16 bits, and the formats
    vLLM and transformers run (AWQ, GPTQ, NVFP4, bitsandbytes), on an NVIDIA GPU when there
    is one."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Its loading bars and the warning that the whole text is longer than the model reads at
    # once (it is cut into chunks): not for the user.
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cuda" if cuda else "cpu",
        dtype="auto" if cuda else torch.float32,
    )
    model.eval()
    tokens = tokenizer(text.read_text(encoding="utf-8"), add_special_tokens=False).input_ids
    starts_with_bos = tokenizer("a").input_ids[:1] == [tokenizer.bos_token_id]
    bos = tokenizer.bos_token_id if starts_with_bos else None
    if bos is not None:
        tokens = [bos, *tokens]  # llama.cpp tokenizes the whole text as one prompt

    def chunk_logprobs(chunk: list[int]) -> list[float]:
        ids = torch.tensor([chunk], device=model.device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        picked = logprobs.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
        return [*picked.tolist(), 0.0]

    result = score_chunks(tokens, piece, bos, chunk_logprobs)
    model = None  # frees the GPU memory before anything runs next
    if cuda:
        torch.cuda.empty_cache()
    return Perplexity(
        result.value, result.error, result.chunks, TRANSFORMERS, transformers.__version__
    )


def record(
    result: Perplexity,
    piece: Slice,
    *,
    model: str,
    label: str,
    kv_cache: str = "f16",
    file: str = "",
    hardware: HardwareProfile | None = None,
    device: str = "",
) -> None:
    """device: where it ran (gpu, cpu, ...): GPUs and CPUs round differently, so llama.cpp's
    perplexity of one file differs a little between them (17.27 on an M1's GPU, 17.24 on its
    CPU), and formats compare best measured on the same one."""
    from ocq.measurements import machine

    results = load()
    entry = {"model": model, "label": label, "runtime": result.runtime, "kv_cache": kv_cache}
    entry |= {"ppl": result.value, "error": result.error, "chunks": result.chunks}
    entry |= {"slice": piece.name, "dataset": DATASET, "chunk_tokens": CHUNK}
    entry |= {"runtime_version": result.build, "device": device}
    if file:
        entry |= {"file": file}
    if hardware is not None:
        entry |= {"machine": machine(hardware)}
    entry |= {"date": datetime.now().isoformat(timespec="seconds")}
    results.append(entry)
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(results, indent=2))


def load() -> list[dict]:
    """Every saved result; anything that is not one (a damaged file) is left out."""
    try:
        results = json.loads(STORE.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, dict) and isinstance(r.get("ppl"), (int, float))]


def results_of(model: str, label: str = "", slice_name: str = "") -> list[dict]:
    """Saved results of a model (a repo id, or local/<file>), in one format when label is
    given, on one slice when slice_name is given."""
    return [
        r
        for r in load()
        if model.lower() in (str(r.get("model", "")).lower(), f"local/{r.get('file', '')}".lower())
        and (not label or str(r.get("label", "")).upper() == label.upper())
        and (not slice_name or r.get("slice") == slice_name)
    ]

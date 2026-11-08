"""Tables of the ocq verify runs and llama-server start-ups in a results folder: what matched,
and the working memory llama.cpp reserved. Writes summary.md next to each and prints them.

uv run python scripts/summarize_verify.py results/<machine>
"""

import gzip
import re
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
COMPUTE = r"sched_reserve:\s+(\S+) compute buffer size =\s+([\d.]+)"
KV = r"llama_kv_cache:\s+\S+ KV buffer size =\s+([\d.]+)"
OUTPUT = r"output buffer size =\s+([\d.]+) MiB"


def real_load(log: str) -> str:
    """Only the real load counts, not llama.cpp's trial pass before it."""
    starts = [m.start() for m in re.finditer(r"load_tensors: loading model tensors", log)]
    return log[starts[-1] :] if starts else log


def in_ram(device: str) -> bool:
    return device.startswith("CPU") or device.endswith("_Host")


def total(log: str, pattern: str) -> float:
    return sum(float(x) for x in re.findall(pattern, log))


def compute(log: str) -> tuple[float, float]:
    """(GPU, RAM) MiB of llama.cpp's compute buffers."""
    buffers = re.findall(COMPUTE, log)
    gpu = sum(float(size) for device, size in buffers if not in_ram(device))
    ram = sum(float(size) for device, size in buffers if in_ram(device))
    return gpu, ram


def read(path: Path) -> str:
    """A log, or its gzipped copy (x.log.gz) as the results folders keep them."""
    packed = path.with_name(path.name + ".gz")
    if path.exists():
        data = path.read_bytes()
    elif packed.exists():
        data = gzip.decompress(packed.read_bytes())
    else:
        return ""
    return real_load(data.decode("utf-8", errors="replace"))


def write(folder: Path, rows: list[str]) -> None:
    summary = "\n".join(rows) + "\n"
    (folder / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


rows = [
    "| run | layers on GPU | checks | GPU compute MiB | RAM compute MiB | output MiB |",
    "|---|---|---|---|---|---|",
]
for text_file in sorted((OUT / "verify").glob("*.txt")):
    text = text_file.read_text(encoding="utf-8")
    log = read(text_file.with_suffix(".log"))
    layers = re.search(r"(\d+/\d+) layers on the GPU", text)
    differs = text.count("differs")
    checks = f"{text.count('│ ok')} ok" + (f", **{differs} differ**" if differs else "")
    error = re.search(r"Error: (.*)", text)
    if error:
        checks = f"error: {error.group(1)[-70:]}"
    gpu, ram = compute(log)
    rows.append(
        f"| {text_file.stem} | {layers.group(1) if layers else '-'} | {checks} "
        f"| {gpu:.2f} | {ram:.2f} | {total(log, OUTPUT):.2f} |"
    )
write(OUT / "verify", rows)

rows = [
    "| run | conversations | KV MiB | GPU compute MiB | RAM compute MiB | output MiB |",
    "|---|---|---|---|---|---|",
]
names = {p.name.removesuffix(".gz") for p in (OUT / "server").glob("*.log*")}
for log_file in sorted(OUT / "server" / name for name in names):
    log = read(log_file)
    seqs = re.search(r"n_seq_max\s+=\s+(\d+)", log)
    gpu, ram = compute(log)
    rows.append(
        f"| {log_file.stem} | {seqs.group(1) if seqs else '-'} | {total(log, KV):.2f} "
        f"| {gpu:.2f} | {ram:.2f} | {total(log, OUTPUT):.2f} |"
    )
write(OUT / "server", rows)

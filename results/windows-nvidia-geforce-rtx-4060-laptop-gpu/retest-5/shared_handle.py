"""verify's old way of reading a child's log (one TemporaryFile, seek + read while it writes),
against a separate read handle. Counts the child's numbered lines that survive.

python -I shared_handle.py   (writer.py next to it)
"""

import os
import subprocess
import sys
import tempfile
import time

LINES = 200000
WRITER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "writer.py")
CHILD = [sys.executable, "-I", WRITER]


def shared() -> bytes:
    with tempfile.TemporaryFile() as out:
        child = subprocess.Popen(CHILD, stdout=out, stderr=subprocess.STDOUT)
        seen = 0
        while child.poll() is None:
            out.seek(seen)
            seen += len(out.read())
            time.sleep(0.001)
        out.seek(0)
        return out.read()


def separate() -> bytes:
    with tempfile.NamedTemporaryFile(delete=False) as named:
        path = named.name
    with open(path, "wb") as out:
        child = subprocess.Popen(CHILD, stdout=out, stderr=subprocess.STDOUT)
        while child.poll() is None:
            with open(path, "rb") as reader:
                reader.read()
            time.sleep(0.001)
    with open(path, "rb") as reader:
        data = reader.read()
    os.unlink(path)
    return data


for name, run in (("shared handle (verify)", shared), ("separate read handle", separate)):
    lines = run().decode(errors="replace").splitlines()
    intact = len({line for line in lines if len(line) == 12 and line.startswith("line ")})
    print(f"{sys.platform} {name}: {intact} of {LINES} lines intact")

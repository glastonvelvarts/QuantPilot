"""One unbuffered write per line, as llama.cpp's logger does."""

import os

for i in range(200000):
    os.write(1, b"line %07d\n" % i)

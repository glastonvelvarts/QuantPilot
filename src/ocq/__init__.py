"""QuantPilot (the ocq package): which LLM, format and runtime to run on this PC, and making it."""

from importlib import metadata

# The name it is installed under (pip install quantpilot, uv tool install quantpilot); the
# Python package and the command keep the name ocq.
DISTRIBUTION = "quantpilot"

try:
    __version__ = metadata.version(DISTRIBUTION)
except metadata.PackageNotFoundError:  # run from the source without being installed
    __version__ = "0+unknown"

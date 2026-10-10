# Reproducible environment for mono-aeb: installs the package with the dev extra and runs the
# zero-data test suite. No video, no model weights and no network access are needed at test time.
#
#   docker build -t mono-aeb .
#   docker run --rm mono-aeb                                     # pytest (the default command)
#   docker run --rm mono-aeb python examples/quickstart.py --check   # zero-data demo
#
# The image is not published to a registry; it is built from this directory.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# The slim base has no OpenGL / glib runtime, which the opencv-python wheel needs at import time.
# Every core dependency ships a manylinux wheel, so no compiler or build toolchain is required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mono-aeb

# Dependency layer first, so editing a source file does not invalidate the install layer.
# README.md and LICENSE are copied because pyproject.toml references them.
COPY pyproject.toml README.md LICENSE ./
COPY aeb/ ./aeb/
RUN pip install -e ".[dev]"

COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/

# The zero-data suite: no dataset, no weights, no network.
CMD ["pytest", "-q", "--cov=aeb", "--cov-report=term-missing"]

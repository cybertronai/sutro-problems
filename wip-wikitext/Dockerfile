# Wikitext-bench base image — torch + WikiText-103 raw splits prebaked.
#
# Built and pushed to ghcr.io/ab-10/wikitext-bench so that submit.py can
# pull it via modal.Image.from_registry(...) and skip the slow
# pip_install(torch)+bake_wikitext layers on every cold start.
#
# Build:
#   docker build --platform=linux/amd64 -t ghcr.io/ab-10/wikitext-bench:latest -f Dockerfile .
# Push:
#   docker push ghcr.io/ab-10/wikitext-bench:latest
#
# --platform=linux/amd64 is mandatory: Modal's backend runs x86_64 Linux
# containers exclusively, so an image built on arm64 (e.g. Apple Silicon)
# without the flag will fail to start on Modal with an exec format error.
# The FROM line below also pins the platform, so a plain `docker build`
# from an amd64 host still produces an amd64 image.
#
# The image is intentionally not GPU-runtime-tagged; Modal layers its
# CUDA driver on top at run time and torch's bundled cu124 libs link
# against it.

FROM --platform=linux/amd64 python:3.11-slim

# Pin versions to what submit.py used to pip_install inline — keeps the
# image and the inline-build path producing identical environments.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 \
 && pip install --no-cache-dir \
        numpy==2.1.3 \
        nvidia-ml-py==12.560.30 \
        pyarrow==18.1.0

# Bake WikiText-103 raw splits into /data — same path bake_wikitext.py
# wrote to. Submissions read from /data via load_wikitext103.
COPY wikitext-103-raw-v1/wiki.train.raw /data/wiki.train.raw
COPY wikitext-103-raw-v1/wiki.valid.raw /data/wiki.valid.raw
COPY wikitext-103-raw-v1/wiki.test.raw  /data/wiki.test.raw

WORKDIR /workspace

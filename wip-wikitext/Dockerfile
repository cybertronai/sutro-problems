# Submitter harness template for the wikitext energy benchmark.
#
# Pinned base: PyTorch 2.5.1 + CUDA 12.4 + cudnn9 (runtime). This is the
# default Lambda On-Demand A100 80GB driver/CUDA combination. Submitters
# may swap in a different framework (JAX, raw CUDA, etc.) so long as
# the resulting container still runs on the same Lambda SKU and exposes
# the NVML energy counter.
#
# Build:
#   docker build -t wikitext-submission .
# Run (training + eval):
#   docker run --gpus all --rm \
#     -v /path/to/wikitext-103:/data:ro \
#     wikitext-submission

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# nvidia-ml-py exposes nvmlDeviceGetTotalEnergyConsumption used by EnergyMeter.
RUN pip install --no-cache-dir nvidia-ml-py==12.560.30

WORKDIR /workspace

# Copy the eval framework + baselines. A submitter replaces / extends
# these with their own training code, but keeps wikitext.py untouched
# (it defines the contract the runner trusts).
COPY wikitext.py baseline_ngram.py baseline_transformer.py run_eval.py ./

# Default entrypoint: the n-gram reference baseline. Submitters override
# CMD or ship a different run.sh.
CMD ["python3", "run_eval.py", "--data-dir", "/data", "--baseline", "ngram", "--n", "5"]

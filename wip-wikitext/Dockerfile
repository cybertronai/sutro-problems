# Submitter harness for the wikitext energy benchmark.
#
# Built and pushed by submit.py with the user's submission file copied
# in as /workspace/submission.py. Cloud-init on the Lambda VM pulls
# this image and runs entrypoint.sh, which writes /results/result.json
# (success) or /results/FAIL (any failure). The orchestrator on the
# developer's laptop SSHes in and blocks on those sentinels.
#
# Pinned base: PyTorch 2.5.1 + CUDA 12.4 + cudnn9. Matches the Lambda
# On-Demand A100 driver/CUDA combo.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# nvidia-ml-py exposes nvmlDeviceGetTotalEnergyConsumption used by
# EnergyMeter; datasets is used by fetch_data.py to materialise the
# WikiText-103 raw splits at first run.
RUN pip install --no-cache-dir nvidia-ml-py==12.560.30 datasets==3.2.0

WORKDIR /workspace

COPY wikitext.py baseline_ngram.py baseline_transformer.py \
     run_eval.py verify_nvml.py fetch_data.py task.py \
     entrypoint.sh submission.py ./

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

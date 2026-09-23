"""Open evaluation labels only after verifying every planned prediction."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SUITE = Path(os.environ.get("AMINIST21_ROOT", "/Users/yaroslavvb/git/aminist-21-validation"))
sys.path.insert(0, str(SUITE))
from aminist21.common import sha, array_sha, read, write, utc
from aminist21.suite import score


def main():
    study = ROOT/"results/all"
    execution = read(study/"execution.json")
    if execution["status"] != "complete":
        raise ValueError("Entire predeclared study must finish before scoring")
    transfer = study/"transfer"
    frozen = read(transfer/"prediction-manifest.json")
    if len(frozen["entries"]) != 55:
        raise ValueError("Expected five datasets and all eleven draws")
    metas = [study/"official/mnist.json"] + [transfer/row["metadata_path"] for row in frozen["entries"]]
    sources = read(study/"official/plan.json")["sources"]
    for name, expected in sources.items():
        if sha(study/"source"/name) != expected:
            raise ValueError("Frozen source snapshot changed")
    for path in metas:
        meta = read(path)
        if meta["provenance"]["sources"] != sources:
            raise ValueError("Models have mixed source versions")
        if meta["metadata"]["epochs_completed"] != 150 or meta["query_labels_supplied"] is not False:
            raise ValueError("Incomplete or invalid training protocol")
        if meta["metadata"]["train_count"] != (60000 if path.parent.name=="official" else 10000):
            raise ValueError("Incorrect training size")
        if sha(path.with_suffix(".npz")) != meta["prediction_file_sha256"]:
            raise ValueError("Prediction file changed")
        if sha(path.with_suffix(".pt")) != meta["checkpoint_sha256"]:
            raise ValueError("Checkpoint file changed")
        logit_file = path.with_name(path.stem+"-logits.npz")
        if sha(logit_file) != meta["logits_file_sha256"]:
            raise ValueError("Logits changed")
        with np.load(logit_file) as a, np.load(path.with_suffix(".npz")) as b:
            logits, predictions = a["logits"], b["predictions"]
        if logits.shape != (10000,10) or not np.isfinite(logits).all():
            raise ValueError("Malformed logits")
        if not np.array_equal(logits.argmax(1), predictions) or array_sha(predictions) != meta["predictions_sha256"]:
            raise ValueError("Predictions disagree with saved logits")
    write(study/"all-predictions-verified.json",{"verified_at_utc":utc(),"model_count":56,
          "query_labels_opened":False,"sources":sources,
          "official_predictions_sha256":sha(study/"official/mnist.npz"),
          "transfer_manifest_sha256":sha(transfer/"prediction-manifest.json")})
    # First access to the official 10,000 MNIST labels in this study.
    file = ROOT.parent/"official-mnist-sample-curve-20260922/raw/source/t10k-labels-idx1-ubyte.gz"
    compressed = file.read_bytes()
    if hashlib.md5(compressed).hexdigest() != "ec29112dd5afa0611ce80d1b7f02629c":
        raise ValueError("Official MNIST test-label checksum mismatch")
    labels = np.frombuffer(gzip.decompress(compressed),np.uint8,offset=8).astype(np.int64)
    with np.load(study/"official/mnist.npz") as archive:
        predictions = archive["predictions"]
    correct = int((predictions == labels).sum())
    result = {"scored_at_utc":utc(),"train_count":60000,"test_count":10000,
              "correct":correct,"errors":10000-correct,"accuracy_percent":correct/100,
              "error_percent":(10000-correct)/100,"seed":11,
              "test_labels_sha256":array_sha(labels),
              "confusion_matrix":np.bincount(labels*10+predictions,minlength=100).reshape(10,10).tolist(),
              "published_mean_accuracy_percent":99.431,
              "accuracy_gap_from_published_pp":correct/100-99.431,
              "interpretation":"One fresh PyTorch reproduction, not the paper's ten-run mean"}
    write(study/"official/scores.json", result)
    transfer_scores = score(transfer,SUITE/"data")
    print(json.dumps({"official":result,"transfer":transfer_scores["fifteen_numbers"]},indent=2))


if __name__ == "__main__":
    main()

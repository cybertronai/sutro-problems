# MNIST-medium ConvNet: ambiguities and problems

This is an accuracy feasibility study on canonical MNIST-medium. Cost translation and complete-task performance measurements are separate stages.

## Accuracy must be evaluated across 11 datasets

The current small/medium rule reports mean accuracy and sample standard deviation over **11 independently resampled train/test datasets** from the 60,000-example training pool. This study used only **one dataset split**. Its three final seeds measure training variability, and its ensemble is one learner on that same split. The ensemble's 5,894 / 6,000 result therefore does not establish the new aggregate requirement or its SD. The frozen learner also checks reference-dataset hashes; an 11-draw evaluation needs input validation against each draw's own manifest, with fresh learning and no cross-draw label or state reuse.

## Two per-dataset diagnostic thresholds

For this historical single-dataset check, **98%** means **5,880 / 6,000 correct**, and **98.14%** means **5,889 / 6,000 correct**. The current 11-draw aggregate instead requires 64,680 or 64,773 correct out of 66,000, respectively. Both checks must use exact counts: a rounded percentage can hide the difference. This study does not change the repository's threshold.

## The passing ensemble was not the validation winner

The predeclared ensemble scored **5,894 / 6,000**, meeting both per-dataset thresholds. The validation-selected primary single model scored **5,868 / 6,000**, missing 98% by 12 predictions. Individual seeds 102 and 103 scored **5,876** and **5,882**; only the latter clears 98%, by two predictions, and none of the individual models clears 98.14%. The ensemble used fixed membership and averaging before evaluation, but validation had selected a single model. Promoting the ensemble now would be a test-informed choice, not a validation-selected passing submission. No such change has been made to the frozen selection.

These results support feasibility on this fixed dataset, with limited margin and noticeable seed variation. The same public test set was used for the previous MLP attempt; this search isolated the test arrays from tuning, but the overall benchmark is not a blind evaluation service.

## Historical data and weights

The historical ConvNet evaluation used a different 10,000 / 10,000 dataset. Its training set includes **4,000 of the current 6,000 test examples**. Historical learned weights would therefore contaminate this evaluation. The new search borrows architecture ideas and trains every candidate from scratch using only the current permitted training data.

## Validation selection and seed variability

Trying several architectures, augmentation settings, checkpoints, and seeds makes the best validation result optimistic. Repeating promising configurations checks seed sensitivity but does not create an independent validation set. A 1,200-example validation set has only 24 errors at 98%, so a few examples can change the ranking.

Final configurations, stopping epochs, seeds, and any ensemble membership must be frozen before test evaluation. All predeclared final seeds should be reported; choosing a favorable seed after test evaluation would be test-informed selection. Further experiments selected using these test results would require that disclosure.

An ensemble's result must be distinguished from individual-model results. Its eventual cost must include training and evaluating every constituent model. Neither one successful seed nor an ensemble establishes that every training seed meets the accuracy target.

## Accuracy does not yet establish a scored submission

Native PyTorch accuracy is not yet a Dally-model implementation. Translation needs explicit arithmetic and the complete algorithm: initialization, augmentation, optimization, training, and prediction. Batch normalization, GELU, AdamW, and image interpolation introduce operations and numerical choices absent from the earlier MLP representation. Simplifying or approximating those operations can change predictions, so any translated implementation must be evaluated again.

A compact representation can avoid enumerating every instruction during scoring, but its totals still need to account for all expanded operations and memory accesses. This accuracy phase does not establish that a suitable translation exists or that it can be scored efficiently.

The A100 used for training is an experiment resource. Search/refit durations are operational records, not benchmark time or energy measurements. Unmeasured metrics must remain unmeasured. When that stage proceeds, execution time uses **ms**, energy **mJ**, occupied scratch area **mm²**, and time to score **s**, with two significant figures in human-readable tables.

- [Accuracy study](index.html)
- [Visible session export](session.html)

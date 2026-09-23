# Fixed-permutation MNIST literature audit

Research date: 2026-09-23. Scope: the original 60,000 training / 10,000 test examples, ten-way digit classification, one fixed pixel permutation shared by training and test, without supplying the original spatial arrangement to the learner.

## Finding

The highest **verified eligible published result found in this search** is the Ladder Network with an augmented MLP decoder, AMLP [2,2]: **99.431% mean accuracy**, equivalently **0.569% mean error, standard error 0.010 percentage points over ten runs**. This is the final ICML 2016 Table 2 result, not the slightly different preprint number. The earlier full Ladder result is effectively tied at **99.43%**, with **0.02 percentage-point standard deviation**. Their numerical difference is only one tenth of one test example in an average; it does not demonstrate a meaningful improvement across papers. I did not find a supported later result that clearly beats this under the specified restrictions. This is a source-verified literature finding, not proof of an exhaustive current leaderboard.

## What counts as pMNIST here

- **Single fixed permutation, full vector:** the usual permutation-invariant MNIST literature means the learning procedure has no knowledge of the original image geometry. It need not be a set-invariant model. An ordinary fully connected network qualifies.
- **Permuted sequential MNIST:** models consume a 784-position sequence, often causally. This adds a sequence-modeling constraint and has a separate literature.
- **Continual-learning Permuted MNIST:** the model learns several differently permuted tasks in succession, and the metric aggregates retained task performance. Those scores do not rank single-task MNIST classifiers.

Mathematically, for fixed permutation matrix P, a dense first layer can replace W by W P^-1, giving the same function on P x. With exchangeable initialization and matched coordinate permutations, learning is equivalent up to floating-point effects. Applying one permutation to every training/test/transfer example therefore tests the non-spatial algorithm without throwing away information. Providing the inverse permutation or performing spatial augmentation before permutation would restore spatial knowledge and would change the benchmark being compared.

## Source-verified candidates

| Method | Published accuracy | Protocol evidence and interpretation | Primary source |
|---|---:|---|---|
| Ladder AMLP [2,2], Pezeshki et al., ICML 2016 | **99.431%**, SE 0.010 pp | Table 2 explicitly labels full setting as 60,000 labels; ten seeds; test excluded from hyperparameter search. Non-spatial dense encoder and learned per-unit denoising decoder. | [Paper, §4.3 and Table 2](https://proceedings.mlr.press/v48/pezeshki16.pdf) |
| Full Ladder, Rasmus et al., NeurIPS 2015 | **99.43%**, SD 0.02 pp | Original 10,000 test held out. Hyperparameters selected with 50,000 train / 10,000 validation; final runs retrain on all 60,000. Ten runs. | [Extended paper, §4.1 and Table 1](https://arxiv.org/pdf/1507.02672) |
| Virtual adversarial training, Miyato et al., ICLR 2016 | **99.363%**, reported ±0.046 pp error | §3.2: 50,000 / 10,000 tuning split, then all training samples; ten initializations. Table 1 error 0.637. | [Paper, §3.2 and Table 1](https://arxiv.org/pdf/1507.00677) |
| Exemplar VAE augmentation, Norouzi et al., 2020 | **99.31%** | Authors explicitly identify permutation-invariant MNIST and report 0.69% error. Learned generative augmentation is different from spatial transformations. Not a contender for best result. | [Author paper abstract](https://arxiv.org/abs/2004.04795) |
| Adversarial maxout, Goodfellow et al., ICLR 2015 | **99.218%** | Reported mean error 0.782%; historical non-spatial benchmark reference. | [Paper, §6](https://arxiv.org/pdf/1412.6572) |
| Maxout, Goodfellow et al., ICML 2013 | **99.06%** | Explicitly unaware of 2D structure; 50k/10k tuning followed by 60k training. Their separate 99.55% result is convolutional and ineligible here. | [Paper, §5.1](https://proceedings.mlr.press/v28/goodfellow13.pdf) |

The frequently quoted DBM + dropout 99.21% and manifold tangent classifier 99.19% are historical comparison rows in the Ladder/VAT papers. Their original experimental pipelines were not fully re-audited here, because they cannot alter the leading choice.

## Reproduction specification for the selected family

**AMLP [2,2]** uses a per-unit decoder combinator with three scalar inputs (vertical reconstruction, corrupted lateral activation, and their product), two hidden layers of width two, and one scalar output. Its hidden activation is leaky ReLU with negative slope 0.1. The main paper trains with Adam at 0.002 for 100 epochs and linearly anneals to zero for another 50. The official supplementary tables specify noise standard deviation 0.3 at all seven levels, reconstruction weights `[2000, 0, 0, 0, 0, 0, 0]`, and combinator initialization standard deviation 0.025. [Main paper, §4.2–4.3](https://proceedings.mlr.press/v48/pezeshki16.pdf), [supplement, Tables 4–5](https://proceedings.mlr.press/v48/pezeshki16-supp.pdf).

**Original full Ladder:** dense widths `784-1000-500-250-250-250-10`, batch size 100, the same 150-epoch Adam schedule, and Gaussian corruption. No original spatial arrangement, geometric transformations, or external data is needed. Its author implementation supplies the exact all-label configuration: noise 0.3 and denoising weights `[1000, 1, 0.01, 0.01, 0.01, 0.01, 0.01]`. The encoder alone classifies at evaluation. [Author implementation](https://github.com/CuriousAI/ladder).

The original author repository is accessible, but requires legacy Theano / Blocks / Fuel. Examined revision: `5a8daa1760535ec4aa25c20c531e1cc31c76d911`. The old `arasmus/ladder` URL redirects to `CuriousAI/ladder`. No author-published AMLP-specific implementation or pretrained checkpoint was located in this search. Therefore an AMLP run in the current environment is a paper-based reimplementation, informed by the original Ladder implementation, and must be labeled as such until its MNIST result is reproduced.

## Claims excluded or kept separate

**Cireșan dense networks:** the 99.65% single-network result and 99.69% committee result use geometric/elastic image distortions; the committee also uses width normalization. Although the classifiers are MLPs, those transformations require the original spatial layout. They therefore belong to a different setting from unknown-layout fixed-permutation MNIST. [Single-network paper](https://arxiv.org/abs/1003.0358), [committee paper](https://arxiv.org/abs/1103.4487).

**Sequential result:** SMPConv, CVPR 2023, reports **99.10% on permuted sequential MNIST**, versus LSSL 98.76%, S4 98.70%, FlexTCN 98.63%, and HiPPO 98.30% in its Table 2. Its 99.75% result is *unpermuted* sequential MNIST. The original 60k refit details were not audited here, so this is a separate sequence-literature result rather than an eligible winner under the exact full-vector protocol. [Paper](https://arxiv.org/abs/2304.02330), [author code](https://github.com/sangnekim/SMPConv).

**Spatial kernel results:** Myrtle5 kernel 99.5% and CKN 99.6% use convolutional structure. The same comparison reports only 98.6–98.8% for the non-convolutional NTK, arccosine, and Gaussian kernels. [Primary technical report, Table 1](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2020/EECS-2020-93.pdf).

**Deep RBF 99.50% / ensemble 99.67%:** its image model uses spatially connected blocks, max pooling, and shifted/scaled/flipped/rotated augmentation. It is not an eligible non-spatial replacement despite having an RBF label. [Primary article, §3.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11120405/).

**Not-So-Random Features 99.65%:** this is a binary MNIST 1-versus-7 task, not ten-class full MNIST. [Primary paper, Table 1](https://openreview.net/pdf?id=Hk8XMWgRb).

**Continual-learning papers:** their task sequence, task identity assumptions, replay, and retained-accuracy aggregation are distinct experimental conditions. A high Split-MNIST accuracy or a particular task score must not be relabeled as full ten-way pMNIST accuracy.

## Additional challenger search

A bounded follow-up on 2026-09-23 used eight targeted searches beyond the phrase "permutation-invariant": fully connected MNIST at 99.5% and 99.6%; fully connected residual models near 0.4% error; recursive feature machines and MNIST (two query variants); self-normalizing networks and test/classification error (two query variants); and kernels near 0.5% MNIST error. These searches produced no additional primary-source result exceeding 99.431% while establishing the required original ten-class 60,000/10,000 protocol without spatial priors. This is a search-scope statement, not a claim that every later paper or unpublished result has been ruled out.

- **Self-normalizing networks (2017):** the paper's dense-MNIST illustration concerns training convergence. Its explicit 99.2% ±0.1 MNIST accuracy is for a CNN with convolution and max pooling, so it neither exceeds the selected result nor supplies a non-spatial challenger. [Primary paper, Figure 1 and §3](https://proceedings.neurips.cc/paper/6698-self-normalizing-neural-networks.pdf).
- **SERLU (2018):** §4 separates a four-hidden-layer, 200-neuron fully connected convergence experiment from the later CNN comparisons. The CNN experiments use a LeNet variant and spatial data augmentation. The inspected source does not establish an eligible test accuracy above the selected result. [Primary paper, §4](https://arxiv.org/html/1807.10117).
- **Recursive feature machines:** the primary paper studies MNIST concatenated with CIFAR-10 to expose simplicity bias; that is a modified classification task. Its original-MNIST references do not supply a competing full ten-way 60,000/10,000 result. The accompanying repository supplies runnable RFM code, but an available method is not evidence for a higher published MNIST score. [Primary paper, Figure 3 and Figure 15](https://arxiv.org/pdf/2212.13881), [author repository](https://github.com/aradha/recursive_feature_machines).

Search hits for higher fully connected MNIST accuracies also require checking augmentation carefully: the absence of convolution alone does not demonstrate independence from original image geometry. Only primary-source claims whose protocol can be verified should displace the current selection.

## Consequences for transfer experiments

Use the chosen method with the same fixed feature permutation throughout each dataset's training and evaluation. Keep the canonical MNIST train/test split intact and preserve the separate transfer suite's specified label spaces and draws. The recovered Aminist 21 suite deliberately repartitions curated pools, so its scores are not original-release test accuracies. If the transfer suite contains new image classes, compare the *learning recipe trained on each dataset*, unless the intended question explicitly concerns pretrained representation transfer. A digit classifier's ten output labels do not directly define accuracy on clothing, letters, or object categories.

For an exact reproduction, preserve 150 epochs and batch size 100, report the all-60k MNIST test once after the fixed recipe, and distinguish any reduced-budget pilot from a completed literature reproduction. The paper's quoted ± values describe across-run uncertainty; a single run has no such estimate. On 10,000 examples, one extra classification error changes accuracy by 0.01 percentage points.

# The 15 validation datasets

Aminist 21 validation measures **transfer of a training procedure**: retrain from fresh initialization on each task, then return its accuracy. It does not transfer an MNIST checkpoint. Each task has ten classes and float32, single-channel 9×9 images. The same 11 draws are used for every submitted procedure: 10,000 training examples and 10,000 evaluation examples per draw, with disjoint images within a draw. Different draws may reuse images.

The canonical processed pools are published as assets of this repository's data release. Their pinned checksums and the executable preparation code define the benchmark; the upstream links below document provenance and allow reconstruction. These are curated, resized adaptations, not the original releases. Code licensing does not replace the dataset terms below.

## Fixed tasks and source pools

Counts below are from the recovered source inventory and exact-image curation. “Decoded” counts valid images in the selected source pool; “clean” counts the canonical pool after duplicate/conflict removal. All fifteen pools support the full 10,000/10,000 protocol.

| Task ID | Source pool | Decoded | Clean | Original class IDs in model-label order |
|---|---|---:|---:|---|
| `kmnist` | [KMNIST](https://github.com/rois-codh/kmnist), official train | 60,000 | 60,000 | 0–9 |
| `emnist_letters_aj` | [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset), official train, A–J | 48,000 | 47,998 | 1–10 |
| `emnist_letters_kt` | EMNIST Letters, official train, K–T | 48,000 | 47,999 | 11–20 |
| `emnist_balanced_aj` | EMNIST Balanced, official train, A–J | 24,000 | 23,998 | 10–19 |
| `emnist_digits` | EMNIST Digits, official train | 240,000 | 240,000 | 0–9 |
| `emnist_mnist` | EMNIST MNIST, official train | 60,000 | 60,000 | 0–9 |
| `qmnist_recovered` | [QMNIST](https://github.com/facebookresearch/qmnist), test rows 10,000–59,999, repartitioned | 50,000 | 50,000 | 0–9 |
| `k49_10` | [Kuzushiji-49](https://github.com/rois-codh/kmnist), ten fixed official-train classes absent from KMNIST | 60,000 | 60,000 | 0, 1, 2, 5, 7, 9, 10, 11, 15, 18 |
| `kannada_digits` | [Kannada-MNIST](https://github.com/vinayprabhu/Kannada_MNIST), official train | 60,000 | 60,000 | 0–9 |
| `devanagari_digits` | [UCI Devanagari](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset), 17,000 train + 3,000 test, repartitioned | 20,000 | 20,000 | `digit_0`–`digit_9` |
| `madbase` | [MADBase](https://datacenter.aucegypt.edu/shazeem/), official train | 60,000 | 59,944 | 0–9 |
| `notmnist_large` | [notMNIST](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html), large archive only | 529,114 | 461,751 | A–J directories |
| `fashion_mnist` | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), official train | 60,000 | 60,000 | 0–9 |
| `svhn` | [SVHN](http://ufldl.stanford.edu/housenumbers/), cropped-digit train; extra excluded | 73,257 | 73,255 | 10, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| `cifar10` | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), official train | 50,000 | 50,000 | 0–9 |

The ten names in each row below correspond to model labels 0 through 9. These mappings were fixed before model results.

| Task | Class names |
|---|---|
| KMNIST | お, き, す, つ, な, は, ま, や, れ, を |
| K49 ten-class subset | あ, い, う, か, く, こ, さ, し, た, て |
| Letters A–J / Balanced A–J / notMNIST | A, B, C, D, E, F, G, H, I, J |
| Letters K–T | K, L, M, N, O, P, Q, R, S, T |
| EMNIST Digits / EMNIST MNIST / QMNIST / SVHN | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Kannada-MNIST | ೦, ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯ |
| Devanagari digits | ०, १, २, ३, ४, ५, ६, ७, ८, ९ |
| MADBase | ٠, ١, ٢, ٣, ٤, ٥, ٦, ٧, ٨, ٩ |
| Fashion-MNIST | T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| CIFAR-10 | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |

EMNIST Letters merges uppercase/lowercase forms; Balanced retains its release's own class-merging rules. notMNIST contains rendered fonts, not handwriting, and contains only A–J: there is no K–T continuation in that release. The suite excludes the original MNIST control and three earlier small-pool tasks (USPS, DiG-MNIST, and ten Omniglot classes) to keep all fifteen tasks at the same sample size. The original notMNIST creator describes the font-rendering process in the [announcement](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).

## Image preparation and leakage boundaries

The preparation code performs exact separable box-area averaging to float32 9×9 pixels in `[0,1]`. EMNIST swaps the native height and width axes. MADBase inverts its white background to bright ink. CIFAR-10 and SVHN first use `0.299 R + 0.587 G + 0.114 B`; other pools retain their native grayscale values and polarity. Native images are 28×28 except Devanagari, CIFAR-10, and SVHN (32×32). The learner controls any further input normalization.

Curation hashes native C-order pixel bytes before 9×9 resizing, after the documented orientation/polarity correction. RGB photograph hashes retain all original channels. Within each task, the first instance of a same-label image group survives; every instance of an image group with conflicting labels is removed. This prevents identical native images appearing on both sides of a draw. It does not remove near-duplicates, identify duplicate writers/fonts, or guarantee that two different originals cannot become identical after downsampling.

The recorded removals are two rows from Letters A–J, one from Letters K–T, two from Balanced A–J, and 56 from MADBase. SVHN loses both members of one conflicting-label image group. notMNIST has 529,119 PNG entries, five undecodable files, 60,043 redundant same-label rows, and 7,320 rows in 195 conflicting-label groups. These data checks occur independently of model accuracy.

These tasks are not fifteen independent populations. EMNIST variants and QMNIST share NIST ancestry; K49 and KMNIST share the Kuzushiji collection. The first 10,000 QMNIST test images reconstruct the standard MNIST test set and are specifically excluded here. The retained 50,000 images form a new train/evaluation pool. See the creators' [QMNIST reconstruction description](https://github.com/facebookresearch/qmnist) and [Kuzushiji description](https://github.com/rois-codh/kmnist).

Evaluation is random-image generalization inside the stated pools, not each source's official held-out-test benchmark. Devanagari combines the original splits; MADBase's original writer separation is not preserved by random draws within its training split. Related fonts may also appear on both sides of notMNIST draws. Use these fifteen numbers to compare procedures under this common protocol, not as official-test state-of-the-art claims. [MADBase source split design](https://datacenter.aucegypt.edu/shazeem/).

## Dataset terms and attribution

This audit records primary-source terms as checked on **2026-09-22**. The code's MIT license applies to the code only. Dataset adaptations retain their applicable original terms; an unspecified license is not a claim of public domain or unrestricted reuse. The release mirrors the curated research inputs with their source attribution and notices, without claiming a new license over the complete collection.

| Dataset family | Terms supported by the primary source | Required notice / qualification |
|---|---|---|
| KMNIST and K49 | [CC BY-SA 4.0](https://github.com/rois-codh/kmnist#license) | Credit CODH and the NIJL source. These selected/resized/deduplicated adaptations are distributed under CC BY-SA 4.0. |
| EMNIST, all five tasks | [University record: Other license; copyright Western Sydney University](https://research-data.westernsydney.edu.au/published/2df91130519411ecb15399911543e199/) | The record marks access open, but its standard license is unspecified. NIST hosting does not establish public-domain status for this derivative release. |
| QMNIST | [BSD 3-clause style license](https://github.com/facebookresearch/qmnist/blob/main/LICENSE), applied to QMNIST by its README | Preserve Facebook's copyright notice, conditions and disclaimer; no endorsement. |
| Kannada-MNIST | [CC BY 4.0, author-linked Zenodo release](https://zenodo.org/records/3359691) | Credit Vinay Uday Prabhu; identify our image adaptations. The Zenodo API records `cc-by-4.0`. |
| Devanagari | [CC BY 4.0, UCI record](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset) | Credit Shailesh Acharya and Prashnna Gyawali; identify our digit selection and image adaptations. |
| MADBase | [No standard license stated on the original page](https://datacenter.aucegypt.edu/shazeem/) | Preserve author attribution. No license from an unrelated mirror is substituted. |
| notMNIST | [No standard license stated on the original announcement](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) | Credit Yaroslav Bulatov. No blanket license for underlying fonts is asserted. |
| Fashion-MNIST | [MIT](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE) | Preserve Zalando SE's copyright and permission notice. |
| SVHN | [Non-commercial use only](http://ufldl.stanford.edu/housenumbers/) | The creator page attaches this restriction to cropped digits as well as full numbers. It is not an MIT/CC license; commercial use is outside these stated terms. |
| CIFAR-10 | [No standard dataset license stated on the original page](https://www.cs.toronto.edu/~kriz/cifar.html) | Credit Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. A third-party loader's software license is not a license for these images. |

KMNIST attribution: “KMNIST Dataset” (created by CODH), adapted from “Kuzushiji Dataset” (created by NIJL and others), [doi:10.20676/00000341](https://doi.org/10.20676/00000341). Our modifications are the stated class selection, image curation and resizing. The adaptations retain [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The machine-readable record includes exact mappings, primary-source links, and the complete QMNIST and Fashion-MNIST license notices. Preserve it alongside redistributed data: [dataset-licenses.json](dataset-licenses.json).

## Citations

Cite the sources relevant to the tasks you report, together with this benchmark version and protocol.

- Clanuwat, T., Bober-Irizar, M., Kitamoto, A., Lamb, A., Yamamoto, K., and Ha, D. (2018). [Deep Learning for Classical Japanese Literature](https://arxiv.org/abs/1812.01718).
- Cohen, G., Afshar, S., Tapson, J., and van Schaik, A. (2017). [EMNIST: an extension of MNIST to handwritten letters](https://doi.org/10.26183/m9k1-zr06); [paper](https://arxiv.org/abs/1702.05373).
- Yadav, C., and Bottou, L. (2019). [Cold Case: The Lost MNIST Digits](https://arxiv.org/abs/1905.10498). NeurIPS 32.
- Prabhu, V. U. (2019). [Kannada-MNIST: A new handwritten digits dataset for the Kannada language](https://arxiv.org/abs/1908.01242); [data release](https://doi.org/10.5281/zenodo.3359691).
- Acharya, S., and Gyawali, P. (2015). [Devanagari Handwritten Character Dataset](https://doi.org/10.24432/C5XS53). UCI Machine Learning Repository.
- Abdelazeem, S., and El-Sherif, E. [The Arabic Handwritten Digits Databases: ADBase & MADBase](https://datacenter.aucegypt.edu/shazeem/). The American University in Cairo.
- Bulatov, Y. (2011). [notMNIST dataset](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).
- Xiao, H., Rasul, K., and Vollgraf, R. (2017). [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).
- Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. (2011). [Reading Digits in Natural Images with Unsupervised Feature Learning](https://ai.stanford.edu/~twangcat/papers/nips2011_housenumbers.pdf). NIPS Workshop on Deep Learning and Unsupervised Feature Learning.
- Krizhevsky, A. (2009). [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf). University of Toronto technical report.

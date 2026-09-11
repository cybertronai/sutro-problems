# Training-only kernel screening

The frozen 32-choice grid screened raw pixels and moment-based deskewing,
four RBF bandwidths, and four ridge penalties on the same 4,800/1,200 split
used by the ConvNet search. Only the canonical training images and labels
were supplied. No medium test arrays or new eleven-draw test labels were used.

The best result was **1,162/1,200 (96.8%)**, using deskewing, gamma 0.5,
and ridge 0.1. This family was not advanced to formal translation or scoring.
The screening used native FP64 PyTorch exponentials and a dense linear solve;
it is not claimed to be a v4 implementation.

The first run computed all results but could not deserialize a PyTorch version
object in the local environment. The source was changed only to serialize
that version as a plain string. `kernel_protocol_transport_failure.json`
retains the original freeze; `kernel_protocol.json` and `kernel_results.json`
record the successful rerun of the unchanged grid. Neither run accessed test
labels. Runtime here is an exploratory diagnostic, not a benchmark score.

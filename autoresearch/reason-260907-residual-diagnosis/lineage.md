# Bounded reasoning record

Initial candidate: response dominance, weak context mapping, normalization drift,
and selection mismatch jointly explain low residual performance.

Independent critic rejected treating loss magnitudes as gradient evidence, noted
the batch transition, and established that block ablations are not fully wired.
It recommended gene-specific direct context modeling and out-of-fold baseline
errors for any additive response correction.

Live artifact analysis established small-amplitude prediction variation, broad
per-gene ordering deficits and concentrated prediction spectrum. A second critique
rejected treating shrinkage as the cause of low Pearson, matching noisy target rank,
or estimating calibration from observed test ratios.

Revised candidate: measured failure is strongly shrunk, weak contextual ordering;
the leading structural explanation is indirect gene-conditioned parameter sharing.
Response conflict and feature drift are testable hypotheses. Prioritize a stable
direct per-gene contextual map, honest calibration, then response incremental value.

Workflow was bounded to concrete evidence gathering and an independent adversarial
review; no randomized judge panel or formal convergence result was run or claimed.

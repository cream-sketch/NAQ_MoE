# NAQ-MoE Method Overview

NAQ-MoE compresses routed MoE experts at neuron granularity. The key idea is
that expert-local neuron indices are not semantically aligned, so a compression
policy should first recover functionally corresponding neurons across experts.

## Offline Reconstruction

1. **Calibration:** collect pre-MoE hidden states on real text.
2. **Functional signatures:** evaluate every routed expert on the same hidden
   states and record each SwiGLU neuron's activation vector.
3. **Neuron alignment:** match neurons across experts in signature space.
4. **Tier allocation:** compute per-slot functional dispersion and assign each
   aligned slot to Universal, Group, or Specialist tiers under a storage budget.
5. **Tiered reconstruction:** store Universal slots as global BF16 centroids,
   Group slots as FP8 group centroids, and Specialist slots as INT4
   expert-specific triplets.

## Spectrum-Stationary Dataflow

The same tier map drives inference scheduling:

- Universal: high reuse, staged near compute, evaluated once per token.
- Group: medium reuse, served from cache/L2, evaluated once per active group.
- Specialist: low reuse, streamed from HBM, evaluated per selected expert.

This connects the algorithmic reuse score to precision, memory placement, and
estimated per-token traffic.


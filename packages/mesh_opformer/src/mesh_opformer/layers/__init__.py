"""
NOTE
Questions to be answered?

1. Where and what kind of normalizations are required?

2. Need to test the L0 @ v_V under what conditions is valid.
What are the theoretical guarantess for this to work, in case
no spectral decomposition takes place?

3. `d_0` and `d_1` are sparse incident non-learnable matrices.
Need to incorporate sparse - dense matrix computations. Cannot
use `einsum` as it does not support sparse - dense computations.
"""
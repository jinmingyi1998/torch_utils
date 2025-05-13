Utilities related to PyTorch. The code is really messy and originally written for my personal usage, but open-source here since someone wants to use it.

Currently contains:

1. Merge multiple Torch Profiler traces from multiple ranks into one big trace (useful when checking cooperation between ranks).
2. Extract kernel time breakdown statistics from profiles.

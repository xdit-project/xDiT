"""Sparsity strategies: pre/post transform pairs around a kernel.

Shared by several kernel families -- sparge is used by MHA v4, both Sage
versions and Flex Block -- so they live here rather than in any one backend
module. Not a framework concept: the framework only knows whether a spec is
sparse, never which strategy it uses.
"""

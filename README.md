# PyCFTBoot

PyCFTBoot is an interface for the conformal bootstrap as discussed in [its 2008 revival](http://arxiv.org/abs/0807.0004). Starting from the analytic structure of conformal blocks, the code formulates semidefinite programs without any proprietary software. The code does NOT perform the actual optimization. It assumes you already have a program for that, namely [SDPB](https://github.com/davidsd/sdpb) by David Simmons-Duffin.

PyCFTBoot is still in the early stages but it supports bounds on single correlator scaling dimensions and OPE coefficients in an arbitrary number of spacetime dimensions.

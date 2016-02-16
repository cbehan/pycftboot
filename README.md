# PyCFTBoot

PyCFTBoot is an interface for the conformal bootstrap as discussed in [its 2008 revival](http://arxiv.org/abs/0807.0004). Starting from the analytic structure of conformal blocks, the code formulates semidefinite programs without any proprietary software. The code does NOT perform the actual optimization. It assumes you already have a program for that, namely [SDPB](https://github.com/davidsd/sdpb) by David Simmons-Duffin.

PyCFTBoot supports bounds on scaling dimensions and OPE coefficients in an arbitrary number of spacetime dimensions. The four-point functions used for the bounds must contain only scalars but they may have any combination of scaling dimensions and transform in arbitrary representations of a global symmetry.

## Installation
If you use one of the mainstream Linux distributions, the following instructions should help you install PyCFTBoot and everything it depends on.

1. Follow [the instructions](https://github.com/davidsd/sdpb/blob/master/Install.md) for installing SDPB. When this is done, you will have [Boost](http://www.boost.org) and [GMP](https://gmplib.org) as well, so we will not need to discuss those further.

2. Additional run-time dependencies are: [Sympy](http://www.sympy.org), [Numpy](http://www.numpy.org/) and [MPFR >= 3.1](http://www.mpfr.org/). The build-time dependencies are: [Cython](http://cython.org/) and [CMake >= 2.8](https://cmake.org/). You should install all of these. You will probably not need to compile them because most distros have these packages in their repositories.

3. There are two dependencies left. One is [Symengine](https://github.com/symengine/symengine) which probably needs to be compiled. One commit that has been tested is 5427bbe. An even better idea is to use the latest commit that has been [marked stable](https://github.com/symengine/symengine.py/blob/master/symengine_version.txt) for language bindings. To compile it with the recommended settings, run:

        mkdir -p build && cd build
        cmake .. -DWITH_TCMALLOC:BOOL=ON -DWITH_PTHREAD:BOOL=ON -DWITH_SYMENGINE_THREAD_SAFE:BOOL=ON -DWITH_MPFR:BOOL=ON
        make
        
The last one is especially important.

4. Lastly, compile and install [Symengine.py](https://github.com/symengine/symengine.py).

5. It is possible that PyCFTBoot works now. To be sure, there are potentially three useful changes to the code in `bootstrap.py`. If the Python binary you use is called something else, change `python2` in the first line to `python` or whatever it is. If the sympy package you installed does not have mpmath as a separate module, change `import mpmath` to `import sympy.mpmath as mpmath`. Finally, change `/usr/bin/sdpb` to a different path if your SDPB is not in the system-wide directory.

6. Have fun constraining CFTs and convincing cluster maintainers to install fairly new software.

## Attribution
If PyCFTBoot is helpful in one of your publications, please cite:

- C. Behan, "PyCFTBoot: A flexible interface for the conformal bootstrap", [arXiv:1602.02810 \[hep-th\]](http://arxiv.org/abs/1602.02810).

# PyCFTBoot

PyCFTBoot is an interface for the conformal bootstrap as discussed in [its 2008 revival](http://arxiv.org/abs/0807.0004). Starting from the analytic structure of conformal blocks, the code formulates semidefinite programs without any proprietary software. The code does NOT perform the actual optimization. It assumes you already have a program for that, namely [SDPB](https://github.com/davidsd/sdpb) by David Simmons-Duffin.

PyCFTBoot supports bounds on scaling dimensions and OPE coefficients in an arbitrary number of spacetime dimensions. The four-point functions used for the bounds must contain only scalars but they may have any combination of scaling dimensions and transform in arbitrary representations of a global symmetry.

## Installation on Linux
If you use one of the mainstream Linux distributions, the following instructions should help you install PyCFTBoot and everything it depends on.

1. Follow [the instructions](https://github.com/davidsd/sdpb/blob/master/Install.md#linux) for installing SDPB. When this is done, you will have [Boost](http://www.boost.org) and [GMP](https://gmplib.org) as well, so we will not need to discuss those further.

2. Additional run-time dependencies are: [Sympy](http://www.sympy.org) and [MPFR >= 4.0](http://www.mpfr.org/). The build-time dependencies are: [Cython](http://cython.org/) and [CMake >= 2.8](https://cmake.org/). You should install all of these. You will probably not need to compile them because most distros have these packages in their repositories.

3. There are two library dependencies left. One is [Symengine](https://github.com/symengine/symengine) which probably needs to be compiled. One commit that has been tested is ec460e7. An even better idea is to use the latest commit that has been [marked stable](https://github.com/symengine/symengine.py/blob/master/symengine_version.txt) for language bindings. To compile it with the recommended settings, run:

        mkdir build && cd build
        # WITH_PTHREAD and WITH_SYMENGINE_THREAD_SAFE might be helpful as well
        cmake .. -DWITH_MPFR:BOOL=ON
        make
        
4. Lastly, compile and install [Symengine.py](https://github.com/symengine/symengine.py).

5. Additionally, extracting the spectrum with PyCFTBoot will require the binary [unisolve](https://numpi.dm.unipi.it/mpsolve-2.2/).

## Installation on Mac
Thanks to Jaehoon Lee for writing these instructions and testing them on OS X 10.11 (El Capitan).

1. Follow the instructions for [installing SDPB](https://github.com/davidsd/sdpb/blob/master/Install.md#mac-os-x) on Mac OS X. Installing gcc takes a long time, so be patient. Also, you don't need `sudo` for installing boost due to recent changes. After that, you will have homebrew, gcc, gmp, mpfr and boost installed. The default compilers should be renamed as `gcc` and `g++` following the instructions.

2. Build all the required packages (Cython, Numpy, Sympy and Mpmath). One might alreday have these packages installed. The following assumes that no package other than the system's Python is installed.

        # Install homebrew's python which comes with pip
        brew install python
        brew linkapps python
        pip install --upgrade pip setuptools

        # Numpy
        brew install homebrew/python/numpy

        # Cython
        pip install cython

        # Sympy 
        pip install sympy

        # Mpmath - technically not required as it is included in sympy
        pip install mpmath

3. Install `cmake` using homebrew:

        brew install cmake

4. Download [Symengine](https://github.com/symengine/symengine) and compile it. If you fail to install and need to rebuild, remove the build folder and start remaking it. Unpack the source file within the directory and run:

        mkdir build && cd build
        # Turning on the MPFR option is critical for using PyCFTBoot 
        CC=gcc CXX=g++ cmake .. -DWITH_MPFR:BOOL=ON
        make
        # Test everything is built correctly
        ctest
        # Install files to default directories
        make install

5. Install the Python bindings [Symengine.py](https://github.com/symengine/symengine.py). Download the source and within the directory run:

        CC=gcc CXX=g++ python setup.py install

## Usage
To test that PyCFTBoot is working, try to run:

        python
        import bootstrap

If that doesn't work, you should check if the dependencies import correctly.

        python
        import symengine

Assuming that all of this works, `python tutorial.py` will enter a tutorial with four examples. There are two changes that you might want to make to `bootstrap.py`. One is changing `python2` to `python` in the first line, for systems that don't append a specific number. The other is setting the path of SDPB and related executables by searching for `/usr/bin/sdpb` and updating this. Have fun constraining CFTs and convincing cluster maintainers to install fairly new software!

## Attribution
If PyCFTBoot is helpful in one of your publications, please cite:

- C. Behan, "PyCFTBoot: A flexible interface for the conformal bootstrap", [arXiv:1602.02810 \[hep-th\]](http://arxiv.org/abs/1602.02810).

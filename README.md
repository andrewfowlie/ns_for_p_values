[![Python application](https://github.com/andrewfowlie/ns_for_p_values/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/andrewfowlie/ns_for_p_values/actions/workflows/python-app.yml)

# p_value module

Contains methods for 

- p-value from brute force
- p-value from MultiNest nested sampling
- p-value from dynesty nested sampling
- p-value from PolyChord nested sampling

PolyChord requires the latest PolyChord version with the logLstopping condition.

# Unit tests

    cd test
    export PYTHONPATH=../:../examples
    python3 test.py

# examples

Contains analytic chi-squared examples and resonance search example.

# plots

Contains code to reproduce figures from the paper.

# Installation

Tested in Python 3. See requirements.txt for Python modules available from PyPI and .github/workflows/python-app.yml for example of a build on Ubuntu.

    # Install PC dependencies
    sudo apt-get install gfortran libopenmpi-dev
    pip install mpi4py

    # Install PC
    pip install git+https://github.com/andrewfowlie/PolyChordLite@hack_stopping # change if merged

    # Install MN dependencies
    sudo apt-get install libblas{3,-dev} liblapack{3,-dev} libatlas{3-base,-base-dev} cmake build-essential git gfortran

    # Install MN
    git clone https://github.com/farhanferoz/MultiNest
    cd MultiNest/MultiNest_v3.12_CMake/multinest
    mkdir build
    cd build
    cmake ..
    make
    sudo make install

    # Install Python dependencies
    python -m pip install --upgrade pip
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    # Test with unittest
    cd test
    export PYTHONPATH=../:../examples:${PYTHONPATH}
    export LD_LIBRARY_PATH=/usr/local/lib/:${LD_LIBRARY_PATH}
    python test.py

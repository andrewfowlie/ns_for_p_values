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

# Dependencies

Tested in Python 3. Requires at least:

- dynesty
- pypolychord
- pymultinest
- scipy
- numpy
- matplotlib

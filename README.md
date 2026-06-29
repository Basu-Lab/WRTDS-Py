# WRTDS-Py

WRTDS-Py is a Python implementation of the Weighted Regressions on Time, Discharge, and Season (WRTDS) method for analyzing water quality data. The code is adapted from the original R implementation in the [`EGRET` package](https://doi-usgs.github.io/EGRET/) by the US Geological Survey. From `EGRET`, only the core WRTDS and WRTDS-K functions are implemented in Python.

## Status

The code is in the early stages of development. Currently, the reference example of the `Choptank River` is implemented and tested. The Python code will be tested against this reference example to ensure that it produces the same results as the R implementation.

## Usage

### Run test example

To run the test example, you can use the following code:

```bash
uv run pytest
```

Running `R 4.5.3` and `EGRET 3.0.11`.
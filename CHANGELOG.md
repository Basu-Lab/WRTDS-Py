# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - YYYY-MM-DD

First stable release, published on PyPI.

### Added

- `WRTDS`: classic WRTDS with a pre-computed prediction surface, fitted by weighted
  left-censored (Tobit) maximum likelihood with an analytic gradient.
- `WRTDSK`: WRTDS-Kalman (Zhang & Hirsch, 2019), following `EGRET`'s `WRTDSKalman`.
  Residuals are standardized by the WRTDS standard error, each gap between samples is filled
  with an AR(1) path conditioned on the samples at both ends, and the result is averaged over
  `niter` Monte Carlo iterations (default 200), which also simulate censored residuals from a
  truncated normal. `seed` accepts an int or a `numpy.random.Generator` and does not change the
  global NumPy random state.
- Zero-flow days: if any day has `Q = 0`, `LogQ` is computed with 0.1% of the mean discharge
  added to every day, following `EGRET`; `Q` itself is unchanged.
- Validation against USGS `EGRET` v3.0.11 on the Choptank River reference dataset.
- Package metadata for PyPI (license, classifiers, project URLs), `CITATION.cff`, and a
  GitHub Actions workflow that publishes tagged releases to PyPI.

[1.0.0]: https://github.com/Basu-Lab/WRTDS-Py/releases/tag/v1.0.0

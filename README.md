# itk_cucim

[![Build, test, package](https://github.com/InsightSoftwareConsortium/itk_cucim/actions/workflows/build-test-package-python.yml/badge.svg)](https://github.com/InsightSoftwareConsortium/itk_cucim/actions/workflows/build-test-package-python.yml)

## Development

```
pip install -e ".[test]"

# Install dependencies for `cucim.skimage` (assuming that CUDA 11.0 is used for CuPy)
pip install cupy-cuda110

pytest
```

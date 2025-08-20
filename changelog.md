# Changelog

## Version 0.3.3

Fix issue with implementation of `regression_idw_interpolation` which caused `IndexErrors` internally when called.

Update documentation to show usage of new sklearn library since the changes in sklearn==1.4. See: https://github.com/shuttle1987/pyidw/issues/6

## Version 0.3.2

Fixed `FutureError` issue in implementation of `idw_interpolation` that is likely to cause breakages in the future. See https://github.com/shuttle1987/pyidw/issues/3

Ran all code through ruff formatting library.

## Version 0.3.1

Fix issues in documentation

## Version 0.3.0

New fork of library created so that this library could be maintained. See: https://github.com/shuttle1987/pyidw

Use uv package manager

Update project from setup.py to more modern pyproject.toml/uv.lock combination

Pin rasterio version to 1.4.0 to deal with this issue: https://github.com/rasterio/rasterio/issues/3382

## Version 0.2.21

Last version released by https://github.com/yahyatamim/pyidw

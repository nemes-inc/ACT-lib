"""rbf namespace package anchor for the mpem extension.

This file ensures that `rbf` is a namespace package so modules in
`python/rbf` (e.g., rbf.rbf_train) are discoverable alongside the
binary extension `rbf.mpem` built here.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

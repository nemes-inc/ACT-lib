"""pyact namespace package anchor for the mpbfgs extension.

This ensures `pyact` is discoverable and can host the compiled extension
module `pyact.mpbfgs` built by scikit-build.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

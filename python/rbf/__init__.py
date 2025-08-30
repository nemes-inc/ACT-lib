# Namespace-friendly rbf package so that installed rbf.mpem (extension) is discoverable
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

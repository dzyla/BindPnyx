"""pxdbench tool backends.

ProtenixFilter pulls in torch + protenix, so it is imported lazily: importing
`pxdbench.tools` (or anything under it, e.g. `pxdbench.tools.boltz`) no longer
needs a GPU stack. `register("public", ...)` stores a factory that imports the
real class on first call; `from pxdbench.tools import ProtenixFilter` still
works through the module-level __getattr__ below.
"""
from .registry import register


def _protenix_filter_factory(*args, **kwargs):
    from .ptx.ptx import ProtenixFilter
    return ProtenixFilter(*args, **kwargs)


def __getattr__(name):                                  # PEP 562: back-compat for `from pxdbench.tools import ProtenixFilter`
    if name == "ProtenixFilter":
        from .ptx.ptx import ProtenixFilter
        return ProtenixFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


register("public", _protenix_filter_factory)

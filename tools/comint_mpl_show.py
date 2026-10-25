# pyright: reportMissingImports=false
# Docker comint display shim: Matplotlib -> PNG, Plotly -> interactive HTML.
import glob
import io
import os
import re
import readline
import rlcompleter
import time
from contextlib import redirect_stderr, redirect_stdout

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as _plt

_PLOT_DIR = "output/.comint-plots"
_plot_count = 0

# Prune old display files.
_cutoff = time.time() - 3 * 86400
for _pattern in ("fig-*.png", "plotly-*.png", "plotly-*.html"):
    for _f in glob.glob(f"{_PLOT_DIR}/{_pattern}"):
        try:
            if os.path.getmtime(_f) < _cutoff:
                os.remove(_f)
        except OSError:
            pass


def _next_plot_path(prefix, suffix):
    global _plot_count
    os.makedirs(_PLOT_DIR, exist_ok=True)
    # Keep root-created files host-writable.
    os.chmod(_PLOT_DIR, 0o777)
    _plot_count += 1
    return f"{_PLOT_DIR}/{prefix}-{os.getpid()}-{_plot_count:04d}.{suffix}"


def _show(*_args, **_kwargs):
    for num in _plt.get_fignums():
        path = _next_plot_path("fig", "png")
        _plt.figure(num).savefig(path, bbox_inches="tight")
        print(f"__OPEN_IMAGE__ {path}", flush=True)
    # Match notebook-cell behavior.
    _plt.close("all")


_plt.show = _show


def _enable_plotly_show():
    try:
        import plotly.io as pio
        from plotly.io.base_renderers import ExternalRenderer
    except ImportError:
        return

    # Plotly's own "browser" renderer serves from inside the container, out of the host's reach.
    class _ComintRenderer(ExternalRenderer):
        def __init__(self):
            # Plotly copies matching `show()` kwargs onto renderer attributes.
            self.config = None

        def render(self, fig_dict):
            path = _next_plot_path("plotly", "html")
            # One shared plotly.min.js beside the files instead of 4.5 MB per figure.
            pio.write_html(fig_dict, path, config=self.config, include_plotlyjs="directory")
            print(f"__OPEN_HTML__ {path}", flush=True)

    pio.renderers["comint"] = _ComintRenderer()
    pio.renderers.default = "comint"


_enable_plotly_show()


# rlcompleter probes `getattr(obj, name)` on every candidate: on a torch Tensor `.volatile`
# warns into python.el's completion stream and `.imag` raises, so `t.<TAB>` yields nothing.
class _QuietCompleter(rlcompleter.Completer):
    def attr_matches(self, text):
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                return super().attr_matches(text)
            except Exception:
                return self._dir_matches(text)

    def _dir_matches(self, text):
        """Match on `dir()` alone, reading no attribute off the object; callables lose their "(" postfix."""
        match = re.match(r"(\w+(\.\w+)*)\.(\w*)", text)
        if not match:
            return []
        expr, attr = match.group(1, 3)
        try:
            names = dir(eval(expr, self.namespace))  # ty: ignore[unresolved-attribute]
        except Exception:
            return []
        # rlcompleter's rule: hide privates until the prefix asks for them.
        hidden = "_" if attr == "" else "__" if attr == "_" else None
        return sorted(
            f"{expr}.{name}" for name in names if name.startswith(attr) and not (hidden and name.startswith(hidden))
        )


# `site` runs its readline hook after PYTHONSTARTUP, but its `import rlcompleter` is a no-op by then.
readline.set_completer(_QuietCompleter().complete)

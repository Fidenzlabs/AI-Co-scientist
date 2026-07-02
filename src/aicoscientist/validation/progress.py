"""Lightweight tqdm wrapper for the (long-running) validation pipeline.

Progress bars go to stderr with a modest ``mininterval`` so they stay readable when the
output is captured into a pipe (the Space streams stdout+stderr to the browser log and to
the container logs). Falls back to a no-op if tqdm is unavailable, so importing this never
breaks Tier-0 or a minimal install.
"""

from __future__ import annotations

import sys


class _NullBar:
    """Stand-in when tqdm is absent or a manual bar is requested without tqdm."""

    def update(self, *a, **k):
        pass

    def set_description(self, *a, **k):
        pass

    def set_postfix_str(self, *a, **k):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def pbar(iterable=None, *, desc: str = "", total=None):
    """Return a tqdm bar (iterable-wrapping or manual) tuned for captured pipes.

    ``pbar(seq, desc=...)`` wraps an iterable; ``pbar(desc=..., total=n)`` returns a manual
    bar you drive with ``.update(1)``. No-ops gracefully without tqdm.
    """
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        return iterable if iterable is not None else _NullBar()
    return tqdm(
        iterable, desc=desc, total=total, file=sys.stderr,
        mininterval=3.0, dynamic_ncols=True, leave=True,
    )

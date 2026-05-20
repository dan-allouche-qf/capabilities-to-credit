"""Light logging helper. INFO by default; module-tagged stream output."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "newperformers") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("newperformers")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)

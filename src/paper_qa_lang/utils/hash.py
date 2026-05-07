"""Hashing utilities for content deduplication and identification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


def md5sum(path: str | os.PathLike) -> str:
    """Compute the MD5 hex digest of a file's contents.

    Args:
        path: Path to the file.

    Returns:
        32-character MD5 hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path points to a directory.
    """
    path_obj = Path(path)
    if path_obj.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")
    h = hashlib.md5()
    with path_obj.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

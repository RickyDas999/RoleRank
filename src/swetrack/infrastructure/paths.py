"""Repository root resolution shared by config loaders and local storage paths.

Locates the root by walking up from this file until a directory containing
``pyproject.toml`` is found, rather than assuming a fixed number of parent
directories. A fixed-depth guess breaks silently whenever the package is
installed non-editable (e.g. inside the Docker image): the installed file
then lives under ``site-packages`` at a different depth than the ``src/``
layout used in a repository checkout, so the same parent-count lands
somewhere else entirely (observed: it resolved to ``/usr/local/lib/python3.11``
inside the container).

The ``SWETRACK_ROOT_DIR`` environment variable overrides the search outright,
for environments -- like the Docker image, where ``pyproject.toml`` is not
copied next to the installed package -- where no marker-based search would
succeed anyway.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("SWETRACK_ROOT_DIR")
    if env_root:
        return Path(env_root).resolve()

    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    raise RuntimeError(
        f"Could not locate the SWETrack repository root: no pyproject.toml found above {here}. "
        "Set the SWETRACK_ROOT_DIR environment variable to override (e.g. in Docker, where the "
        "installed package is not part of a repository checkout)."
    )

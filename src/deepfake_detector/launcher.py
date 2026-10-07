"""Environment identity for the single project launcher."""
from pathlib import Path


def needs_reexec(current_prefix, environment):
    """Virtual environments may share their executable with the base Python."""
    return Path(current_prefix).resolve() != Path(environment).resolve()

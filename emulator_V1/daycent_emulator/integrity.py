"""Keep generated outputs outside the saved model package."""
from pathlib import Path

RELEASE_ROOT = Path(__file__).resolve().parent.parent


def writable_output(path):
    path = Path(path).expanduser().resolve()
    if path == RELEASE_ROOT or RELEASE_ROOT in path.parents:
        raise ValueError(f'Use an output directory outside {RELEASE_ROOT.name} to preserve the saved models and results.')
    return path

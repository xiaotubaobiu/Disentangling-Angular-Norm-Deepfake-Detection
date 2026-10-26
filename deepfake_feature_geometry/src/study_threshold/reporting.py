"""Small shared helpers for the threshold-transfer scripts: paths, tables, figures.

Path handling rule for this phase: scripts resolve their own location, never the
current working directory. `phase_paths()` turns the phase directory plus the
YAML `paths` block into absolute paths, and `repo_path()` resolves the
repo-relative inputs declared in the YAML.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def repo_root(phase_dir: Path) -> Path:
    """`experiments/<phase>/` -> repository root."""
    return Path(phase_dir).resolve().parents[1]


def repo_path(phase_dir: Path, rel: str) -> Path:
    return repo_root(phase_dir) / rel


def phase_paths(phase_dir: Path, cfg: dict) -> dict:
    """Absolute output directories, created on demand."""
    phase_dir = Path(phase_dir).resolve()
    out = {name: phase_dir / rel for name, rel in cfg["paths"].items()}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def write_table(df: pd.DataFrame, path: Path, float_format: str = "%.6f") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)
    print(f"[write] {path}  ({len(df)} rows)")
    return path


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}", max_rows: int | None = None) -> str:
    """Markdown table without a hard dependency on `tabulate`."""
    show = df if max_rows is None else df.head(max_rows)
    cols = list(show.columns)

    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"| … ({len(df) - max_rows} more rows) |" + " |" * (len(cols) - 1))
    return "\n".join(lines)


def new_figure(figsize=(7, 4.5)):
    """Agg-backed figure (scripts must never need a display)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def save_figure(fig, path: Path, dpi: int = 150) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[figure] {path}")
    return path


def cell_id(backbone: str, seed: int) -> str:
    return f"{backbone}_raw_s{seed}"

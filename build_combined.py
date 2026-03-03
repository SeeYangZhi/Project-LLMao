"""
Merge notebooks 01-05 into one combined notebook.
Output: colab_package/sarcasm_classification.ipynb
"""
import json
from pathlib import Path

ROOT     = Path(".")
NB_DIR   = ROOT / "notebooks"
OUT_DIR  = ROOT / "colab_package"
OUT_DIR.mkdir(exist_ok=True)

# ── Load all notebooks ────────────────────────────────────────────────────────
def load_nb(name):
    path = NB_DIR / name
    nb = json.loads(path.read_bytes().decode("utf-8"))
    cells = []
    for c in nb["cells"]:
        src = c["source"]
        if isinstance(src, list):
            src = "".join(src)
        cells.append({"type": c["cell_type"], "src": src})
    return cells

nb01 = load_nb("01_data_preparation.ipynb")
nb02 = load_nb("02_tfidf_lr_baseline.ipynb")
nb03 = load_nb("03_naive_bayes_baseline.ipynb")
nb04 = load_nb("04_bert_classification.ipynb")
nb05 = load_nb("05_error_analysis.ipynb")


# ── Single combined setup cell ────────────────────────────────────────────────
# Merges all imports + file detection + all output-dir definitions

SETUP_SRC = "\n".join([
    "# ============================================================",
    "# SETUP — imports, file upload, paths",
    "# ============================================================",
    "from __future__ import annotations",
    "import json, hashlib, random, os, warnings, shutil",
    "from dataclasses import dataclass",
    "from pathlib import Path",
    "from collections import Counter",
    "from urllib.parse import urlparse",
    "from typing import Optional",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import matplotlib.gridspec as gridspec",
    "import seaborn as sns",
    "",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer",
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.naive_bayes import MultinomialNB, ComplementNB",
    "from sklearn.model_selection import GridSearchCV, GroupKFold",
    "from sklearn.metrics import (",
    "    accuracy_score, precision_score, recall_score,",
    "    f1_score, classification_report, confusion_matrix,",
    ")",
    "",
    "warnings.filterwarnings('ignore')",
    "",
    "SEED = 42",
    "random.seed(SEED)",
    "np.random.seed(SEED)",
    "",
    "# ── Locate or upload the JSONL data file ─────────────────────────────────",
    'FILENAME = "sarcasm_pairs_step35_clean.jsonl"',
    "",
    "def _locate_file(filename):",
    "    candidates = []",
    "    for root in [Path.cwd()] + list(Path.cwd().parents):",
    "        for sub in [",
    '            Path("data") / "processed" / filename,',
    '            Path("data") / filename,',
    "            Path(filename),",
    "        ]:",
    "            candidates.append(root / sub)",
    "    for p in [",
    '        Path("/content") / filename,',
    '        Path("/mnt/data") / filename,',
    "    ]:",
    "        candidates.append(p)",
    "    _c = Path('/content')",
    "    for p in (_c.rglob(filename) if _c.exists() else []):",
    "        candidates.append(p)",
    "    for p in candidates:",
    "        if p.is_file():",
    "            return p",
    "    return None",
    "",
    "print(f'cwd: {Path.cwd()}')",
    "print(f'files in cwd: {[p.name for p in Path.cwd().iterdir()][:10]}')",
    "",
    "DATA_FILE = _locate_file(FILENAME)",
    "if DATA_FILE is None:",
    "    try:",
    "        from google.colab import files as _cf",
    '        print(f"Upload {FILENAME!r}:")',
    "        _up = _cf.upload()",
    "        if not _up:",
    '            raise RuntimeError("No file uploaded.")',
    "        _name = list(_up.keys())[0]",
    '        DATA_FILE = Path("/content") / FILENAME',
    "        if Path(_name) != DATA_FILE:",
    "            shutil.move(_name, str(DATA_FILE))",
    '        print(f"Saved to {DATA_FILE}")',
    "    except ImportError:",
    "        raise FileNotFoundError(",
    '            f"Cannot find {FILENAME!r}. Place it in the same folder as this notebook."',
    "        )",
    "",
    "# ── Project root + all output directories ────────────────────────────────",
    "def _find_root(data_file):",
    '    for parent in [data_file.parent] + list(data_file.parents):',
    '        if any((parent / m).exists() for m in ["outputs","notebooks","data"]):',
    "            return parent",
    "    return data_file.parent",
    "",
    "ROOT = _find_root(DATA_FILE)",
    "",
    "OUT_DATASETS  = ROOT / 'outputs' / 'datasets'",
    "OUT_SPLITS    = ROOT / 'outputs' / 'splits'",
    "OUT_TFIDF     = ROOT / 'outputs' / 'classical' / 'tfidf_lr'",
    "OUT_NB        = ROOT / 'outputs' / 'classical' / 'naive_bayes'",
    "BERT_OUT      = ROOT / 'outputs' / 'bert'",
    "REPORTS_DIR   = ROOT / 'outputs' / 'reports'",
    "SPLITS        = OUT_SPLITS",
    "",
    "for d in [OUT_DATASETS, OUT_SPLITS, OUT_TFIDF, OUT_NB, BERT_OUT, REPORTS_DIR]:",
    "    d.mkdir(parents=True, exist_ok=True)",
    "",
    'print(f"Data   : {DATA_FILE}")',
    'print(f"Root   : {ROOT}")',
    'print(f"Output : {ROOT / \'outputs\'}")',
]) + "\n"

# Validate setup cell
try:
    compile(SETUP_SRC, "<setup>", "exec")
    print("Setup cell: OK")
except SyntaxError as e:
    lines = SETUP_SRC.split("\n")
    print(f"SyntaxError line {e.lineno}: {lines[e.lineno-1]!r} — {e}")
    raise


# ── Helper: make a code cell ──────────────────────────────────────────────────
def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }


# ── Determine which cells are "setup" cells to skip ──────────────────────────
# A setup cell is: first code cell in each notebook that defines ROOT/_find_
def is_setup_cell(src):
    triggers = [
        "_find_project_root",
        "_find_data_file",
        "_locate_file",
        "Colab / environment setup",
        "import json, hashlib, random",
        "SEED = 42\nrandom.seed",
    ]
    return any(t in src for t in triggers)


# ── Collect content cells from each notebook (skip setup cells) ───────────────
def content_cells(cells):
    """Return cells that are not setup cells."""
    out = []
    for c in cells:
        if c["type"] == "code" and is_setup_cell(c["src"]):
            continue
        # Patch nb02's OUT_DIR references to use OUT_TFIDF
        src = c["src"].replace(
            'ROOT / "outputs" / "classical" / "tfidf_lr"',
            "OUT_TFIDF"
        ).replace(
            "OUT_DIR", "OUT_TFIDF"
        ) if c["type"] == "code" else c["src"]

        out.append({"type": c["type"], "src": src})
    return out

def content_cells_nb(cells, out_dir_var, out_dir_val):
    """Return cells, replacing OUT_DIR with the right variable."""
    result = []
    for c in cells:
        if c["type"] == "code" and is_setup_cell(c["src"]):
            continue
        src = c["src"]
        if c["type"] == "code":
            src = src.replace("OUT_DIR", out_dir_var)
        result.append({"type": c["type"], "src": src})
    return result


# ── Build combined cell list ──────────────────────────────────────────────────
all_cells = []

# Title
all_cells.append(md_cell(
    "# Sarcasm Classification — Complete Pipeline\n\n"
    "**Sections**:\n"
    "1. Setup & Data Preparation\n"
    "2. TF-IDF + Logistic Regression Baseline\n"
    "3. Naive Bayes Baseline\n"
    "4. BERT / DistilBERT Classification\n"
    "5. Error Analysis & Model Comparison\n\n"
    "**Run all cells in order (Runtime → Run all).**"
))

# Setup cell
all_cells.append(code_cell(SETUP_SRC))

# ── Section 1: Data Preparation ───────────────────────────────────────────────
all_cells.append(md_cell("---\n# Part 1 — Data Preparation"))
for c in content_cells(nb01):
    if c["type"] == "markdown":
        all_cells.append(md_cell(c["src"]))
    else:
        all_cells.append(code_cell(c["src"]))

# ── Section 2: TF-IDF + LR ───────────────────────────────────────────────────
all_cells.append(md_cell("---\n# Part 2 — TF-IDF + Logistic Regression Baseline"))
for c in content_cells_nb(nb02, "OUT_TFIDF", "OUT_TFIDF"):
    if c["type"] == "markdown":
        all_cells.append(md_cell(c["src"]))
    else:
        all_cells.append(code_cell(c["src"]))

# ── Section 3: Naive Bayes ────────────────────────────────────────────────────
all_cells.append(md_cell("---\n# Part 3 — Naive Bayes Baseline"))
for c in content_cells_nb(nb03, "OUT_NB", "OUT_NB"):
    if c["type"] == "markdown":
        all_cells.append(md_cell(c["src"]))
    else:
        all_cells.append(code_cell(c["src"]))

# ── Section 4: BERT ───────────────────────────────────────────────────────────
all_cells.append(md_cell("---\n# Part 4 — BERT / DistilBERT Classification"))
for c in content_cells_nb(nb04, "BERT_OUT", "BERT_OUT"):
    if c["type"] == "markdown":
        all_cells.append(md_cell(c["src"]))
    else:
        all_cells.append(code_cell(c["src"]))

# ── Section 5: Error Analysis ─────────────────────────────────────────────────
all_cells.append(md_cell("---\n# Part 5 — Error Analysis & Model Comparison"))
for c in content_cells_nb(nb05, "REPORTS_DIR", "REPORTS_DIR"):
    if c["type"] == "markdown":
        all_cells.append(md_cell(c["src"]))
    else:
        # Also patch CLASSICAL and BERT_OUT references (already correct var names)
        all_cells.append(code_cell(c["src"]))

# ── Compile-check all code cells ──────────────────────────────────────────────
errors = []
for i, c in enumerate(all_cells):
    if c["cell_type"] == "code":
        src = c["source"]
        try:
            compile(src, f"cell{i}", "exec")
        except SyntaxError as e:
            errors.append((i, e.lineno, str(e), src.split("\n")[e.lineno-1] if e.lineno else ""))

if errors:
    print(f"\n{len(errors)} syntax error(s):")
    for i, ln, msg, line in errors:
        print(f"  Cell {i:3d} line {ln}: {line!r}")
        print(f"           {msg}")
else:
    print(f"All {len(all_cells)} cells: syntax OK")

# ── Write combined notebook ───────────────────────────────────────────────────
combined = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {"provenance": []},
    },
    "cells": all_cells,
}

out_nb = OUT_DIR / "sarcasm_classification.ipynb"
out_nb.write_bytes(json.dumps(combined, indent=1).encode("utf-8"))
print(f"\nSaved: {out_nb}")
print(f"Total cells: {len(all_cells)}")
print(f"\ncolab_package/ contents:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
print(f"\nAlso copy into colab_package/:")
print(f"  sarcasm_pairs_step35_clean.jsonl  (from data/processed/)")

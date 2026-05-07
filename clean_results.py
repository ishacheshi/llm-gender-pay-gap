"""
clean_results.py
----------------
Discovers every  results/{model}/{prompt}/{date}/raw_results.csv
produced by collect_results.py, parses each raw_response into a numeric
salary, and writes a single aggregated cleaned_results.csv at the project
root for the analysis notebook.

Also handles the legacy root-level raw_results.csv (backfills model /
prompt_name for that old single-provider format).

Output columns: resume_gender, salary, model, prompt_name

Edge cases handled:
  - Currency symbols  ($120,000  →  120000)
  - Commas            (120,000   →  120000)
  - K / k suffix      (120K      →  120000)
  - Ranges            (110–130K  →  120000, midpoint)
  - Extra prose       ("approximately $120,000" → 120000)
  - Blanks / errors   → dropped from cleaned output
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR     = Path(__file__).parent
RESULTS_DIR  = BASE_DIR / "results"
LEGACY_FILE  = BASE_DIR / "raw_results.csv"   # pre-directory-structure runs
CLEAN_FILE   = BASE_DIR / "cleaned_results.csv"

CLEAN_COLS = ["resume_gender", "salary", "model", "prompt_name"]


# ---------------------------------------------------------------------------
# Salary parser
# ---------------------------------------------------------------------------

_KSFX = re.compile(r"([\d,]+(?:\.\d+)?)\s*[Kk]\b")

_RANGE = re.compile(
    r"[$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*[Kk]?"
    r"\s*[-–—to]+\s*"
    r"[$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*[Kk]?",
    re.IGNORECASE,
)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_salary(raw: Optional[str]) -> Optional[float]:
    """Return a single float salary, or None if unparseable.

    Handles non-string inputs (int/float) that pandas may infer when the model
    responds with a bare number like 140000.
    """
    # Coerce to string; treat pandas NaN / None as missing.
    if raw is None:
        return None
    try:
        raw = str(raw).strip()
    except Exception:
        return None
    if not raw or raw.upper() in {"", "ERROR", "N/A", "NONE", "NULL", "NAN"}:
        return None

    text = re.sub(r"[$€£¥]", "", raw)
    text = re.sub(r"\s+", " ", text)

    # ── Range → midpoint ──────────────────────────────────────────────────
    m = _RANGE.search(text)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if re.search(r"\d\s*[Kk]\b", m.group(0), re.IGNORECASE):
            if lo < 10_000: lo *= 1_000
            if hi < 10_000: hi *= 1_000
        return (lo + hi) / 2

    # ── Single number with K suffix ───────────────────────────────────────
    m = _KSFX.search(text)
    if m:
        val = _to_float(m.group(1))
        return val * 1_000 if val < 10_000 else val

    # ── Plain number ──────────────────────────────────────────────────────
    m = re.search(r"([\d,]+(?:\.\d+)?)", text)
    if m:
        return _to_float(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _latest_run_csvs() -> list[Path]:
    """
    For each (model, prompt_name) pair, return only the CSV from the most
    recent run directory (sorted lexicographically: 2026-05-07 > 2026-05-06,
    2026-05-07_2 > 2026-05-07).  Older runs under the same model/prompt are
    silently skipped, preventing different prompt versions from being mixed.
    """
    all_csvs = sorted(RESULTS_DIR.glob("*/*/*/raw_results.csv"))
    # Group by (model_dir, prompt_dir) — the two path components above the date.
    groups: dict[tuple[str, str], list[Path]] = {}
    for csv_path in all_csvs:
        key = (csv_path.parts[-4], csv_path.parts[-3])   # (model, prompt_name)
        groups.setdefault(key, []).append(csv_path)
    # Keep only the last (latest) path in each group.
    return [sorted(paths, key=lambda p: (len(p.parts[-2]), p.parts[-2]))[-1]
            for paths in groups.values()]


def _load_run_csvs() -> pd.DataFrame:
    """
    Load only the latest run per (model, prompt_name) and concatenate.
    Also loads the legacy root-level raw_results.csv when no run dirs exist.
    """
    frames: list[pd.DataFrame] = []

    # ── New-style run directories — latest run only ──────────────────────────
    run_csvs = _latest_run_csvs()
    for path in run_csvs:
        df = pd.read_csv(path)
        print(f"  {path.relative_to(BASE_DIR)}  ({len(df):,} rows)")
        frames.append(df)

    # ── Legacy root-level CSV (old single-provider format) ───────────────────
    if LEGACY_FILE.exists():
        df_legacy = pd.read_csv(LEGACY_FILE)
        if "model" not in df_legacy.columns:
            df_legacy["model"] = "claude-sonnet-4-6"
        if "prompt_name" not in df_legacy.columns:
            df_legacy["prompt_name"] = "salary_numerical"
        # Only include if not already represented in the run directories.
        if not run_csvs:
            print(f"  {LEGACY_FILE.name}  ({len(df_legacy):,} rows)  [legacy]")
            frames.append(df_legacy)
        else:
            print(
                f"  Skipping {LEGACY_FILE.name} (run directories present; "
                "delete the legacy file if it's already been migrated)."
            )

    if not frames:
        raise FileNotFoundError(
            "No raw_results.csv found.\n"
            f"  Looked in: {RESULTS_DIR}/*/*/*/raw_results.csv\n"
            f"  Legacy:    {LEGACY_FILE}\n"
            "Run collect_results.py first."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Discovering raw result files …")
    df = _load_run_csvs()
    print(f"\nTotal rows loaded: {len(df):,}")

    models  = sorted(df["model"].unique())
    prompts = sorted(df["prompt_name"].unique())
    print(f"\nModels in data:  {models}")
    print(f"Prompts in data: {prompts}")

    df["salary"] = df["raw_response"].apply(parse_salary)

    n_total  = len(df)
    n_failed = df["salary"].isna().sum()
    n_parsed = n_total - n_failed
    print(f"\nParsed:  {n_parsed:,} / {n_total:,}  |  Failed: {n_failed:,}")

    if n_failed > 0:
        print("\nSample of unparseable responses:")
        bad = df.loc[df["salary"].isna(), "raw_response"].value_counts().head(20)
        for resp, cnt in bad.items():
            print(f"  [{cnt:4d}×]  {repr(resp)}")

    valid = df["salary"].dropna()
    too_low  = (valid < 20_000).sum()
    too_high = (valid > 2_000_000).sum()
    if too_low:
        print(f"\n⚠  {too_low} salaries below $20,000 — may indicate a parsing error.")
    if too_high:
        print(f"\n⚠  {too_high} salaries above $2,000,000 — may indicate a parsing error.")

    # ── Write cleaned file ────────────────────────────────────────────────
    clean = df.loc[df["salary"].notna(), CLEAN_COLS].copy()
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"\nWrote {len(clean):,} rows → {CLEAN_FILE.name}")

    # ── Per-model × gender summary ────────────────────────────────────────
    print("\n── Summary by model × gender ─────────────────────────────────────────")
    summary = (
        clean.groupby(["model", "prompt_name", "resume_gender"])["salary"]
        .agg(N="count", mean="mean", median="median", std="std")
        .rename(columns={"N": "N"})
    )
    fmt = summary.copy()
    for col in ["mean", "median", "std"]:
        fmt[col] = fmt[col].apply(lambda v: f"${v:,.0f}")
    print(fmt.to_string())

    # ── Mean gap summary ──────────────────────────────────────────────────
    print("\n── Mean salary gap (male − female) by model + prompt ─────────────────")
    pivot = (
        clean.groupby(["model", "prompt_name", "resume_gender"])["salary"]
        .mean()
        .unstack("resume_gender")
    )
    if "male" in pivot.columns and "female" in pivot.columns:
        pivot["gap_$"] = pivot["male"] - pivot["female"]
        pivot["gap_%"] = 100 * pivot["gap_$"] / pivot["female"]
        out = pd.DataFrame({
            "female":  pivot["female"].apply(lambda v: f"${v:,.0f}"),
            "male":    pivot["male"].apply(lambda v: f"${v:,.0f}"),
            "gap_$":   pivot["gap_$"].apply(lambda v: f"${v:+,.0f}"),
            "gap_%":   pivot["gap_%"].apply(lambda v: f"{v:+.2f}%"),
        })
        print(out.to_string())


if __name__ == "__main__":
    main()

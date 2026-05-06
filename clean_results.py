"""
clean_results.py
----------------
Reads raw_results.csv, parses each raw_response into a numeric salary,
and writes cleaned_results.csv with columns: resume_gender, salary.

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

BASE_DIR   = Path(__file__).parent
RAW_FILE   = BASE_DIR / "raw_results.csv"
CLEAN_FILE = BASE_DIR / "cleaned_results.csv"


# ---------------------------------------------------------------------------
# Salary parser
# ---------------------------------------------------------------------------

# Matches a number (with optional commas and decimal) followed optionally by K/k.
_NUM  = r"([\d,]+(?:\.\d+)?)\s*[Kk]?"
_FULL = re.compile(r"[$€£¥]?\s*" + _NUM)
_KSFX = re.compile(r"([\d,]+(?:\.\d+)?)\s*[Kk]\b")

# Matches two numbers separated by a range indicator.
_RANGE = re.compile(
    r"[$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*[Kk]?"
    r"\s*[-–—to]+\s*"
    r"[$€£¥]?\s*([\d,]+(?:\.\d+)?)\s*[Kk]?",
    re.IGNORECASE,
)


def _strip_to_float(s: str) -> float:
    """Remove commas and cast to float."""
    return float(s.replace(",", ""))


def parse_salary(raw: Optional[str]) -> Optional[float]:
    """Return a single float salary, or None if unparseable."""
    if not raw or raw.strip().upper() in {"", "ERROR", "N/A", "NONE", "NULL"}:
        return None

    # Remove currency symbols; collapse whitespace.
    text = re.sub(r"[$€£¥]", "", raw.strip())
    text = re.sub(r"\s+", " ", text)

    # ── Range: "110,000 - 130,000" or "110K-130K" → midpoint ──────────────
    m = _RANGE.search(text)
    if m:
        lo_raw = m.group(1)
        hi_raw = m.group(2)
        lo = _strip_to_float(lo_raw)
        hi = _strip_to_float(hi_raw)

        # Apply K multiplier if K appeared anywhere in the matched segment.
        segment = m.group(0)
        has_k   = bool(re.search(r"\d\s*[Kk]\b", segment, re.IGNORECASE))
        if has_k:
            if lo < 10_000:
                lo *= 1_000
            if hi < 10_000:
                hi *= 1_000

        return (lo + hi) / 2

    # ── Single value with explicit K suffix: "120K" → 120000 ───────────────
    m = _KSFX.search(text)
    if m:
        val = _strip_to_float(m.group(1))
        if val < 10_000:
            val *= 1_000
        return val

    # ── Plain number (may include commas): "120,000" → 120000 ──────────────
    m = re.search(r"([\d,]+(?:\.\d+)?)", text)
    if m:
        return _strip_to_float(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df):,} rows from {RAW_FILE.name}")

    df["salary"] = df["raw_response"].apply(parse_salary)

    n_total   = len(df)
    n_failed  = df["salary"].isna().sum()
    n_parsed  = n_total - n_failed

    print(f"Parsed:  {n_parsed:,} / {n_total:,}")
    print(f"Failed:  {n_failed:,}")

    if n_failed > 0:
        print("\nSample of unparseable responses:")
        bad = df.loc[df["salary"].isna(), "raw_response"].value_counts().head(20)
        for resp, cnt in bad.items():
            print(f"  [{cnt:4d}x]  {repr(resp)}")

    # Warn about suspicious outliers (but keep all values).
    valid = df["salary"].dropna()
    too_low  = (valid < 20_000).sum()
    too_high = (valid > 2_000_000).sum()
    if too_low:
        print(f"\n⚠  {too_low} salaries below $20,000 — may indicate parsing error.")
    if too_high:
        print(f"\n⚠  {too_high} salaries above $2,000,000 — may indicate parsing error.")

    # Write cleaned file (drop unparseable rows).
    clean = df.loc[df["salary"].notna(), ["resume_gender", "salary"]].copy()
    clean.to_csv(CLEAN_FILE, index=False)
    print(f"\nWrote {len(clean):,} rows → {CLEAN_FILE.name}")

    print("\n── Summary by gender ─────────────────────────────────────────────────")
    summary = (
        clean.groupby("resume_gender")["salary"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .rename(columns={"n": "N"})
    )
    summary["mean"]   = summary["mean"].map("${:,.0f}".format)
    summary["median"] = summary["median"].map("${:,.0f}".format)
    summary["std"]    = summary["std"].map("${:,.0f}".format)
    summary["min"]    = summary["min"].map("${:,.0f}".format)
    summary["max"]    = summary["max"].map("${:,.0f}".format)
    print(summary.to_string())


if __name__ == "__main__":
    main()

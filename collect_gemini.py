"""
collect_gemini.py
-----------------
Gemini-specific collection script for the LLM salary experiment.

Enforces strict 12s spacing between API calls to stay within Google's
free-tier limit of 5 requests per minute on gemini-3-flash-preview.

Writes to the same results/ directory structure as collect_results.py,
so clean_results.py picks it up automatically.

Usage:
    python collect_gemini.py --n 20       # test run  (40 calls, ~8 min)
    python collect_gemini.py              # full run  (3000 calls, ~10h)
    python collect_gemini.py --fresh      # new run directory
    python collect_gemini.py --prompt salary_numerical   # explicit prompt
"""

import argparse
import csv
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as google_errors

from collect_results import (
    BASE_DIR, FIELDNAMES, RESUMES,
    _append_row, _extract_pdf_text, _write_config,
    resolve_run_dir,
)
from config import get_config
from prompts import DEFAULT_PROMPT, PROMPTS

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL       = "gemini-3-flash"
_CFG        = get_config(MODEL)
API_MODEL   = _CFG.get("api_model_id", MODEL)
INTERVAL    = _CFG["delay"]
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Guarantee at least INTERVAL seconds between the *start* of each call."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._last     = 0.0          # monotonic time of last call start

    def wait(self) -> None:
        now     = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()  # record just before the API call


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call(
    client:      genai.Client,
    resume_text: str,
    gender:      str,
    prompt:      str,
    prompt_name: str,
    rate:        _RateLimiter,
) -> dict:
    rate.wait()
    response = client.models.generate_content(
        model=API_MODEL,
        contents=f"{resume_text}\n\n{prompt}",
    )
    raw = (response.text or "").strip()
    return {
        "resume_gender": gender,
        "raw_response":  raw,
        "model":         MODEL,
        "prompt_name":   prompt_name,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the salary experiment for Gemini, "
            "respecting the free-tier 5 RPM rate limit."
        )
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help=f"Named prompt from prompts.py (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--n", type=int, default=1500,
        help="API calls per resume (default: 1500)",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start a new run directory rather than resuming the latest one",
    )
    args = parser.parse_args()

    prompt_name = args.prompt
    n_per       = args.n

    if prompt_name not in PROMPTS:
        raise ValueError(
            f"Unknown prompt '{prompt_name}'. Available: {list(PROMPTS)}"
        )
    prompt = PROMPTS[prompt_name]

    # ── Directories ──────────────────────────────────────────────────────────
    date_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir     = resolve_run_dir(MODEL, prompt_name, date_str, fresh=args.fresh)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_csv  = run_dir / "raw_results.csv"
    config_json = run_dir / "config.json"

    resuming   = output_csv.exists()
    started_at = datetime.now(timezone.utc).isoformat()

    _write_config(
        config_json,
        model=MODEL,
        provider="google",
        prompt_name=prompt_name,
        prompt_text=prompt,
        n_per_resume=n_per,
        date_str=date_str,
        started_at=started_at,
    )

    if not resuming:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # ── API client ───────────────────────────────────────────────────────────
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    rate   = _RateLimiter(INTERVAL)

    # ── Resume data ──────────────────────────────────────────────────────────
    print("Loading resume PDFs …")
    resume_texts = {g: _extract_pdf_text(p) for g, p in RESUMES.items()}
    print("  OK\n")

    # ── Task list (resume-aware) ─────────────────────────────────────────────
    already_done = 0
    if resuming:
        already_done = max(0, sum(1 for _ in open(output_csv)) - 1)

    seed = hash((MODEL, prompt_name, str(run_dir))) & 0xFFFF_FFFF
    rng  = random.Random(seed)
    tasks: list[tuple[str, str]] = (
        [("female", resume_texts["female"])] * n_per +
        [("male",   resume_texts["male"])]   * n_per
    )
    rng.shuffle(tasks)
    tasks_remaining = tasks[already_done:]
    total = len(tasks)

    eta_min = len(tasks_remaining) * INTERVAL / 60
    print(f"Run directory : {run_dir.relative_to(BASE_DIR)}")
    print(f"Mode          : {'resume' if resuming else 'fresh'}" +
          (f"  (skipping first {already_done} rows)" if resuming else ""))
    print(f"Model         : {MODEL}  →  {API_MODEL}")
    print(f"Prompt        : {prompt_name}")
    print(f"Calls         : {total} total  ({n_per}/resume)  "
          f"|  remaining: {len(tasks_remaining)}")
    rpm = 60 / INTERVAL
    print(f"Rate limit    : 1 call / {INTERVAL}s  ({rpm:.0f} RPM)")
    print(f"Est. time     : {eta_min:.0f} min  ({eta_min/60:.1f}h)")
    print()

    start  = time.time()
    errors = 0
    counts = {"female": 0, "male": 0}

    for i, (gender, text) in enumerate(tasks_remaining, already_done + 1):
        success = False
        retries = 0

        while not success and retries < MAX_RETRIES:
            try:
                row = _call(client, text, gender, prompt, prompt_name, rate)
                _append_row(output_csv, row)
                counts[gender] += 1
                success = True

            except (google_errors.ClientError, google_errors.ServerError) as exc:
                retries += 1
                msg  = str(exc)
                code = msg.split()[0] if msg[:3].isdigit() else "???"
                # 429 = rate limit (wait longer), 503 = transient overload
                wait = 60 if "429" in msg else 30
                print(
                    f"\n  [Google {code}] waiting {wait}s"
                    f"  (attempt {retries}/{MAX_RETRIES}) …",
                    flush=True,
                )
                time.sleep(wait)

        if not success:
            errors += 1
            _append_row(output_csv, {
                "resume_gender": gender,
                "raw_response":  "ERROR",
                "model":         MODEL,
                "prompt_name":   prompt_name,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            })

        if i % 20 == 0 or i == total:
            elapsed    = time.time() - start
            done_here  = i - already_done
            rate_actual = done_here / max(elapsed, 1e-9)
            eta        = (total - i) / rate_actual if rate_actual > 0 else 0
            print(
                f"[{i:5d}/{total}]  "
                f"F={counts['female']:5d}  M={counts['male']:5d}  "
                f"elapsed {elapsed/60:6.1f}m  ETA {eta/3600:5.1f}h  "
                f"errors {errors}",
                flush=True,
            )

    elapsed_total = time.time() - start

    _write_config(
        config_json,
        model=MODEL,
        provider="google",
        prompt_name=prompt_name,
        prompt_text=prompt,
        n_per_resume=n_per,
        date_str=date_str,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed_total,
        n_success=total - errors,
        n_errors=errors,
    )

    print(f"\nDone in {elapsed_total/60:.1f} min.")
    print(f"Successes: {total - errors}  |  Errors: {errors}")
    print(f"Results   → {output_csv.relative_to(BASE_DIR)}")
    print(f"Config    → {config_json.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

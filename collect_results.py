"""
collect_results.py
------------------
Multi-provider resume salary experiment.

Each run writes to its own self-documenting directory:
    results/{model}/{prompt_name}/{YYYY-MM-DD}/
        raw_results.csv   — one row per API call
        config.json       — full config snapshot (written at start, updated at end)

Usage:
    # Quick sanity check (20 calls per resume)
    python collect_results.py --model claude-sonnet-4-6 --n 20

    # Full Anthropic run
    python collect_results.py --model claude-sonnet-4-6

    # Full OpenAI run (separate directory, same date folder)
    python collect_results.py --model gpt-4.1

    # Different prompt variant
    python collect_results.py --model gpt-4.1 --prompt salary_market

    # New run (old results preserved; new directory gets a _2, _3 … suffix)
    python collect_results.py --model claude-sonnet-4-6 --fresh

    # Resume an interrupted same-day run (default — appends to latest directory)
    python collect_results.py --model claude-sonnet-4-6

Provider detection:
    "claude-*"  → Anthropic SDK (PDF document type, prompt caching)
    "gpt-*"     → OpenAI SDK   (PDF text extraction, chat completions)

Resume content delivered to each provider:
    Anthropic: base64-encoded PDF (native document type, cached)
    OpenAI:    text extracted from the same PDF (pypdf)
Both contain the same resume text; only the name differs between the two resumes.

OPENAI_API_KEY must be set in .env alongside ANTHROPIC_API_KEY.
"""

import argparse
import base64
import csv
import json
import os
import re
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from google import genai as google_genai
from google.genai import errors as google_errors
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

from config import get_config, get_provider
from prompts import DEFAULT_PROMPT, PROMPTS

# Load .env before SDK clients are created (they read from os.environ).
load_dotenv()

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

RESUMES = {
    "female": BASE_DIR / "resumes" / "female" / "Ishani_Cheshire_resume.pdf",
    "male":   BASE_DIR / "resumes" / "male"   / "Ishan_Cheshire_resume.pdf",
}

FIELDNAMES = ["resume_gender", "raw_response", "model", "prompt_name", "timestamp"]

MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Run-directory helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Make a model or prompt name safe for use as a directory component."""
    return re.sub(r"[^\w.\-]", "_", name)


def _dir_sort_key(d: Path) -> tuple:
    """Natural sort key for run directories: (date, run_number).

    Handles both  YYYY-MM-DD  and  YYYY-MM-DD_N  directory names so that
    cross-date resume always picks the globally most recent run.
    """
    parts = d.name.split("_", 1)
    date_part = parts[0]                             # "YYYY-MM-DD"
    run_num   = int(parts[1]) if len(parts) > 1 else 0
    return (date_part, run_num)


def resolve_run_dir(model: str, prompt_name: str, date_str: str, fresh: bool = False) -> Path:
    """Return the run directory for this model/prompt.

    Default (fresh=False): resume the globally most recent run directory,
    regardless of date — so an interrupted run is always picked up even after
    the calendar date rolls over.

    With fresh=True: create a new directory for today, preserving old results.
        First run today  →  …/{date}/
        Second run today →  …/{date}_2/
        Third run today  →  …/{date}_3/   etc.
    """
    base = RESULTS_DIR / _sanitize(model) / _sanitize(prompt_name)

    if not fresh:
        # Resume: search across all dates and return the most recent directory.
        all_dirs = sorted(
            [d for d in base.glob("*") if d.is_dir()],
            key=_dir_sort_key,
        )
        return all_dirs[-1] if all_dirs else base / date_str

    # Fresh: create the next available directory for today's date.
    today_dirs = sorted(
        [d for d in base.glob(f"{date_str}*") if d.is_dir()],
        key=_dir_sort_key,
    )
    if not today_dirs:
        return base / date_str
    return base / f"{date_str}_{len(today_dirs) + 1}"


# ---------------------------------------------------------------------------
# Resume loading
# ---------------------------------------------------------------------------

def _load_pdf_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()


# ---------------------------------------------------------------------------
# Provider-specific API call functions
# ---------------------------------------------------------------------------

def _call_anthropic(
    client: anthropic.Anthropic,
    pdf_b64: str,
    gender: str,
    model: str,
    prompt: str,
    prompt_name: str,
    max_tokens: int,
) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {
                    # cache_control on the last stable block caches the whole
                    # prefix (document + prompt) — ~90% savings on repeated calls.
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        }],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "").strip()
    return _make_row(gender, raw, model, prompt_name)


def _call_openai(
    client: openai.OpenAI,
    resume_text: str,
    gender: str,
    model: str,
    prompt: str,
    prompt_name: str,
    max_tokens: int,
    use_max_completion_tokens: bool = False,
) -> dict:
    token_kwargs = (
        {"max_completion_tokens": max_tokens}
        if use_max_completion_tokens
        else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{resume_text}\n\n{prompt}",
        }],
        **token_kwargs,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _make_row(gender, raw, model, prompt_name)


def _call_google(
    client: google_genai.Client,
    resume_text: str,
    gender: str,
    api_model_id: str,   # actual API model identifier (e.g. "models/gemini-3-flash-preview")
    model_label: str,    # label stored in CSV (e.g. "gemini-3-flash")
    prompt: str,
    prompt_name: str,
) -> dict:
    response = client.models.generate_content(
        model=api_model_id,
        contents=f"{resume_text}\n\n{prompt}",
    )
    raw = (response.text or "").strip()
    return _make_row(gender, raw, model_label, prompt_name)


def _make_row(gender: str, raw: str, model: str, prompt_name: str) -> dict:
    return {
        "resume_gender": gender,
        "raw_response":  raw,
        "model":         model,
        "prompt_name":   prompt_name,
        "timestamp":     datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# CSV and config-snapshot helpers
# ---------------------------------------------------------------------------

def _init_csv(path: Path, fresh: bool) -> None:
    """Write a fresh header (new or overwrite), or leave existing file for resume."""
    if fresh or not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def _append_row(path: Path, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def _write_config(
    path: Path,
    *,
    model: str,
    provider: str,
    prompt_name: str,
    prompt_text: str,
    n_per_resume: int,
    date_str: str,
    started_at: str,
    completed_at: str | None = None,
    elapsed_seconds: float | None = None,
    n_success: int | None = None,
    n_errors: int | None = None,
) -> None:
    snapshot = {
        "model":            model,
        "provider":         provider,
        "prompt_name":      prompt_name,
        "prompt_text":      prompt_text,
        "n_per_resume":     n_per_resume,
        "n_total":          n_per_resume * len(RESUMES),
        "date":             date_str,
        "started_at":       started_at,
        "completed_at":     completed_at,
        "elapsed_seconds":  round(elapsed_seconds, 1) if elapsed_seconds is not None else None,
        "n_success":        n_success,
        "n_errors":         n_errors,
        "output_csv":       "raw_results.csv",
    }
    path.write_text(json.dumps(snapshot, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM salary experiment for a given model and prompt."
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Model ID (default: claude-sonnet-4-6). Provider is auto-detected.",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help=f"Named prompt from prompts.py (default: {DEFAULT_PROMPT}).",
    )
    parser.add_argument(
        "--n", type=int, default=1500,
        help="API calls per resume (default: 1500).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Start a new run directory (old results preserved with _2, _3 … suffix).",
    )
    args = parser.parse_args()

    model       = args.model
    prompt_name = args.prompt
    n_per       = args.n

    if prompt_name not in PROMPTS:
        raise ValueError(
            f"Unknown prompt '{prompt_name}'. Available: {list(PROMPTS)}"
        )

    prompt        = PROMPTS[prompt_name]
    cfg           = get_config(model)
    provider      = cfg["provider"]
    delay         = cfg["delay"]
    max_tok       = cfg["max_tokens"]
    use_mct       = cfg.get("max_completion_tokens", False)
    api_model_id  = cfg.get("api_model_id", model)  # falls back to model name

    # ── Resolve and create the run directory ────────────────────────────────
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir  = resolve_run_dir(model, prompt_name, date_str, fresh=args.fresh)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_csv  = run_dir / "raw_results.csv"
    config_json = run_dir / "config.json"

    # Fresh always produces a brand-new directory; resume reuses the latest.
    resuming = output_csv.exists()
    mode = "resume" if resuming else "fresh"

    started_at = datetime.now(timezone.utc).isoformat()

    # Write the initial config snapshot (completed_at / stats filled in later).
    _write_config(
        config_json,
        model=model,
        provider=provider,
        prompt_name=prompt_name,
        prompt_text=prompt,
        n_per_resume=n_per,
        date_str=date_str,
        started_at=started_at,
    )

    # ── Initialise the provider client ──────────────────────────────────────
    if provider == "anthropic":
        client = anthropic.Anthropic()
    elif provider == "openai":
        client = openai.OpenAI()
    elif provider == "google":
        client = google_genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        raise ValueError(f"Unknown provider '{provider}'")

    # ── Load resume data in the format each provider expects ────────────────
    print("Loading resume PDFs …")
    if provider == "anthropic":
        resume_data: dict[str, str] = {
            g: _load_pdf_b64(p) for g, p in RESUMES.items()
        }

        def call_fn(gender: str, data: str) -> dict:
            return _call_anthropic(
                client, data, gender, model, prompt, prompt_name, max_tok  # type: ignore[arg-type]
            )
    elif provider == "openai":
        resume_data = {g: _extract_pdf_text(p) for g, p in RESUMES.items()}

        def call_fn(gender: str, data: str) -> dict:
            return _call_openai(
                client, data, gender, model, prompt, prompt_name, max_tok,  # type: ignore[arg-type]
                use_max_completion_tokens=use_mct,
            )
    else:  # google
        resume_data = {g: _extract_pdf_text(p) for g, p in RESUMES.items()}

        def call_fn(gender: str, data: str) -> dict:
            return _call_google(
                client, data, gender, api_model_id, model, prompt, prompt_name  # type: ignore[arg-type]
            )
    print("  OK\n")

    # ── If resuming, skip rows already written ───────────────────────────────
    already_done = 0
    if resuming:
        already_done = sum(1 for _ in open(output_csv)) - 1  # subtract header
        already_done = max(already_done, 0)

    # ── Build the randomised task list ──────────────────────────────────────
    # Seed from the run directory path so each run has its own reproducible
    # order, and resume attempts reconstruct the exact same shuffle.
    seed = hash((model, prompt_name, str(run_dir))) & 0xFFFF_FFFF
    rng  = random.Random(seed)
    tasks = (
        [("female", resume_data["female"])] * n_per +
        [("male",   resume_data["male"])]   * n_per
    )
    rng.shuffle(tasks)
    total = len(tasks)

    # Skip tasks that were already written in a previous (interrupted) run.
    tasks_remaining = tasks[already_done:]

    _init_csv(output_csv, args.fresh)

    print(f"Run directory : {run_dir.relative_to(BASE_DIR)}")
    print(f"Mode          : {mode}" +
          (f"  (skipping first {already_done} rows)" if resuming else ""))
    print(f"Model         : {model}  |  Provider: {provider}")
    print(f"Prompt        : {prompt_name}")
    print(f"Calls         : {total} total  ({n_per}/resume)  |  Delay: {delay}s")
    print()

    start  = time.time()
    errors = 0
    counts = {"female": 0, "male": 0}

    for i, (gender, data) in enumerate(tasks_remaining, already_done + 1):
        success = False
        retries = 0

        while not success and retries < MAX_RETRIES:
            try:
                row = call_fn(gender, data)
                _append_row(output_csv, row)
                counts[gender] += 1
                success = True

            except (anthropic.RateLimitError, openai.RateLimitError):
                retries += 1
                wait = min(60 * retries, 300)
                print(
                    f"\n  [rate-limit] waiting {wait}s "
                    f"(attempt {retries}/{MAX_RETRIES}) …",
                    flush=True,
                )
                time.sleep(wait)

            except google_errors.ClientError as exc:
                retries += 1
                # Respect the retry-after hint from Google's error when present.
                wait = 60 if "429" in str(exc) else 10 * retries
                print(
                    f"\n  [Google error] {str(exc)[:120]} "
                    f"— waiting {wait}s (attempt {retries}/{MAX_RETRIES}) …",
                    flush=True,
                )
                time.sleep(wait)

            except (anthropic.APIError, openai.APIError) as exc:
                retries += 1
                print(
                    f"\n  [API error] {exc} "
                    f"(attempt {retries}/{MAX_RETRIES}) …",
                    flush=True,
                )
                time.sleep(10 * retries)

        if not success:
            errors += 1
            _append_row(output_csv, {
                "resume_gender": gender,
                "raw_response":  "ERROR",
                "model":         model,
                "prompt_name":   prompt_name,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            })

        if i % 100 == 0 or i == total:
            elapsed = time.time() - start
            rate    = (i - already_done) / max(elapsed, 1e-9)
            eta     = (total - i) / rate
            print(
                f"[{i:5d}/{total}]  "
                f"F={counts['female']:5d}  M={counts['male']:5d}  "
                f"elapsed {elapsed/60:5.1f}m  ETA {eta/60:5.1f}m  "
                f"errors {errors}",
                flush=True,
            )

        if i < total:
            time.sleep(delay)

    elapsed_total = time.time() - start
    total_success = total - already_done - errors

    # ── Overwrite config.json with final stats ───────────────────────────────
    _write_config(
        config_json,
        model=model,
        provider=provider,
        prompt_name=prompt_name,
        prompt_text=prompt,
        n_per_resume=n_per,
        date_str=date_str,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=elapsed_total,
        n_success=total_success + already_done,
        n_errors=errors,
    )

    print(f"\nDone in {elapsed_total/60:.1f} min.")
    print(f"Successes: {total_success + already_done}  |  Errors: {errors}")
    print(f"Results   → {output_csv.relative_to(BASE_DIR)}")
    print(f"Config    → {config_json.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()

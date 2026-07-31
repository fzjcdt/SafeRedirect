#!/usr/bin/env python3
"""
Run all experiments: run_safe_redirect → extract → judge for every combination,
then collect results into a CSV.
"""

import io
import json
import re
import subprocess
import sys
import csv
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_PATH = SCRIPT_DIR / f"experiment_results_{TIMESTAMP}.csv"
LOG_PATH = SCRIPT_DIR / f"experiment_log_{TIMESTAMP}.log"

_log_file = None


class Tee:
    """Write to both console and log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging():
    global _log_file
    _log_file = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, _log_file)
    sys.stderr = Tee(sys.__stderr__, _log_file)

MODELS = [
    "x-ai/grok-4.1-fast",
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4.5",
    "z-ai/glm-5",
    "google/gemini-2.5-pro",
    "minimax/minimax-m2.7",
    "moonshotai/kimi-k2.5",
]

DEFENSES = ["none", "spd", "safe-redirect"]

TASKS = [
    ("ai-guard", 0),
    ("ai-detoxify", 0),
    ("ai-outlier", 5),
]

SAFE_REDIRECT_VERSION = 1
CONCURRENT = 10


def result_file_path(model: str, task: str, samples: int, defense: str) -> Path:
    model_slug = model.replace("/", "-")
    suffix_map = {"none": "", "spd": "_spd"}
    if defense == "safe-redirect":
        suffix = f"_safe_redirect_{SAFE_REDIRECT_VERSION}"
    else:
        suffix = suffix_map.get(defense, f"_{defense.replace('-', '_')}")
    return RESULTS_DIR / model_slug / "jbb" / task / f"{samples}sample{suffix}.json"


def run_cmd(cmd: list) -> str:
    """Run a command, capture combined output, return it."""
    print(f"\n{'─' * 60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 60}")
    proc = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = proc.stdout + proc.stderr
    print(output)
    return output


def parse_run_safe_redirect(output: str) -> dict:
    """Parse: Success: X, Errors: Y, Empty: Z"""
    m = re.search(r"Success:\s*(\d+),\s*Errors:\s*(\d+),\s*Empty:\s*(\d+)", output)
    if m:
        return {"success": int(m.group(1)), "errors": int(m.group(2)), "empty": int(m.group(3))}
    return {"success": -1, "errors": -1, "empty": -1}


def parse_extract(output: str) -> dict:
    """Parse: Success: X, NotFound: Y, Skipped: Z, Error: W"""
    m = re.search(
        r"Success:\s*(\d+),\s*NotFound:\s*(\d+),\s*Skipped:\s*(\d+),\s*Error:\s*(\d+)",
        output,
    )
    if m:
        return {
            "extract_success": int(m.group(1)),
            "extract_not_found": int(m.group(2)),
            "extract_skipped": int(m.group(3)),
            "extract_error": int(m.group(4)),
        }
    return {"extract_success": -1, "extract_not_found": -1, "extract_skipped": -1, "extract_error": -1}


def parse_judge(output: str) -> dict:
    """Parse: filename.json: X/Y unsafe (Z.Z%)"""
    m = re.search(r"(\d+)/(\d+)\s+unsafe\s+\(([\d.]+)%\)", output)
    if m:
        return {"unsafe": int(m.group(1)), "total": int(m.group(2)), "unsafe_pct": float(m.group(3))}
    return {"unsafe": -1, "total": -1, "unsafe_pct": -1.0}


def main():
    setup_logging()
    print(f"Log file: {LOG_PATH}")
    total = len(MODELS) * len(DEFENSES) * len(TASKS)
    current = 0
    rows = []

    for model in MODELS:
        for defense in DEFENSES:
            for task, samples in TASKS:
                current += 1
                tag = f"[{current}/{total}] {model} | {defense} | {task} {samples}sample"
                print(f"\n{'=' * 70}")
                print(f"  EXPERIMENT {tag}")
                print(f"{'=' * 70}")

                row = {
                    "model": model,
                    "defense": defense,
                    "task": task,
                    "samples": samples,
                }

                # Step 1: run_safe_redirect (skip if result file already exists)
                rpath = result_file_path(model, task, samples, defense)
                if rpath.exists():
                    print(f"\n>>> Step 1/3: run_safe_redirect — SKIPPED (file exists: {rpath})")
                    # Count entries from existing file for CSV stats
                    try:
                        with open(rpath, "r", encoding="utf-8") as f:
                            entries = json.load(f)
                        row.update({"success": len(entries), "errors": 0, "empty": 0})
                    except Exception:
                        row.update({"success": -1, "errors": -1, "empty": -1})
                else:
                    print(f"\n>>> Step 1/3: run_safe_redirect")
                    out = run_cmd([
                        str(SCRIPT_DIR / "run_safe_redirect.py"),
                        "-m", model,
                        "-d", defense,
                        "-t", task,
                        "-s", str(samples),
                        "-c", str(CONCURRENT),
                        "-v", str(SAFE_REDIRECT_VERSION),
                    ])
                    row.update(parse_run_safe_redirect(out))
                    if not rpath.exists():
                        print(f"WARNING: result file not found: {rpath}")
                        row.update({"extract_success": -1, "extract_not_found": -1,
                                    "extract_skipped": -1, "extract_error": -1,
                                    "unsafe": -1, "total": -1, "unsafe_pct": -1.0})
                        rows.append(row)
                        continue

                # Step 2: extract
                print(f"\n>>> Step 2/3: extract")
                out = run_cmd([
                    str(SCRIPT_DIR / "extract.py"),
                    "-p", str(rpath),
                    "-c", str(CONCURRENT),
                ])
                row.update(parse_extract(out))

                # Step 3: judge
                print(f"\n>>> Step 3/3: judge")
                out = run_cmd([
                    str(SCRIPT_DIR / "judge.py"),
                    "-p", str(rpath),
                    "-c", str(CONCURRENT),
                ])
                row.update(parse_judge(out))

                rows.append(row)

    # Write CSV
    fieldnames = [
        "model", "defense", "task", "samples",
        "success", "errors", "empty",
        "extract_success", "extract_not_found", "extract_skipped", "extract_error",
        "unsafe", "total", "unsafe_pct",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS DONE — {total} experiments")
    print(f"  Results CSV: {CSV_PATH}")
    print(f"  Log file: {LOG_PATH}")
    print(f"{'=' * 70}")

    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()

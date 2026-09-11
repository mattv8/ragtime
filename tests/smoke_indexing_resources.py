"""Read-only acceptance monitor for an indexing-resource smoke run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import quantiles
from typing import Any

HEALTH_P95_SECONDS = 1.0
ADMIN_P95_SECONDS = 2.0


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[94]


def get_json(url: str, session: str | None = None) -> tuple[Any, float, int]:
    headers = {"Cookie": f"ragtime_session={session}"} if session else {}
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=5) as response:  # nosec B310: operator-supplied local URL
        payload = json.loads(response.read().decode("utf-8"))
        return payload, time.monotonic() - started, response.status


def docker_metrics(container: str = "ragtime-dev") -> dict[str, Any]:
    """Collect Docker-reported metrics only; never alters a container."""
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", container],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        stats = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {"state": json.loads(inspect.stdout), "stats": json.loads(stats.stdout)}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {"error": type(exc).__name__}


def _sample(base_url: str, session: str, job_ids: set[str]) -> dict[str, Any]:
    sample: dict[str, Any] = {"timestamp": time.time(), "docker": docker_metrics()}
    for key, path, authenticated in (
        ("health", "/health", False),
        ("jobs", "/indexes/jobs", True),
        ("resources", "/indexes/resources", True),
    ):
        try:
            payload, latency, status = get_json(f"{base_url}{path}", session if authenticated else None)
            sample[key] = {"status": status, "latency_seconds": latency, "payload": payload}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            sample[key] = {"error": type(exc).__name__}
    jobs = sample.get("jobs", {}).get("payload", [])
    if isinstance(jobs, list):
        sample["tracked_jobs"] = {
            job["id"]: {"status": job.get("status"), "phase": job.get("phase")} for job in jobs if isinstance(job, dict) and job.get("id") in job_ids
        }
    return sample


def run_monitor(args: argparse.Namespace) -> int:
    session = os.environ.get("RAGTIME_SMOKE_SESSION")
    if not session:
        print("RAGTIME_SMOKE_SESSION is required", file=sys.stderr)
        return 2
    base_url = args.base_url.rstrip("/")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    job_ids = set(args.job_id)
    with output.open("w", encoding="utf-8") as evidence:
        while time.monotonic() - started <= args.deadline_seconds:
            sample = _sample(base_url, session, job_ids)
            samples.append(sample)
            evidence.write(json.dumps(sample, separators=(",", ":")) + "\n")
            evidence.flush()
            tracked = sample.get("tracked_jobs", {})
            if len(tracked) == len(job_ids) and all(item.get("status") == "completed" for item in tracked.values()):
                break
            if any(item.get("status") in {"failed", "cancelled"} for item in tracked.values()):
                break
            time.sleep(args.interval)

    def latencies(key: str) -> list[float]:
        return [entry[key]["latency_seconds"] for entry in samples if "latency_seconds" in entry.get(key, {})]

    health_p95 = percentile95(latencies("health"))
    admin_p95 = percentile95(latencies("jobs") + latencies("resources"))
    final_jobs = samples[-1].get("tracked_jobs", {}) if samples else {}
    completed = len(final_jobs) == len(job_ids) and all(job.get("status") == "completed" for job in final_jobs.values())
    api_errors = any("error" in entry.get(key, {}) for entry in samples for key in ("health", "jobs", "resources"))
    passed = completed and not api_errors and (health_p95 or float("inf")) <= HEALTH_P95_SECONDS and (admin_p95 or float("inf")) <= ADMIN_P95_SECONDS
    summary = {
        "passed": passed,
        "completed": completed,
        "health_p95_seconds": health_p95,
        "admin_p95_seconds": admin_p95,
        "samples": len(samples),
        "jobs": final_jobs,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if passed else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--job-id", action="append", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.deadline_seconds <= 0:
        parser.error("--interval and --deadline-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run_monitor(parse_args()))

"""Deployment / release evidence collector.

Produces a dated, reproducible *deployment record* for the "release to
production" item of the testing-and-deployment appendix. It does not change
anything — it observes the running stack and captures:

- the pinned source revision (git commit, branch, date);
- the running containers and their health (docker compose / docker ps);
- the API image identity (id, created, size);
- the live ``GET /health`` response (status, service, version, environment).

Run it from the **host** (it needs the docker CLI and reaches the API on the
published port), after the stack is up:

    python -m harness.release_evidence --base-url http://localhost:8000

Full release sequence this record is meant to evidence (run separately):

    docker compose build
    docker compose up -d
    # wait for healthchecks, then:
    python -m harness.release_evidence

Output: ``harness/runs/<timestamp>_release/`` containing ``release_record.json``
and ``release_record.md`` (a snippet ready to paste into the appendix,
secrets excluded — this collector never reads .env or credentials).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    """Run a command, capturing stdout/stderr/rc without raising."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except FileNotFoundError:
        return {"cmd": " ".join(cmd), "error": f"command not found: {cmd[0]}"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "error": f"timed out after {timeout}s"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"cmd": " ".join(cmd), "error": f"{type(exc).__name__}: {exc}"}


def _git_revision() -> dict[str, Any]:
    return {
        "commit_short": _run(["git", "rev-parse", "--short", "HEAD"]).get("stdout"),
        "commit_full": _run(["git", "rev-parse", "HEAD"]).get("stdout"),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).get("stdout"),
        "committed_at": _run(["git", "log", "-1", "--format=%cI"]).get("stdout"),
        "describe": _run(["git", "describe", "--tags", "--always"]).get("stdout"),
        "dirty_tracked_files": bool(
            _run(["git", "status", "--porcelain", "--untracked-files=no"]).get("stdout")
        ),
    }


def _containers() -> dict[str, Any]:
    return {
        "compose_ps": _run(["docker", "compose", "ps"]).get("stdout"),
        "docker_ps": _run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}\t{{.Status}}\t{{.Ports}}",
            ]
        ).get("stdout"),
    }


def _api_image(image_ref: str) -> dict[str, Any]:
    out = _run(
        [
            "docker",
            "image",
            "inspect",
            image_ref,
            "--format",
            "{{.Id}}|{{.Created}}|{{.Size}}",
        ]
    )
    raw = out.get("stdout") or ""
    if out.get("returncode") == 0 and "|" in raw:
        image_id, created, size = raw.split("|", 2)
        return {
            "ref": image_ref,
            "id": image_id,
            "created": created,
            "size_bytes": int(size) if size.isdigit() else size,
        }
    return {"ref": image_ref, "error": out.get("stderr") or out.get("error") or "not found"}


def _health(base_url: str, timeout: int = 10) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local health check)
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body}
            return {"url": url, "http_status": resp.status, "body": parsed}
    except Exception as exc:
        return {"url": url, "error": f"{type(exc).__name__}: {exc}"}


def collect(base_url: str, image_ref: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "kind": "release_record",
        "collected_at": now.isoformat(),
        "source_revision": _git_revision(),
        "containers": _containers(),
        "api_image": _api_image(image_ref),
        "health": _health(base_url),
    }


def _to_markdown(record: dict[str, Any]) -> str:
    rev = record.get("source_revision", {})
    health = record.get("health", {})
    body = health.get("body", {}) if isinstance(health, dict) else {}
    lines = [
        "### Deployment record",
        "",
        f"- **Collected (UTC):** {record.get('collected_at')}",
        f"- **Source revision:** `{rev.get('commit_short')}` "
        f"(branch `{rev.get('branch')}`, committed {rev.get('committed_at')})",
        f"- **Working tree clean (tracked):** {not rev.get('dirty_tracked_files')}",
        f"- **Health:** `GET /health` -> HTTP {health.get('http_status', 'n/a')}",
    ]
    if isinstance(body, dict):
        lines += [
            f"  - status: `{body.get('status')}`",
            f"  - service: `{body.get('service')}`",
            f"  - version: `{body.get('version')}`",
            f"  - environment: `{body.get('environment')}`",
        ]
    img = record.get("api_image", {})
    if "id" in img:
        lines.append(f"- **API image:** `{img.get('id', '')[:19]}...` created {img.get('created')}")
    lines += [
        "",
        "```",
        record.get("containers", {}).get("docker_ps", "") or "(no container listing)",
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Collect a non-destructive deployment/release record.")
    p.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running API")
    p.add_argument(
        "--image-ref",
        default="workshops-ai-api",
        help="Docker image or container reference for the API",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: harness/runs/<ts>_release)",
    )
    args = p.parse_args()

    # Portable console output (Windows host consoles default to a legacy codepage).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    record = collect(args.base_url, args.image_ref)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else Path("harness/runs") / f"{ts}_release"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "release_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "release_record.md").write_text(_to_markdown(record), encoding="utf-8")

    print(_to_markdown(record))
    print(f"\n[release_evidence] written to {out_dir}", file=sys.stderr)
    health_ok = isinstance(record.get("health"), dict) and record["health"].get("http_status") == 200
    return 0 if health_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

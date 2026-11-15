"""
Agent mode (`run.py --agent`, M11 T11.21): the server takes work only from $OA_HOME/queue/<remote_job_id>.json files.

A job file (strict schema, stdlib validator; anything else is rejected and logged, never executed):

  {"remote_job_id": "...", "model": "...", "params": {...},
   "inputs": [{"url": "https://signed...", "sha256": "<64 hex>", "name": "start.png"}],
   "output": {"put_url": "https://signed...", "content_type": "video/mp4", "max_bytes": 123},
   "timeout_s": 600}                      (optional "video" / "project": a record carries one or the other)

Flow: claim (queue/running/) -> pull inputs (sha256 checked) -> render -> PUT the output -> result.json -> job.done.
An input `name` is a plain file name; its stem is the job key the renderer reads it under ("start.png" -> job["start"]).
Events go through oa.emit (stdout + logs/events.jsonl); signed URLs are never written to events or logs.
An expired URL (HTTP 403) writes need_urls/<id>.json and waits (bounded) for a refreshed job file with the same id.
A heartbeat event is emitted at a fixed interval for as long as the agent runs.
"""
from __future__ import annotations

# Single source of the agent version: gpu/image/Dockerfile bakes it into /opt/oneart/AGENT_VERSION (scripts/gpu-image.sh reads
# this line) and the health check (#638) compares it with the version the platform expects.
AGENT_VERSION = "1"

import hashlib
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import oa

HEARTBEAT_S = 30.0
URL_WAIT_S = 600.0
POLL_S = 2.0

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TOP = {"remote_job_id", "model", "params", "inputs", "output", "timeout_s", "video", "project"}


class JobFailed(Exception):
    def __init__(self, reason: str, fault: str = "ours"):
        super().__init__(reason)
        self.reason, self.fault = reason, fault


class UrlExpired(Exception):
    def __init__(self, what: str):
        super().__init__(what)
        self.what = what


def video_ref(rec: dict) -> str | None:
    """The video a record belongs to: the M1 `video`, else the legacy `project` (OQ-52)."""
    v = rec.get("video")
    return v if v is not None else rec.get("project")


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _http_url(x: object) -> bool:
    return isinstance(x, str) and re.match(r"^https?://[^\s/]+", x) is not None


def validate(job: object) -> list[str]:
    """Strict schema check; returns the list of problems (empty = valid)."""
    if not isinstance(job, dict):
        return ["job is not an object"]
    errs = [f"unknown key: {k}" for k in job if k not in _TOP]
    rid = job.get("remote_job_id")
    if not (isinstance(rid, str) and _ID.match(rid)):
        errs.append("remote_job_id invalid")
    if not (isinstance(job.get("model"), str) and job["model"]):
        errs.append("model invalid")
    if not isinstance(job.get("params"), dict):
        errs.append("params must be an object")
    ins = job.get("inputs")
    if not isinstance(ins, list):
        errs.append("inputs must be a list")
    else:
        seen = set()
        for i, it in enumerate(ins):
            if not isinstance(it, dict) or set(it) != {"url", "sha256", "name"}:
                errs.append(f"inputs[{i}] must have exactly url, sha256, name")
                continue
            if not _http_url(it["url"]):
                errs.append(f"inputs[{i}].url invalid")
            if not (isinstance(it["sha256"], str) and _SHA.match(it["sha256"])):
                errs.append(f"inputs[{i}].sha256 invalid")
            if not (isinstance(it["name"], str) and _NAME.match(it["name"]) and ".." not in it["name"]) or it["name"] in seen:
                errs.append(f"inputs[{i}].name invalid or repeated")
            else:
                seen.add(it["name"])
    out = job.get("output")
    if not isinstance(out, dict) or set(out) != {"put_url", "content_type", "max_bytes"}:
        errs.append("output must have exactly put_url, content_type, max_bytes")
    else:
        if not _http_url(out["put_url"]):
            errs.append("output.put_url invalid")
        if not (isinstance(out["content_type"], str) and re.match(r"^[\w.+-]+/[\w.+-]+$", out["content_type"])):
            errs.append("output.content_type invalid")
        if not (_is_int(out["max_bytes"]) and out["max_bytes"] > 0):
            errs.append("output.max_bytes invalid")
    t = job.get("timeout_s")
    if not (_is_int(t) and 1 <= t <= 86400):
        errs.append("timeout_s invalid")
    for k in ("video", "project"):
        if k in job and not (isinstance(job[k], str) and job[k]):
            errs.append(f"{k} invalid")
    return errs


class Agent:
    def __init__(self, fake_render: bool = False, heartbeat_s: float = HEARTBEAT_S, url_wait_s: float = URL_WAIT_S, poll_s: float = POLL_S, fake_render_ms: float = 0.0):
        self.fake, self.heartbeat_s, self.url_wait_s, self.poll_s = fake_render, heartbeat_s, url_wait_s, poll_s
        self.fake_render_ms = fake_render_ms  # test flag: how long the fake render sleeps
        self.queue = oa.OA_HOME / "queue"
        self.need_urls = oa.OA_HOME / "need_urls"
        for d in ("running", "done", "failed", "rejected"):
            (self.queue / d).mkdir(parents=True, exist_ok=True)
        self.need_urls.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()

    # ---------------------------------------------------------------- heartbeat
    def _beat(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            oa.emit({"ev": "heartbeat"})

    # ---------------------------------------------------------------- queue
    def _reject(self, f: Path, errors: list[str]) -> None:
        oa.emit({"ev": "job.rejected", "file": f.name, "errors": errors[:20]})
        dest = self.queue / "rejected" / f.name
        try:
            f.replace(dest)
        except FileNotFoundError:
            pass

    def take(self) -> tuple[dict, Path] | None:
        """Oldest valid job file, claimed by moving it into queue/running/. Invalid files are rejected and logged."""
        for f in sorted(self.queue.glob("*.json"), key=lambda p: p.stat().st_mtime):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                self._reject(f, [f"unreadable: {type(e).__name__}"])
                continue
            errs = validate(job)
            if not errs and f.name != f"{job['remote_job_id']}.json":
                errs = ["file name does not match remote_job_id"]
            if errs:
                self._reject(f, errs)
                continue
            dest = self.queue / "running" / f.name
            try:
                f.replace(dest)
            except FileNotFoundError:
                continue
            return job, dest
        return None

    # ---------------------------------------------------------------- transfers
    def _get(self, item: dict, dest: Path) -> str:
        h = hashlib.sha256()
        try:
            with urllib.request.urlopen(item["url"], timeout=120) as r, open(dest, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    h.update(chunk)
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise UrlExpired(f"input {item['name']}") from None
            raise JobFailed(f"input {item['name']}: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise JobFailed(f"input {item['name']}: {type(e).__name__}") from None
        return h.hexdigest()

    def _put(self, job: dict, data: bytes) -> None:
        out = job["output"]
        req = urllib.request.Request(out["put_url"], data=data, method="PUT", headers={"Content-Type": out["content_type"]})
        try:
            with urllib.request.urlopen(req, timeout=600):
                pass
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise UrlExpired("output") from None
            raise JobFailed(f"output PUT: HTTP {e.code}") from None
        except (urllib.error.URLError, OSError) as e:
            raise JobFailed(f"output PUT: {type(e).__name__}") from None

    def _await_refresh(self, job: dict, what: str) -> dict:
        """An URL expired: ask for fresh ones and wait (bounded) for a new job file with the same id."""
        rid = job["remote_job_id"]
        (self.need_urls / f"{rid}.json").write_text(json.dumps({"remote_job_id": rid, "expired": what}))
        oa.emit({"ev": "job.need_urls", "remote_job_id": rid, "expired": what})
        deadline = time.time() + self.url_wait_s
        fresh = self.queue / f"{rid}.json"
        while time.time() < deadline:
            if fresh.exists():
                try:
                    new = json.loads(fresh.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    new = None
                errs = validate(new) if new is not None else ["unreadable"]
                if errs or new["remote_job_id"] != rid:
                    self._reject(fresh, errs or ["remote_job_id differs"])
                else:
                    fresh.replace(self.queue / "running" / fresh.name)
                    (self.need_urls / f"{rid}.json").unlink(missing_ok=True)
                    return new
            time.sleep(self.poll_s)
        raise JobFailed(f"urls not refreshed within {int(self.url_wait_s)} s")

    # ---------------------------------------------------------------- render
    def _render(self, job: dict, work: Path, rel_inputs: dict[str, str]) -> tuple[Path, int]:
        """Returns (output file, peak_vram_mb)."""
        dest = work / "output.bin"
        if self.fake:
            if self.fake_render_ms > 0:
                time.sleep(self.fake_render_ms / 1000)
            seed = hashlib.sha256(json.dumps([job["model"], job["params"]], sort_keys=True).encode()).hexdigest()
            dest.write_bytes(f"fake-render:{seed}\n".encode())
            return dest, 0
        import run  # the existing render pipeline (ComfyUI models); lazy: it needs the full server image
        from models import MODELS
        spec = MODELS.get(job["model"])
        if not spec or spec.get("engine") != "comfy":
            raise JobFailed(f"model {job['model']} is not supported in agent mode")
        rid = job["remote_job_id"]
        rj = {**job["params"], "id": rid, "model": job["model"], **rel_inputs}
        rj.setdefault("variant", "agent")
        oi = run.comfy_ready([0], spec.get("needs_node"))
        attention = (oa.OA_HOME / "attention.txt").read_text().strip() if (oa.OA_HOME / "attention.txt").exists() else "sdpa"
        with oa.VramSampler(gpu=0) as vram:
            path, _meta = run.render_comfy(rj, oi, attention, 0)
        return path, vram.peak_mb

    def _render_timed(self, job: dict, work: Path, rel_inputs: dict[str, str]) -> tuple[Path, int]:
        box: dict = {}

        def go() -> None:
            try:
                box["r"] = self._render(job, work, rel_inputs)
            except BaseException as e:  # noqa: BLE001 — handed to the caller
                box["e"] = e

        t = threading.Thread(target=go, daemon=True)
        t.start()
        t.join(job["timeout_s"])
        if t.is_alive():
            if not self.fake:
                try:
                    oa.comfy_post("/interrupt", {})
                except Exception:  # noqa: BLE001
                    pass
            raise JobFailed(f"timeout after {job['timeout_s']} s")
        if "e" in box:
            e = box["e"]
            raise e if isinstance(e, JobFailed) else JobFailed(f"{type(e).__name__}: {e}"[:500])
        return box["r"]

    # ---------------------------------------------------------------- one job
    def _pull(self, job: dict, work: Path) -> dict[str, str]:
        rel: dict[str, str] = {}
        for item in job["inputs"]:
            dest = work / "inputs" / item["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if self._get(item, dest) != item["sha256"]:
                dest.unlink(missing_ok=True)
                raise JobFailed(f"sha256 mismatch for input {item['name']}")
            rel[Path(item["name"]).stem] = str(dest.relative_to(oa.INPUTS)) if dest.is_relative_to(oa.INPUTS) else str(dest)
        return rel

    def handle(self, job: dict, claimed: Path) -> bool:
        rid = job["remote_job_id"]
        ref = video_ref(job)
        base = {"remote_job_id": rid, "model": job["model"], **({"video": ref} if ref else {})}
        oa.emit({"ev": "job.started", **base})
        work = oa.INPUTS / "agent" / rid
        t0 = time.time()
        try:
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            while True:  # a refreshed job file restarts the pull/PUT with the new URLs (same render result)
                try:
                    rel = self._pull(job, work)
                    break
                except UrlExpired as e:
                    job = self._await_refresh(job, e.what)
            t_render = time.time()
            out, vram = self._render_timed(job, work, rel)
            exec_ms = int((time.time() - t_render) * 1000)
            data = out.read_bytes()
            if len(data) > job["output"]["max_bytes"]:
                raise JobFailed(f"output {len(data)} bytes exceeds max_bytes")
            digest = hashlib.sha256(data).hexdigest()
            while True:
                try:
                    self._put(job, data)
                    break
                except UrlExpired as e:
                    job = self._await_refresh(job, e.what)
            result = {**base, "sha256": digest, "bytes": len(data), "exec_ms": exec_ms, "peak_vram_mb": vram}
            res_dir = oa.OUT / "agent" / rid
            res_dir.mkdir(parents=True, exist_ok=True)
            (res_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
            oa.emit({"ev": "job.done", **result, "ms": int((time.time() - t0) * 1000)})
            ok = True
        except JobFailed as e:
            oa.emit({"ev": "job.failed", **base, "fault": e.fault, "error": e.reason[:500], "ms": int((time.time() - t0) * 1000)})
            ok = False
        except Exception as e:  # noqa: BLE001 — one job never stops the agent
            oa.emit({"ev": "job.failed", **base, "fault": "ours", "error": f"{type(e).__name__}: {e}"[:500], "ms": int((time.time() - t0) * 1000)})
            ok = False
        finally:
            shutil.rmtree(work, ignore_errors=True)
            (self.need_urls / f"{rid}.json").unlink(missing_ok=True)
        (self.queue / ("done" if ok else "failed")).mkdir(parents=True, exist_ok=True)
        try:
            claimed.replace(self.queue / ("done" if ok else "failed") / claimed.name)
        except FileNotFoundError:
            pass
        return ok

    # ---------------------------------------------------------------- loop
    def run(self, exit_when_idle: bool = False) -> int:
        beat = threading.Thread(target=self._beat, daemon=True)
        beat.start()
        n = 0
        try:
            while True:
                got = self.take()
                if got is None:
                    if exit_when_idle:
                        return n
                    time.sleep(self.poll_s)
                    continue
                self.handle(*got)
                n += 1
        finally:
            self._stop.set()


def main(argv: list[str]) -> int:
    def opt(name: str, default: float) -> float:
        return next((float(a.split("=", 1)[1]) for a in argv if a.startswith(f"--{name}=")), default)

    oa.ensure_dirs()
    agent = Agent(fake_render="--fake-render" in argv, heartbeat_s=opt("heartbeat-s", HEARTBEAT_S),
                  url_wait_s=opt("url-wait-s", URL_WAIT_S), poll_s=opt("poll-s", POLL_S),
                  fake_render_ms=opt("fake-render-ms", 0.0))
    agent.run(exit_when_idle="--exit-when-idle" in argv)
    return 0

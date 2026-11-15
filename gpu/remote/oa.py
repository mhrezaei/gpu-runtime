"""
Shared helpers for the OneArt GPU-server scripts.

Event protocol: every timed step prints one line
    @@ {"ev": "start"|"end", "id": "...", "phase": "...", "label": "...", ...}
on stdout AND appends it to $OA_HOME/logs/events.jsonl, so the API can build its timeline live over SSH
and can recover it later if the SSH connection drops.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

OA_HOME = Path(os.environ.get("OA_HOME", "/workspace/oneart"))
INPUTS = OA_HOME / "inputs"
OUT = OA_HOME / "out"
LOGS = OA_HOME / "logs"
MODELS = Path(os.environ.get("OA_MODELS", "/workspace/models"))
REPOS = OA_HOME / "repos"
# Versioned runtime image (gpu/image/Dockerfile): software lives under IMAGE_HOME, marked by IMAGE_VERSION.
IMAGE_HOME = Path(os.environ.get("OA_IMAGE", "/opt/oneart"))
IMAGE_VERSION_FILE = IMAGE_HOME / "IMAGE_VERSION"
COMFY_DIR = Path(os.environ.get("COMFY_DIR", str(IMAGE_HOME / "ComfyUI" if IMAGE_VERSION_FILE.exists() else "/workspace/ComfyUI")))
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_BASE_PORT = 8188
_emit_lock = threading.Lock()


def gpu_ids() -> list[int]:
    """GPUs to use: $OA_GPUS ("0,1,2" or a count) or every GPU nvidia-smi lists."""
    want = os.environ.get("OA_GPUS", "").strip()
    try:
        n = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout.splitlines() if l.startswith("GPU")])
    except Exception:
        n = 1
    if want:
        ids = [int(x) for x in re.split(r"[ ,]+", want) if x] if "," in want else list(range(min(int(want), max(n, 1))))
        return ids or [0]
    return list(range(max(n, 1)))


def comfy_url(gpu: int | None = None) -> str:
    """One ComfyUI instance per GPU: GPU i listens on 8188+i (COMFY_URL overrides GPU 0, e.g. for local tests)."""
    return COMFY_URL if not gpu else f"http://127.0.0.1:{COMFY_BASE_PORT + gpu}"



def ensure_dirs() -> None:
    for d in (INPUTS, OUT, LOGS, MODELS, REPOS):
        d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"


def emit(ev: dict) -> None:
    ev = {"t": now_iso(), **ev}
    line = json.dumps(ev, ensure_ascii=False)
    with _emit_lock:
        LOGS.mkdir(parents=True, exist_ok=True)
        print("@@ " + line, flush=True)
        with open(LOGS / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")


class VramSampler:
    """Samples nvidia-smi memory.used once a second (one GPU, or the max over all); .peak_mb after stop()."""

    def __init__(self, interval: float = 1.0, gpu: int | None = None):
        self.interval = interval
        self.gpu = gpu
        self.peak_mb = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", *(["-i", str(self.gpu)] if self.gpu is not None else []), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                self.peak_mb = max([self.peak_mb, *[int(x) for x in out.split() if x.strip().isdigit()]])
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=3)


@contextlib.contextmanager
def phase(phase_name: str, label: str, gpu: int | None = None, **extra):
    """Timed step (optionally on one GPU). Yields a dict; anything put in it is reported as meta on the end event."""
    pid = uuid.uuid4().hex[:10]
    meta: dict = {} if gpu is None else {"gpu": gpu}
    emit({"ev": "start", "id": pid, "phase": phase_name, "label": label, **({"gpu": gpu} if gpu is not None else {}), **extra})
    t0 = time.time()
    with VramSampler(gpu=gpu) as vram:
        try:
            yield meta
        except BaseException as e:  # noqa: BLE001 — report every failure, then re-raise
            emit({"ev": "end", "id": pid, "phase": phase_name, "status": "failed", "ms": int((time.time() - t0) * 1000),
                  "error": f"{type(e).__name__}: {e}"[:2000], "meta": {**meta, "peak_vram_mb": vram.peak_mb}})
            raise
    emit({"ev": "end", "id": pid, "phase": phase_name, "status": "done", "ms": int((time.time() - t0) * 1000),
          "meta": {**meta, "peak_vram_mb": vram.peak_mb}})


def sh(cmd: str, cwd: str | Path | None = None, env: dict | None = None, log: str | None = None) -> None:  # noqa: D401
    """Run a shell command, streaming output into a log file; raise on failure with the log tail."""
    log_path = LOGS / (log or "sh.log")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {cmd}\n")
        lf.flush()
        p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, env={**os.environ, **(env or {})}, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"exit {p.returncode}: {cmd}\n{tail}")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def video_seconds(path: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return round(float(out), 3)
    except Exception:
        return None


# ------------------------------------------------------------------ ComfyUI client

def comfy_get(path: str, base: str | None = None):
    with urllib.request.urlopen((base or COMFY_URL) + path, timeout=30) as r:
        return json.loads(r.read())


def comfy_post(path: str, body: dict, base: str | None = None):
    req = urllib.request.Request((base or COMFY_URL) + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:  # node_errors come back as 400 with a JSON body
        raise RuntimeError(f"ComfyUI {e.code}: {e.read().decode()[:3000]}") from None


def comfy_alive(base: str | None = None) -> bool:
    try:
        comfy_get("/system_stats", base)
        return True
    except Exception:
        return False


def comfy_run(workflow: dict, timeout_s: int = 3600, base: str | None = None) -> dict:
    """Queue an API-format workflow on one instance, wait for it, return {prompt_id, outputs, exec_ms}."""
    client_id = uuid.uuid4().hex
    r = comfy_post("/prompt", {"prompt": workflow, "client_id": client_id}, base)
    pid = r["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hist = comfy_get(f"/history/{pid}", base)
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI execution error: {json.dumps(msgs)[:3000]}")
            if st.get("completed", False) or h.get("outputs"):
                return {"prompt_id": pid, "outputs": h.get("outputs", {}), "exec_ms": int((time.time() - t0) * 1000)}
        time.sleep(1)
    raise TimeoutError(f"ComfyUI prompt {pid} did not finish in {timeout_s}s")


def comfy_output_files(outputs: dict) -> list[Path]:
    """Every file a run saved (videos/images/gifs), as absolute paths under ComfyUI/output."""
    files: list[Path] = []
    for node in outputs.values():
        for key in ("videos", "images", "gifs", "audio"):
            for item in node.get(key, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    files.append(COMFY_DIR / item.get("type", "output") / item.get("subfolder", "") / item["filename"])
    return files


CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def map_widgets(items: list, values: list, prefix: str = "") -> dict:
    """
    Walk a node's widget specs (in input_order) against its UI widgets_values.
    V3 dynamic combos store [key, <selected option's sub-widgets>...]; their inputs are named "<combo>.<sub>".
    A seed-like INT may be followed by a hidden control_after_generate value ("fixed"/"randomize"…), skipped here.
    """
    out: dict = {}
    for name, spec in items:
        if not values:
            break
        typ = spec[0]
        if typ == "COMFY_DYNAMICCOMBO_V3":
            key = values.pop(0)
            out[prefix + name] = key
            opt = next((o for o in spec[1].get("options", []) if o.get("key") == key), None)
            if opt:
                sub = opt.get("inputs", {})
                sub_items = list(sub.get("required", {}).items()) + list(sub.get("optional", {}).items())
                out.update(map_widgets(sub_items, values, f"{prefix}{name}."))
        elif isinstance(typ, list) or typ in WIDGET_TYPES or (isinstance(typ, str) and any(t in WIDGET_TYPES for t in typ.split(","))):
            # multi-type sockets ("FLOAT,INT", e.g. LTXVEmptyLatentAudio.frame_rate) are widgets too: skipping one
            # shifted every later value (frame rate 25 landed in batch_size and broke every LTX FLF job, 2026-10-21)
            out[prefix + name] = values.pop(0)
            if typ == "INT" and values and isinstance(values[0], str) and values[0] in CONTROL_VALUES:
                values.pop(0)
    return out


VIRTUAL = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode", "SetNode", "GetNode"}


def ui_to_api(ui: dict, object_info: dict) -> dict:
    """
    Convert a ComfyUI UI-format workflow (official templates, node-pack examples) to API format.

    Handles: nested subgraphs (flattened; inner ids become "<instance>:<inner>"), KJNodes Set/Get pairs,
    Reroute, frontend PrimitiveNode, muted (mode 2) and bypassed (mode 4) nodes, and widget values mapped
    to input names via /object_info (including the hidden control_after_generate widget after seeds).
    """
    subdefs = {sd["id"]: sd for sd in ui.get("definitions", {}).get("subgraphs", [])}
    N: dict[str, dict] = {}            # node id -> node (with "_sd" for subgraph instances)
    L: dict[str, tuple[str, int]] = {}  # link id -> (origin node id, origin slot)

    def add_scope(nodes, links, prefix):
        for l in links:
            lid, o, os_ = (l[0], l[1], l[2]) if isinstance(l, list) else (l["id"], l["origin_id"], l["origin_slot"])
            L[f"{prefix}{lid}"] = (f"{prefix}{o}", os_)
        for n in nodes:
            nid = f"{prefix}{n['id']}"
            N[nid] = {**n, "_id": nid, "_prefix": prefix}
            if n["type"] in subdefs:
                sd = subdefs[n["type"]]
                N[nid]["_sd"] = sd
                add_scope(sd["nodes"], sd.get("links", []), f"{nid}:")

    add_scope(ui["nodes"], ui.get("links", []), "")
    setters = {}
    for nid, n in N.items():
        if n["type"] == "SetNode" and n.get("widgets_values"):
            setters[n["widgets_values"][0]] = n

    def input_link(n, name=None, index=None):
        for k, i in enumerate(n.get("inputs", [])):
            if (name is not None and i["name"] == name) or (index is not None and k == index):
                return None if i.get("link") is None else f"{n['_prefix']}{i['link']}"
        return None

    def resolve(origin: str, slot: int, depth=0):
        """-> ("link", [id, slot]) | ("value", v) | None (unconnected: keep the target's own widget value)"""
        if depth > 50:
            raise RecursionError("link cycle")
        if origin.endswith(":-10"):  # subgraph input -> the instance's external connection
            inst = N[origin[: -len(":-10")]]
            name = inst["_sd"]["inputs"][slot]["name"]
            lk = input_link(inst, name=name)
            return resolve(*L[lk], depth + 1) if lk else None
        n = N[origin]
        if "_sd" in n:  # subgraph instance output -> inner origin
            lid = n["_sd"]["outputs"][slot]["linkIds"][0]
            return resolve(*L[f"{origin}:{lid}"], depth + 1)
        t = n["type"]
        if t in ("Reroute", "SetNode"):  # SetNode also passes its input straight through
            lk = input_link(n, index=0)
            return resolve(*L[lk], depth + 1) if lk else None
        if t == "GetNode":
            st = setters.get((n.get("widgets_values") or [None])[0])
            lk = st and input_link(st, index=0)
            if not lk:
                raise KeyError(f"GetNode without SetNode: {n.get('widgets_values')}")
            return resolve(*L[lk], depth + 1)
        if t == "PrimitiveNode":
            return ("value", n["widgets_values"][0])
        if n.get("mode", 0) == 4:  # bypassed: pass through the first input of the same type
            out_type = n["outputs"][slot]["type"]
            for i in n.get("inputs", []):
                if i.get("type") == out_type and i.get("link") is not None:
                    return resolve(*L[f"{n['_prefix']}{i['link']}"], depth + 1)
            return None
        return ("link", [origin, slot])

    api: dict = {}
    for nid, n in N.items():
        if n["type"] in VIRTUAL or "_sd" in n or n.get("mode", 0) in (2, 4):
            continue
        info = object_info.get(n["type"])
        if info is None:
            raise KeyError(f"node type not installed: {n['type']}")
        req = info["input"].get("required", {})
        opt = info["input"].get("optional", {})
        order = info.get("input_order") or {"required": list(req), "optional": list(opt)}
        items = [(k, req[k]) for k in order.get("required", []) if k in req] + [(k, opt[k]) for k in order.get("optional", []) if k in opt]
        inputs: dict = {}
        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            inputs.update({k: v for k, v in wv.items() if k in req or k in opt})
        elif wv:
            inputs.update(map_widgets(items, list(wv)))
        for i in n.get("inputs", []):
            if i.get("link") is None:
                continue
            r = resolve(*L[f"{n['_prefix']}{i['link']}"])
            if r is not None:
                inputs[i["name"]] = r[1]
        api[nid] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title") or n["type"]}}

    return prune(api)


OUTPUT_NODES = {"SaveVideo", "VHS_VideoCombine", "SaveImage", "SaveAudio", "SaveAnimatedWEBP"}


def prune(api: dict, keep: set[str] | None = None) -> dict:
    """Keep only output nodes (or the given ids) and everything they depend on; drops PreviewAny/debug branches."""
    roots = keep if keep is not None else {k for k, v in api.items() if v["class_type"] in OUTPUT_NODES}
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        k = stack.pop()
        if k in seen or k not in api:
            continue
        seen.add(k)
        for v in api[k]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return {k: v for k, v in api.items() if k in seen}


if __name__ == "__main__":
    # `python3 oa.py --selfcheck [--no-gpu]`: verify the runtime image without touching a GPU (the check never calls CUDA).
    if "--selfcheck" in sys.argv[1:]:
        import bootstrap  # noqa: E402 — lazy: bootstrap imports this module

        sys.exit(bootstrap.selfcheck())
    sys.exit("usage: python3 oa.py --selfcheck [--no-gpu]")

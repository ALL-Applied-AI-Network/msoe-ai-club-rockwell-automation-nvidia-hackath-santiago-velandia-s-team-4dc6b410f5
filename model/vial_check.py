"""Check every vial in a fill-camera video (or a live camera) against its recipe.

For each vial the program finds when the vial is under the nozzle, takes a picture of the finished vial, lets
the trained model estimate how much orange and blue dye is in it, and compares that with what a correct vial of
the recipe should look like. If the difference is bigger than the alert threshold, the vial gets an ALERT
(possible carryover); otherwise OK.

Two ways to run it, from version_1/ (needs .venv: numpy, pandas, opencv-python, torch, torchvision):

    # 1. Operator dashboard (opens at http://127.0.0.1:8000). Its setup screen asks for the video (upload a
    #    file, connect a camera by number or stream address, or connect the digital twin), the recipe queue CSV
    #    and which tank holds which color; then it shows OK / ALERT as each vial finishes.
    .venv/bin/python model/vial_check.py --live          # --speed 10 plays uploaded videos 10x faster
    # -> model/results/live_<name>_<date-time>/vials.csv (saved after every vial), snapshots/, recipe_queue.csv

    # 2. A whole recorded video at once (no dashboard), with a summary and a picture sheet of all vials:
    .venv/bin/python model/vial_check.py --video <video.mp4> --recipes <recipe queue CSV> [--out-dir <folder>]
    # -> vials.csv, metrics.json, contact_sheet.jpg, snapshots/  (default folder: model/results/check_<video name>)
    # The video 5 validation: --video ../csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4
    #   --recipes ../csi-vial-fill-dataset/vial_fill/labels/video5_vial_recipes.csv --out-dir model/results/video5_validation

Recipe queue CSV: one row per vial, in order, with CSI's columns `vial, recipe_id, t100_ml, t200_ml, t400_ml`
(mL from each tank; blank = tank not used). If it also has CSI's `start_s, end_s` columns (when each vial was
filled, in video seconds), vials are matched by time and the run is scored. Without them, vials are matched in
order (in mode 2 the number of vials found must equal the number of rows).

How a vial is found (settings chosen on real videos 1-4; nothing is tuned on the video being checked):
  - The video is sampled 5 times per second.
  - A big movement of the carrier ring (CARRIER_REGION) means vials are changing; the vial's time under the
    nozzle is between two such movements.
  - Movement inside the vial (VIAL_REGION) is a pour; back-to-back pours with no pause count as one.
  - The picture is taken 1 second before the carrier moves the vial away (the finished vial).
  - A gap with no pours and no colored liquid (an empty carrier passing) is skipped.
Scoring (only with start_s/end_s): CSI's labels don't say which vials have carryover, so the first vial after a
recipe change is counted as carryover and the others as clean (CSI's run plan has 3 vials per recipe). The first
vial in the queue is not scored (nothing came before it).
"""
import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))
from build_table import NOMINAL_RESIDUE_ML, clean_vial_shares, expected_shares        # noqa: E402
from train_model import (CENTER, LOAD, PANEL_ROI, RECIPE_COLS, RECIPE_SCALE, build_net,  # noqa: E402
                       network_input, scores)

CHECKPOINT = MODEL_DIR / "results" / "twin_test" / "model.pt"   # from train_model.py (not in git)
FPS_SAMPLE = 5
VIAL_REGION = (295, 280, 326, 440)        # x0, y0, x1, y1: the vial below the carrier
CARRIER_REGION = (240, 200, 380, 270)     # the carrier ring; moves when vials change
CARRIER_MOTION = 20                       # mean abs change per sample that means "carrier moving"
POUR_MOTION = 1.2                         # smoothed vial motion that means "pouring"
POUR_GAP_S, POUR_MIN_S = 0.6, 0.8
FINAL_BEFORE_S = 1.0
LIQUID_SAT = 45                           # mean saturation of the vial strip that means "has colored liquid"


# ----------------------------------------------------------------------------------------------
# Finding vials
# ----------------------------------------------------------------------------------------------
def signals(video):
    """Per sample: time, vial motion, carrier motion, vial saturation."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / FPS_SAMPLE))
    rows, prev, i = [], None, 0
    while cap.grab():
        if i % step == 0:
            _, f = cap.retrieve()
            x0, y0, x1, y1 = VIAL_REGION
            cx0, cy0, cx1, cy1 = CARRIER_REGION
            vial, carrier = f[y0:y1, x0:x1].astype(np.float32), f[cy0:cy1, cx0:cx1].astype(np.float32)
            sat = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 1].mean()
            mv = 0.0 if prev is None else float(np.abs(vial - prev[0]).mean())
            mc = 0.0 if prev is None else float(np.abs(carrier - prev[1]).mean())
            rows.append((i / fps, mv, mc, sat))
            prev = (vial, carrier)
        i += 1
    return np.array(rows)


def find_vials(sig):
    t, mv, mc, sat = sig.T
    dt = t[1] - t[0]
    moves = []                                                   # carrier movements (merged spikes)
    for s in t[mc > CARRIER_MOTION]:
        if moves and s - moves[-1][1] <= 3:
            moves[-1][1] = s
        else:
            moves.append([s, s])
    edges = [[t[0], t[0]]] + moves + [[t[-1], t[-1]]]            # the video start and end bound vials too
    smooth = smoothed(mv)
    vials = [vial_window(t, smooth, sat, a + 0.5, b - 0.5, dt) for (_, a), (b, _) in zip(edges[:-1], edges[1:])]
    return [v for v in vials if v]


def smoothed(mv):
    return pd.Series(mv).rolling(3, center=True, min_periods=1).median().values


def count_pours(t, smooth, c0, c1, dt):
    m = (t >= c0) & (t <= c1)
    pours = []
    for tk, on in zip(t[m], smooth[m] > POUR_MOTION):
        if on:
            if pours and tk - pours[-1][1] <= POUR_GAP_S + dt:
                pours[-1][1] = tk
            else:
                pours.append([tk, tk])
    return len([p for p in pours if p[1] - p[0] >= POUR_MIN_S])


def vial_window(t, smooth, sat, c0, c1, dt):
    """The vial between two carrier moves (c0..c1), or None if it is too short or empty."""
    if c1 - c0 < 3 or not ((t >= c0) & (t <= c1)).any():
        return None
    pours = count_pours(t, smooth, c0, c1, dt)
    final = c1 - FINAL_BEFORE_S
    if not pours and (c1 - c0 < 10 or sat[np.argmin(np.abs(t - final))] < LIQUID_SAT):
        return None                                              # empty, or cut off by the end of the video
    return {"start_s": round(c0, 1), "end_s": round(c1, 1), "pours_seen": pours, "final_s": round(final, 2)}


def frame_at(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, f = cap.read()
    return f if ok else None


def match_queue(queue, detected):
    """Recipe-queue row index -> detected vial. By time overlap if the queue has start_s/end_s, else in order."""
    if {"start_s", "end_s"} <= set(queue.columns):
        match = {}
        for i, q in enumerate(queue.itertuples()):
            best, j = max(((min(d["end_s"], q.end_s) - max(d["start_s"], q.start_s), j)
                           for j, d in enumerate(detected)), default=(0, None))
            if best > 0:
                match[i] = detected[j]
        return match
    if len(detected) != len(queue):
        raise SystemExit(f"STOP: found {len(detected)} vials but the recipe queue has {len(queue)}; "
                         "add start_s/end_s to the queue or check the video")
    return dict(enumerate(detected))


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def recipe_inputs(queue):
    """The recipe inputs for the final pour of each vial, computed exactly as for the twin table."""
    rows = []
    for q in queue.itertuples():
        r = {"t200_ml": q.t200_ml, "t400_ml": q.t400_ml, "t100_ml": q.t100_ml}
        r["total_ml"] = sum(r.values())
        r["stages_total"] = r["stage"] = sum(v > 0 for v in (q.t200_ml, q.t400_ml, q.t100_ml))
        exp_o, exp_b = expected_shares(r)
        cln_o, cln_b = clean_vial_shares(r, NOMINAL_RESIDUE_ML)
        rows.append({**r, "expected_orange": exp_o, "expected_blue": exp_b,
                     "expected_clean_orange": cln_o, "expected_clean_blue": cln_b})
    return pd.DataFrame(rows)


def load_model():
    import torch
    import torchvision
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net, _ = build_net(torch, torchvision, pretrained=False, n_recipe=len(ckpt["recipe_cols"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt["threshold"]


def estimate(net, frames, rec):
    import torch
    x0, y0, x1, y1 = LOAD
    px0, py0, px1, py1 = PANEL_ROI
    rgb = [f[..., ::-1] for f in frames]
    crops = np.stack([f[y0:y1, x0:x1] for f in rgb])
    panels = np.stack([np.median(f[py0:py1, px0:px1].reshape(-1, 3), axis=0) for f in rgb]).astype(np.float32)
    recipe_in = (rec[RECIPE_COLS].astype(float).values / np.array(RECIPE_SCALE, float)).astype(np.float32)
    with torch.no_grad():
        est = net(torch.from_numpy(network_input(crops, panels, [CENTER] * len(frames))),
                  torch.from_numpy(recipe_in)).numpy()
    return est


# ----------------------------------------------------------------------------------------------
def contact_sheet(frames, df, path):
    tiles = []
    for f, r in zip(frames, df.itertuples()):
        tile = f[140:475, 220:400].copy()
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                      (0, 0, 255) if r.flag else (90, 90, 90), 4 if r.flag else 1)
        lines = [f"vial {r.vial}  recipe {r.recipe_id}"]
        if "truth_carryover" in df.columns:
            lines.append({1: "CARRYOVER", 0: "clean", -1: "(not scored)"}[r.truth_carryover])
        bottom = [f"est {100 * r.est_orange:.0f}%o {100 * r.est_blue:.0f}%b",
                  f"exp {100 * r.expected_orange:.0f}%o {100 * r.expected_blue:.0f}%b",
                  f"gap {100 * r.gap:.0f} pts" + ("  FLAG" if r.flag else "")]
        for y, t in [(16 + 16 * j, t) for j, t in enumerate(lines)] + [(268 + 16 * j, t) for j, t in enumerate(bottom)]:
            cv2.putText(tile, t, (4, y), 0, 0.42, (0, 0, 0), 3)
            cv2.putText(tile, t, (4, y), 0, 0.42, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))

# ----------------------------------------------------------------------------------------------
# Live mode: the same vial finder and model, run as frames arrive, with a dashboard in the browser.
# The setup screen takes the video (upload or camera), the recipe queue CSV and the tank assignment.
# ----------------------------------------------------------------------------------------------
RULES = MODEL_DIR.parent / "twin_tools" / "recipe_rules.json"
TANK_IDS = ["T100", "T200", "T300", "T400"]


def default_tanks():
    """Color -> tank, from the tank assignment in twin_tools/recipe_rules.json."""
    tanks = json.loads(RULES.read_text())["tanks"]
    return {c: next(t for t, v in tanks.items() if v["color"] == c) for c in ["orange", "blue", "clear"]}


def read_queue(csv_file, tanks):
    """Recipe queue CSV (CSI's columns) -> the queue the model uses: t200_ml = orange, t400_ml = blue,
    t100_ml = clear (the twin's tank layout), whatever tanks hold those colors on this line."""
    raw = pd.read_csv(csv_file)
    if len(set(tanks.values())) != 3:
        raise ValueError("each color needs its own tank")
    if "recipe_id" not in raw.columns:
        raise ValueError("the recipe CSV needs a recipe_id column")
    for tank in TANK_IDS:
        col = f"{tank.lower()}_ml"
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
            if tank not in tanks.values() and (raw[col] > 0).any():
                raise ValueError(f"the recipes use {tank}, but no color is assigned to it")
    queue = pd.DataFrame({"vial": raw["vial"] if "vial" in raw.columns else range(1, len(raw) + 1),
                          "recipe_id": raw["recipe_id"].astype(str)})
    for color, internal in [("orange", "t200_ml"), ("blue", "t400_ml"), ("clear", "t100_ml")]:
        col = f"{tanks[color].lower()}_ml"
        queue[internal] = raw[col] if col in raw.columns else 0.0
    if ((queue.t200_ml + queue.t400_ml + queue.t100_ml) <= 0).any():
        raise ValueError("every vial needs at least one tank with mL > 0")
    for c in ["start_s", "end_s"]:
        if c in raw.columns:
            queue[c] = pd.to_numeric(raw[c], errors="coerce")
    if {"start_s", "end_s"} <= set(queue.columns) and queue[["start_s", "end_s"]].isna().any().any():
        queue = queue.drop(columns=["start_s", "end_s"])
    return queue


class Dashboard:
    def __init__(self, results_dir, speed):
        self.results_dir, self.speed = results_dir, speed
        self.lock, self.stop, self.jpeg, self.out = threading.Lock(), threading.Event(), b"", None
        self.state = {"mode": "setup", "tanks": default_tanks(), "error": None}

    def update(self, **kw):
        with self.lock:
            self.state.update(kw)

    def start(self, source, queue, name):
        out = self.results_dir / f"live_{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        (out / "snapshots").mkdir(parents=True, exist_ok=True)
        queue.to_csv(out / "recipe_queue.csv", index=False)
        self.out, self.jpeg = out, b""
        self.stop.clear()
        with self.lock:
            self.state.update(mode="running", error=None, t=0.0, current=None, results=[], events=[],
                              source=str(source), out=str(out), csv=str(out / "vials.csv"), phase=None,
                              render=None, twin=False)

        def work():
            try:
                video = source
                if isinstance(source, Twin):                      # render first, then play at normal speed
                    video = source.wait_for_video(out, self)
                    self.update(twin=True)
                if video is not None:
                    run_live(video, queue, out, 1.0 if isinstance(source, Twin) else self.speed, self)
            except Exception as e:                                # shown on the dashboard
                self.update(error=str(e))
            self.update(mode="finished", current=None)

        threading.Thread(target=work, daemon=True).start()

    def serve(self, host, port):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, ctype, code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                snap = dash.out / "snapshots" / Path(self.path).name if dash.out else None
                if self.path == "/":
                    self.send(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    with dash.lock:
                        body = json.dumps(dash.state).encode()
                    self.send(body, "application/json")
                elif self.path.startswith("/frame.jpg") and dash.jpeg:
                    with dash.lock:
                        jpeg = dash.jpeg
                    self.send(jpeg, "image/jpeg")
                elif self.path.startswith("/snapshots/") and snap and snap.is_file():
                    self.send(snap.read_bytes(), "image/jpeg")
                else:
                    self.send_error(404)

            def do_POST(self):
                size = int(self.headers.get("Content-Length", 0))
                if self.path.startswith("/upload"):                # the video file, saved in chunks
                    name = Path(urllib.parse.unquote(self.path.partition("name=")[2])).name or "video.mp4"
                    uploads = dash.results_dir / "live_uploads"
                    uploads.mkdir(parents=True, exist_ok=True)
                    with open(uploads / name, "wb") as f:
                        while size > 0:
                            chunk = self.rfile.read(min(size, 1 << 20))
                            if not chunk:
                                break
                            f.write(chunk)
                            size -= len(chunk)
                    return self.send(json.dumps({"video": name}).encode(), "application/json")
                body = self.rfile.read(size)
                if self.path == "/stop":
                    dash.stop.set()
                    return self.send(b"{}", "application/json")
                if self.path == "/new":
                    if dash.state["mode"] != "running":
                        dash.update(mode="setup", error=None)
                    return self.send(b"{}", "application/json")
                if self.path != "/start" or dash.state["mode"] == "running":
                    return self.send_error(400)
                try:
                    form = json.loads(body)
                    queue = read_queue(io.StringIO(form["recipes_csv"]), form["tanks"])
                    if form.get("twin"):
                        source, name = Twin(form["twin"]), "twin"
                        # the twin makes its own timing, so a queue's video times (start_s/end_s) don't apply
                        queue = queue.drop(columns=["start_s", "end_s"], errors="ignore")
                        source.send_queue(queue)
                    elif form.get("camera"):
                        cam = str(form["camera"]).strip()
                        source, name = (int(cam), f"camera{cam}") if cam.isdigit() else (cam, "camera")
                    else:
                        source = dash.results_dir / "live_uploads" / Path(form["video"]).name
                        if not source.is_file():
                            raise ValueError("the uploaded video was not found; upload it again")
                        source, name = str(source), source.stem
                    dash.update(tanks=form["tanks"])
                    dash.start(source, queue, name)
                    self.send(b"{}", "application/json")
                except Exception as e:
                    self.send(json.dumps({"error": str(e)}).encode(), "application/json", 400)

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"dashboard: http://{host}:{port}", flush=True)


class Twin:
    """The digital twin (twin_tools/render_live_video.py), reached over HTTP. The dashboard sends it the recipe
    queue; the twin renders the whole fill-camera video; then the dashboard downloads it and plays it."""

    def __init__(self, address):
        self.url = address.strip().rstrip("/")

    def call(self, path, data=None, timeout=10):
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                body = json.loads(body).get("error") or body
            except ValueError:
                pass
            raise RuntimeError(f"the twin answered {e.code}: {body}")
        except OSError as e:
            raise RuntimeError(f"cannot reach the twin at {self.url} ({e}); is the SSH tunnel open?")

    def send_queue(self, queue):
        vials = [{"vial": int(q.vial), "recipe_id": str(q.recipe_id), "orange_ml": float(q.t200_ml),
                  "blue_ml": float(q.t400_ml), "clear_ml": float(q.t100_ml)} for q in queue.itertuples()]
        self.call("/queue", json.dumps({"vials": vials}).encode())

    def wait_for_video(self, out, dash):
        """Show the render progress until the video is ready, then download it. Returns its path or None."""
        while not dash.stop.is_set():
            status = json.loads(self.call("/status"))
            dash.update(phase="rendering", render=status)
            try:
                jpeg = self.call("/latest.jpg")
                with dash.lock:
                    dash.jpeg = jpeg
            except RuntimeError:
                pass                                              # no frame yet
            if status["state"] == "done":
                dash.update(phase="downloading")
                path = out / "twin_video.mp4"
                path.write_bytes(self.call("/video", timeout=300))
                return path
            if status["state"] in ("error", "cancelled", "idle"):
                raise RuntimeError(f"the twin stopped: {status.get('error') or status['state']}")
            time.sleep(2)
        self.call("/cancel", b"{}")                               # Stop pressed while rendering
        return None


def run_live(source, queue, out, speed, dash):
    net, threshold = load_model()
    scored = {"start_s", "end_s"} <= set(queue.columns)
    queue = queue.assign(recipe_id=queue.recipe_id.astype(str)).reset_index(drop=True)
    prev_recipe = queue.recipe_id.shift()
    is_file = not isinstance(source, int) and Path(str(source)).is_file()
    cap = cv2.VideoCapture(str(source) if is_file else source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open the video source {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / FPS_SAMPLE))
    dash.update(speed=speed if is_file else 1, threshold=round(float(threshold), 3), scored=scored,
                vials_in_queue=len(queue), kind="file" if is_file else "camera", phase="playing")

    sig, recent, used, rows = [], deque(), set(), []
    prev, last_spike, c0, i, last_sample, last_jpeg = None, None, 0.5, 0, -1.0, 0.0
    wall0 = time.monotonic()

    def next_row():
        return next((k for k in range(len(queue)) if k not in used), None)

    def close_vial(c1):
        a = np.array(sig)
        v = vial_window(a[:, 0], smoothed(a[:, 1]), a[:, 3], c0, c1, a[1, 0] - a[0, 0]) if len(a) > 1 else None
        if v is None:
            return
        frame = next((f for ft, f in recent if ft >= v["final_s"] - 1e-6), recent[-1][1])
        if scored:                                            # largest time overlap, as in the recorded mode
            free = [k for k in range(len(queue)) if k not in used]
            best, k = max(((min(v["end_s"], queue.end_s[k]) - max(v["start_s"], queue.start_s[k]), k) for k in free),
                          default=(0, None))
            k = k if best > 0 else None
        else:
            k = next_row()
        if k is None:
            with dash.lock:
                dash.state["events"].insert(0, f"{v['start_s']:.0f}-{v['end_s']:.0f} s: vial detected with no recipe "
                                               "in the queue (not checked)")
            return
        used.add(k)
        q = queue.iloc[[k]].reset_index(drop=True)
        rec = recipe_inputs(q)
        est = estimate(net, [frame], rec)[0]
        exp_o, exp_b = float(rec.expected_clean_orange[0]), float(rec.expected_clean_blue[0])
        gap = max(abs(est[0] - exp_o), abs(est[1] - exp_b))
        name = f"vial{int(q.vial[0]):02d}_final.jpg"
        cv2.imwrite(str(out / "snapshots" / name), frame)
        r = {"vial": int(q.vial[0]), "recipe_id": q.recipe_id[0],
             "orange_ml": float(q.t200_ml[0]), "blue_ml": float(q.t400_ml[0]), "clear_ml": float(q.t100_ml[0]),
             "detected_start_s": v["start_s"], "detected_end_s": v["end_s"], "snapshot_s": v["final_s"],
             "pours_seen": v["pours_seen"], "snapshot": f"snapshots/{name}",
             "expected_orange": round(exp_o, 3), "expected_blue": round(exp_b, 3),
             "est_orange": round(float(est[0]), 3), "est_blue": round(float(est[1]), 3),
             "gap": round(float(gap), 3), "flag": int(gap > threshold)}
        if scored:
            pr = prev_recipe[k]
            r["truth_carryover"] = -1 if pd.isna(pr) else int(q.recipe_id[0] != pr)
        rows.append(r)
        pd.DataFrame(rows).to_csv(out / "vials.csv", index=False)
        dash.update(results=rows)
        print(f"vial {r['vial']} recipe {r['recipe_id']}: est {100 * r['est_orange']:.0f}% orange "
              f"{100 * r['est_blue']:.0f}% blue, expected {100 * exp_o:.0f}/{100 * exp_b:.0f}, gap "
              f"{100 * gap:.0f} pts -> {'ALERT' if r['flag'] else 'OK'}", flush=True)

    while not dash.stop.is_set():
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[:2] != (480, 640):
            raise RuntimeError(f"the video is {f.shape[1]}x{f.shape[0]}; the model needs the 640x480 fill-camera view")
        t = i / fps if is_file else time.monotonic() - wall0
        if is_file:
            ahead = t / speed - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)
        recent.append((t, f))
        while recent[0][0] < t - 3:
            recent.popleft()
        if (i % step == 0) if is_file else (t - last_sample >= 1 / FPS_SAMPLE - 1e-6):
            last_sample = t
            x0, y0, x1, y1 = VIAL_REGION
            cx0, cy0, cx1, cy1 = CARRIER_REGION
            vial, carrier = f[y0:y1, x0:x1].astype(np.float32), f[cy0:cy1, cx0:cx1].astype(np.float32)
            sat = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 1].mean()
            mv = 0.0 if prev is None else float(np.abs(vial - prev[0]).mean())
            mc = 0.0 if prev is None else float(np.abs(carrier - prev[1]).mean())
            sig.append((t, mv, mc, sat))
            prev = (vial, carrier)
            if mc > CARRIER_MOTION:
                if last_spike is None or t - last_spike > 3:
                    close_vial(t - 0.5)                         # a new carrier move ends the vial under the nozzle
                last_spike, c0 = t, t + 0.5
            a = np.array(sig[-600:])
            sm = smoothed(a[:, 1])
            k = next_row()
            cur = None if k is None else {"vial": int(queue.vial[k]), "recipe_id": queue.recipe_id[k],
                                          "orange_ml": float(queue.t200_ml[k]), "blue_ml": float(queue.t400_ml[k]),
                                          "clear_ml": float(queue.t100_ml[k])}
            if cur:
                moving = last_spike is not None and t - last_spike < 1.5
                cur.update(pours=count_pours(a[:, 0], sm, c0, t, a[1, 0] - a[0, 0]) if len(a) > 1 else 0,
                           status="carrier moving" if moving else "pouring" if sm[-1] > POUR_MOTION else "waiting")
            dash.update(t=round(t, 1), current=cur)
        if time.monotonic() - last_jpeg >= 0.1:
            last_jpeg = time.monotonic()
            view = f.copy()
            cv2.rectangle(view, VIAL_REGION[:2], VIAL_REGION[2:], (0, 220, 255), 1)
            with dash.lock:
                dash.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        i += 1
    cap.release()
    if sig and not dash.stop.is_set():
        close_vial(sig[-1][0] - 0.5)                            # the end of the video bounds the last vial
    pd.DataFrame(rows).to_csv(out / "vials.csv", index=False)
    print(f"done: {len(rows)} vials checked, {sum(r['flag'] for r in rows)} alerts; wrote {out / 'vials.csv'}",
          flush=True)


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vial Color Check</title>
<style>
:root{--bg:#10141a;--panel:#19202a;--line:#2a3442;--text:#e6edf3;--dim:#8b98a8;--ok:#2ea043;--bad:#e5534b;
--orange:#f0883e;--blue:#4c8eda;--accent:#388bfd}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.4 system-ui,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 20px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0}header span{color:var(--dim);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:16px;padding:16px 20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 10px}
button{font:inherit;padding:8px 16px;border-radius:6px;border:1px solid var(--line);background:var(--line);color:var(--text);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);font-weight:600}button.stop{background:var(--bad);border-color:var(--bad)}
button:disabled{opacity:.5;cursor:default}
#setup{max-width:760px;margin:20px auto;padding:0 16px;display:grid;gap:16px}
#setup label{display:block;margin:8px 0 4px;color:var(--dim);font-size:13px}
#setup input[type=text],#setup select{font:inherit;padding:7px 9px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--text);width:100%}
.choice{display:flex;gap:18px;margin-bottom:6px}.choice label{display:flex;gap:6px;align-items:center;color:var(--text)!important;font-size:15px!important}
.tanks{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.hint{color:var(--dim);font-size:13px;margin:6px 0 0}
#err{color:var(--bad);min-height:1.4em}
#live{width:100%;border-radius:4px;display:block;background:#000;aspect-ratio:4/3}
.now{display:flex;gap:24px;flex-wrap:wrap;margin-top:12px}.now div{color:var(--dim);font-size:12px}
.now div b{display:block;font-size:22px;color:var(--text)}
#banner{font-size:26px;font-weight:700;padding:14px;border-radius:6px;text-align:center;margin-bottom:12px;background:var(--line)}
#banner.ok{background:var(--ok)}#banner.bad{background:var(--bad)}
.last{display:flex;gap:14px}.last img{height:230px;border-radius:4px;background:#000}
.bars{flex:1}.bar{margin:6px 0 12px}.bar label{font-size:12px;color:var(--dim)}
.track{position:relative;height:16px;background:var(--bg);border-radius:3px;margin-top:3px}
.fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px}
.mark{position:absolute;top:-4px;bottom:-4px;width:3px;background:var(--text)}
.totals{display:flex;gap:20px;margin-top:4px;flex-wrap:wrap}.totals div{color:var(--dim);font-size:12px}
.totals b{display:block;font-size:20px;color:var(--text)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:5px 6px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500}td.bad{color:var(--bad);font-weight:700}td.ok{color:var(--ok)}
.wide{grid-column:1/-1}.scroll{max-height:320px;overflow:auto}#events{color:var(--dim);font-size:13px}
#saved{color:var(--dim);font-size:13px;word-break:break-all}
@media (max-width:820px){main{grid-template-columns:1fr;padding:12px 16px}.last img{height:160px}.tanks{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Vial Color Check &middot; Fill Station</h1><span id="hdr"></span></header>

<div id="setup" hidden>
 <section class="card"><h2>1 &middot; Video</h2>
  <div class="choice"><label><input type="radio" name="src" value="file" checked> Upload a video</label>
   <label><input type="radio" name="src" value="camera"> Connect a camera</label>
   <label><input type="radio" name="src" value="twin"> Digital twin</label></div>
  <div id="src-file"><input type="file" id="video" accept="video/*"><p class="hint">The fill-camera view, 640&times;480.</p></div>
  <div id="src-camera" hidden><input type="text" id="camera" placeholder="0  (USB camera number)  or  rtsp://camera-address/stream">
   <p class="hint">Camera number for a USB camera, or the camera's stream address.</p></div>
  <div id="src-twin" hidden><input type="text" id="twin" value="http://localhost:20700">
   <p class="hint">The twin (twin_tools/render_live_video.py) running in Isaac Sim. Start check sends it the recipe queue;
    the twin plays out the vials and renders the fill-camera video (about 2 minutes per vial), then the video plays here at
    normal speed.</p></div>
 </section>
 <section class="card"><h2>2 &middot; Recipe queue</h2>
  <input type="file" id="recipes" accept=".csv,text/csv">
  <p class="hint">One row per vial, in order: <code>vial, recipe_id, t100_ml, t200_ml, t400_ml</code> (CSI's format; blank = tank not used).
   Optional <code>start_s, end_s</code> columns match vials by time and score the run.</p>
 </section>
 <section class="card"><h2>3 &middot; Tank assignment</h2>
  <div class="tanks"><div><label>Orange dye</label><select id="t-orange"></select></div>
   <div><label>Blue dye</label><select id="t-blue"></select></div>
   <div><label>Clear (diluent)</label><select id="t-clear"></select></div></div>
  <p class="hint">Which tank holds each color on this line. Tanks not listed are treated as out of service.</p>
 </section>
 <div><button class="primary" id="start">Start check</button> <span id="err"></span></div>
</div>

<main id="run" hidden>
 <section class="card"><h2>Fill camera</h2><img id="live" alt="live fill camera">
  <div class="now" id="now"></div>
  <p><button class="stop" id="stopbtn">Stop</button> <button id="newbtn" hidden>New run</button></p>
  <div id="saved"></div></section>
 <section class="card"><h2>Last finished vial</h2><div id="banner">Waiting for the first vial</div>
  <div class="last"><img id="snap" alt="" hidden><div class="bars" id="bars"></div></div>
  <div class="totals" id="totals"></div><div id="events"></div></section>
 <section class="card wide"><h2>Vial log</h2><div class="scroll"><table id="log"></table></div></section>
</main>
<script>
const esc=x=>String(x).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const $=id=>document.getElementById(id),pct=v=>Math.round(100*v)+"%";
const ml=r=>[r.orange_ml&&r.orange_ml+" mL orange",r.blue_ml&&r.blue_ml+" mL blue",r.clear_ml&&r.clear_ml+" mL clear"].filter(Boolean).join(" + ");
let filled=false,mode=null;
const srcKind=()=>document.querySelector("input[name=src]:checked").value;
document.querySelectorAll("input[name=src]").forEach(r=>r.onchange=()=>{const k=srcKind();
 $("src-file").hidden=k!="file";$("src-camera").hidden=k!="camera";$("src-twin").hidden=k!="twin"});
const readText=f=>new Promise((ok,no)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=no;r.readAsText(f)});
$("start").onclick=async()=>{
 $("err").textContent="";const kind=srcKind(),cam=kind=="camera",tw=kind=="twin";
 const rec=$("recipes").files[0],vid=$("video").files[0];
 if(!rec)return $("err").textContent="Choose the recipe queue CSV.";
 if(cam&&!$("camera").value.trim())return $("err").textContent="Enter the camera number or stream address.";
 if(tw&&!$("twin").value.trim())return $("err").textContent="Enter the twin's address.";
 if(kind=="file"&&!vid)return $("err").textContent="Choose a video file.";
 $("start").disabled=true;$("start").textContent=kind=="file"?"Uploading...":"Connecting...";
 const body={recipes_csv:await readText(rec),tanks:{orange:$("t-orange").value,blue:$("t-blue").value,clear:$("t-clear").value}};
 if(tw)body.twin=$("twin").value.trim();
 else if(cam)body.camera=$("camera").value.trim();
 else{const u=await fetch("/upload?name="+encodeURIComponent(vid.name),{method:"POST",body:vid}).catch(()=>null);
  if(!u||!u.ok){$("start").disabled=false;$("start").textContent="Start check";return $("err").textContent="Upload failed."}
  body.video=(await u.json()).video}
 const r=await fetch("/start",{method:"POST",body:JSON.stringify(body)});
 $("start").disabled=false;$("start").textContent="Start check";
 if(!r.ok)$("err").textContent=(await r.json().catch(()=>({error:"could not start"}))).error;
};
$("stopbtn").onclick=()=>fetch("/stop",{method:"POST"});
$("newbtn").onclick=()=>fetch("/new",{method:"POST"});
function bar(name,est,exp,color){return `<div class="bar"><label>${name}: measured ${pct(est)}, expected ${pct(exp)}</label>
 <div class="track"><div class="fill" style="width:${Math.min(100,100*est)}%;background:${color}"></div><div class="mark" style="left:${100*exp}%"></div></div></div>`}
async function tick(){
 let s;try{s=await (await fetch("/state")).json()}catch(e){return}
 if(!filled){for(const c of ["orange","blue","clear"])$("t-"+c).innerHTML=["T100","T200","T300","T400"].map(t=>`<option${s.tanks[c]==t?" selected":""}>${t}</option>`).join("");filled=true}
 if(s.mode!=mode){mode=s.mode;$("setup").hidden=mode!="setup";$("run").hidden=mode=="setup";
  if(mode=="running"){$("banner").className="";$("banner").textContent="Waiting for the first vial";$("snap").hidden=true;$("bars").innerHTML=""}}
 if(mode=="setup"){$("hdr").textContent="Setup";if(s.error)$("err").textContent=s.error;return}
 const run=mode=="running";$("stopbtn").hidden=!run;$("newbtn").hidden=run;
 $("hdr").textContent=`${run?(s.phase=="rendering"?"TWIN RENDERING":s.phase=="downloading"?"DOWNLOADING":s.twin?"TWIN VIDEO":s.kind=="camera"?"LIVE":"PLAYING"):"FINISHED"} · ${Math.floor((s.t||0)/60)}:${String(Math.floor((s.t||0)%60)).padStart(2,"0")}`+(s.speed>1?` · ${s.speed}x`:"");
 $("saved").innerHTML=(s.error?`<p style="color:var(--bad)">${esc(s.error)}</p>`:"")+(run?`Predictions are saved as each vial finishes:<br>${s.csv}`:`Predictions saved to:<br>${s.csv}`);
 if(run&&(s.phase=="rendering"||s.phase=="downloading")){const r=s.render||{},p=Math.round(100*(r.progress||0)),m=r.eta_s!=null?Math.ceil(r.eta_s/60):null;
  $("banner").className="";$("banner").textContent=s.phase=="downloading"?"Video ready: loading it":`The twin is rendering the video: ${p}%`;
  $("now").innerHTML=`<div>Rendered<b>${p}%</b></div><div>Time left<b>${m==null?"estimating":"about "+m+" min"}</b></div><div>Vials<b>${r.vials||"?"}</b></div>
   <div style="flex-basis:100%">The twin plays out the recipe queue and renders the fill camera. When it is done, the video plays here at normal speed and every vial is checked.</div>`;return}
 const c=s.current;
 $("now").innerHTML=c?`<div>Now filling<b>Vial ${c.vial}</b></div><div>Recipe<b>${esc(c.recipe_id)}</b></div>
  <div>Pours seen<b>${c.pours}</b></div><div>Status<b>${c.status}</b></div><div style="flex-basis:100%">${ml(c)}</div>`:`<div>${run?"Recipe queue finished":"Run finished"}</div>`;
 const R=s.results||[],last=R[R.length-1];
 if(last){$("banner").className=last.flag?"bad":"ok";
  $("banner").textContent=last.flag?`ALERT · vial ${last.vial}: possible carryover`:`OK · vial ${last.vial} matches recipe ${last.recipe_id}`;
  const im=$("snap");im.hidden=false;if(!im.src.endsWith(last.snapshot))im.src="/"+last.snapshot;
  $("bars").innerHTML=`<div style="color:var(--dim);font-size:13px">Recipe ${esc(last.recipe_id)}: ${ml(last)}</div>`+
   bar("Orange",last.est_orange,last.expected_orange,"var(--orange)")+bar("Blue",last.est_blue,last.expected_blue,"var(--blue)")+
   `<div>Gap <b>${Math.round(100*last.gap)} points</b> (alert above ${Math.round(100*s.threshold)})</div>`}
 let tot=`<div>Vials checked<b>${R.length} of ${s.vials_in_queue||0}</b></div><div>Alerts<b>${R.filter(r=>r.flag).length}</b></div>`;
 if(s.scored){const sc=R.filter(r=>r.truth_carryover>=0),co=sc.filter(r=>r.truth_carryover==1),cl=sc.filter(r=>r.truth_carryover==0);
  tot+=`<div>Carryover caught*<b>${co.filter(r=>r.flag).length} of ${co.length}</b></div><div>Clean flagged*<b>${cl.filter(r=>r.flag).length} of ${cl.length}</b></div>`}
 $("totals").innerHTML=tot;
 $("events").innerHTML=(s.scored?"<p>* scored against an assumed label: the first vial after a recipe change is carryover.</p>":"")+(s.events||[]).map(e=>`<p>${esc(e)}</p>`).join("");
 $("log").innerHTML=`<tr><th>Vial</th><th>Recipe</th><th>Measured (orange / blue)</th><th>Expected</th><th>Gap</th><th>Result</th>${s.scored?"<th>Assumed label</th>":""}</tr>`+
  R.slice().reverse().map(r=>`<tr><td>${r.vial}</td><td>${esc(r.recipe_id)}</td><td>${pct(r.est_orange)} / ${pct(r.est_blue)}</td><td>${pct(r.expected_orange)} / ${pct(r.expected_blue)}</td>
  <td>${Math.round(100*r.gap)}</td><td class="${r.flag?"bad":"ok"}">${r.flag?"ALERT":"OK"}</td>${s.scored?`<td>${{"1":"carryover","0":"clean","-1":"not scored"}[r.truth_carryover]}</td>`:""}</tr>`).join("")}
setInterval(tick,500);tick();
const live=$("live");let loading=false;live.onload=live.onerror=()=>{loading=false};
setInterval(()=>{if(mode&&mode!="setup"&&!loading){loading=true;live.src="/frame.jpg?"+Date.now()}},100);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="fill-camera video (640x480, the vial_fill view)")
    ap.add_argument("--recipes", help="recipe queue CSV (CSI's video5_vial_recipes.csv format)")
    ap.add_argument("--out-dir", default=None, help="default: model/results/check_<video name>")
    ap.add_argument("--live", action="store_true",
                    help="open the dashboard: pick a video or camera, the recipe CSV and the tank assignment there")
    ap.add_argument("--speed", type=float, default=1.0, help="--live: replay speed for uploaded videos")
    ap.add_argument("--host", default="127.0.0.1",
                    help="--live: dashboard address (0.0.0.0 = other screens on the network too; no password, "
                         "so only on a trusted network)")
    ap.add_argument("--port", type=int, default=8000, help="--live: dashboard port")
    args = ap.parse_args()
    if args.live:
        Dashboard(MODEL_DIR / "results", args.speed).serve(args.host, args.port)
        print("Ctrl+C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
    if not (args.video and args.recipes):
        ap.error("--video and --recipes are needed (or --live to pick them in the dashboard)")
    video = Path(args.video).expanduser().resolve()
    out = Path(args.out_dir or MODEL_DIR / "results" / f"check_{video.stem}")
    (out / "snapshots").mkdir(parents=True, exist_ok=True)

    queue = pd.read_csv(args.recipes)
    for c in ["t100_ml", "t200_ml", "t400_ml"]:
        queue[c] = pd.to_numeric(queue[c], errors="coerce").fillna(0.0)
    scored = {"start_s", "end_s"} <= set(queue.columns)
    detected = find_vials(signals(video))
    match = match_queue(queue, detected)
    used = [id(d) for d in match.values()]
    false_detections = [[d["start_s"], d["end_s"]] for d in detected if id(d) not in used]
    missed = [int(queue.vial[i]) for i in range(len(queue)) if i not in match]
    print(f"{len(queue)} vials in the recipe queue, {len(detected)} detected: {len(match)} matched, "
          f"false detections {false_detections}, missed {missed}", flush=True)

    n_queue = len(queue)
    prev_recipe = queue.recipe_id.shift()                      # from the full queue, before dropping missed vials
    rows = sorted(match)
    queue, prev_recipe = queue.iloc[rows].reset_index(drop=True), prev_recipe.iloc[rows].reset_index(drop=True)
    vials = [match[i] for i in rows]
    cap = cv2.VideoCapture(str(video))
    frames = []
    for v, q in zip(vials, queue.itertuples()):
        f = frame_at(cap, v["final_s"])
        cv2.imwrite(str(out / "snapshots" / f"vial{int(q.vial):02d}_final.jpg"), f)
        frames.append(f)

    rec = recipe_inputs(queue)
    net, threshold = load_model()
    est = estimate(net, frames, rec)
    df = pd.DataFrame({"vial": queue.vial, "recipe_id": queue.recipe_id,
                       "detected_start_s": [v["start_s"] for v in vials], "detected_end_s": [v["end_s"] for v in vials],
                       "snapshot_s": [v["final_s"] for v in vials], "pours_seen": [v["pours_seen"] for v in vials],
                       "snapshot": [f"snapshots/vial{int(v):02d}_final.jpg" for v in queue.vial],
                       "expected_orange": rec.expected_clean_orange.round(3),
                       "expected_blue": rec.expected_clean_blue.round(3),
                       "est_orange": est[:, 0].round(3), "est_blue": est[:, 1].round(3)})
    df["gap"] = np.maximum(np.abs(df.est_orange - df.expected_orange), np.abs(df.est_blue - df.expected_blue)).round(3)
    df["flag"] = (df.gap > threshold).astype(int)

    here = MODEL_DIR.parent                                    # paths relative to version_1/
    metrics = {"info": {"video": os.path.relpath(video, here),
                        "recipes": os.path.relpath(Path(args.recipes).resolve(), here),
                        "model": str(CHECKPOINT.relative_to(MODEL_DIR.parent)), "threshold": round(float(threshold), 3),
                        "check": "final vial (last pour)", "calibration": "none",
                        "vials_in_queue": n_queue, "vials_detected": len(detected), "vials_matched": len(match),
                        "false_detections_s": false_detections, "missed_vials": missed},
               "vials_flagged": f"{int(df.flag.sum())} of {len(df)}"}
    if scored:
        df["truth_carryover"] = np.where(prev_recipe.isna(), -1, (queue.recipe_id != prev_recipe).astype(int))
        s = df[df.truth_carryover >= 0]
        co, clean = s[s.truth_carryover == 1], s[s.truth_carryover == 0]
        metrics["info"]["ground_truth"] = ("first vial after a recipe change = carryover (assumption); "
                                           "first vial in the queue not scored")
        metrics["info"]["matching"] = "detected vial window with the largest overlap with the label's dispense window"
        metrics.update({"carryover_vials_caught": f"{int(co.flag.sum())} of {len(co)}",
                        "clean_vials_falsely_flagged": f"{int(clean.flag.sum())} of {len(clean)}",
                        "scores": scores(s.truth_carryover.values, s.gap.values, s.flag.values),
                        "mix_gap_median": {"carryover": round(float(co.gap.median()), 3),
                                           "clean": round(float(clean.gap.median()), 3)}})
    df.to_csv(out / "vials.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    contact_sheet(frames, df, out / "contact_sheet.jpg")
    print(json.dumps({k: v for k, v in metrics.items() if k != "info"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

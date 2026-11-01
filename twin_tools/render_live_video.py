"""The digital twin as a fill camera on demand: it waits for a recipe queue, plays out the vials the way the
real line would, renders the fill-camera video, and hands the finished video to the dashboard
(model/vial_check.py), which then plays it at normal speed and checks every vial.

What the twin decides on its own (nothing about the result is set by the user):
  - Carryover, from the leftover physics: the station keeps one typical leftover in its shared mixing path
    (drawn once per run, 5-30 mL), with a little variation from vial to vial (about 1.5 mL). Every vial starts
    with that leftover of the previous recipe's mix, so a vial right after a recipe change is off; the same
    recipe again is clean. This is the same model the training pictures use (render_dataset.py stages()).
  - Timing: how long each vial waits, how fast each tank pours (5-7 mL per second) and the pauses between pours.
  - Where the vial stops under the nozzle (a little off each time) and, now and then, an empty carrier that
    passes without being filled (about 1 in 10 changes), as in real video 5.

    # In the YAAD team workspace: copy this file, render_dataset.py and recipe_rules.json to /workspace/sim/,
    # then run Isaac Sim 5.1 as a container (first start compiles shaders for a few minutes):
    docker run -d --name isaac -u 0 --network host --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
        -v /workspace:/workspace --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:5.1.0 \
        /workspace/sim/render_live_video.py --twin /workspace/csi_digital_twin_v1 --port 20700 \
        --out-dir /workspace/twin_live_runs
    docker logs -f isaac            # wait for "ready: waiting for a recipe queue"; docker rm -f isaac when done
    # Laptop (VPN on): ssh -N -L 20700:127.0.0.1:20700 yt-teamsantiago86bb@172.30.102.165
    # then in the dashboard pick "Digital twin" with http://localhost:20700 and start a check.

    # On a laptop, no Isaac Sim:
    python3 twin_tools/render_live_video.py --dry-run --queue twin_tools/demo_queues/queue_1_normal_batch.csv
    #   prints the plan (timing, leftover, true mix of every vial) and writes truth.csv to --out-dir
    python3 twin_tools/render_live_video.py --preview --port 20700
    #   same server with simple drawn pictures instead of Isaac Sim (to test the dashboard connection)

Server: POST /queue starts a run; GET /status gives progress and time left; GET /latest.jpg is the newest frame
(a preview while rendering); GET /video is the finished video; POST /cancel stops a run.
The video has 10 frames per twin second. Rendering one picture takes about 2 s on the Spark, so a picture is only
re-rendered when something changes (liquid, stream, carriers), at most every 2nd frame; while nothing moves, the
last picture is reused with fresh camera noise.

Each run is saved to --out-dir/twin_<date-time>/: video.mp4, queue.csv, run_info.json and truth.csv (the true
mix and carryover of every vial, and when it was under the nozzle). truth.csv is NOT sent to the dashboard.
Nothing in the twin files is changed: all edits stay in memory.
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from render_dataset import (DEFAULT_TWIN, FULL_ML, LAST_FRAME, LIQUID, LIQUID_MESH, ORGANIZERS_OPACITY,  # noqa: E402
                            STREAM, TANKS, capture, liquid_bottom, liquid_color, load_rules, validate)

FPS = 10                        # twin frames per twin second
RENDER_EVERY = 2                # while something moves, render every 2nd frame (5 per twin second)
SLOT = 3.34                     # distance between carriers along the track (twin units)
STREAM_FRAMES = (120, 340)      # twin animation frames where the pouring stream is fully formed

# Which twin parts move together (prim names under FillStation_LocalRoot). Carrier 0 is under the nozzle and
# holds the vial with the liquid and barcode; 1 and 2 wait behind it.
CARRIER_PARTS = {0: ["V3_carrier_0_", "V3_estimated_fastener_0", "Fill_vial_active", "Fill_liquid",
                     "Fill_barcode", "Fill_label_top_rule"],
                 1: ["V3_carrier_1_", "V3_estimated_fastener_1", "Fill_vial_neighbour_1"],
                 2: ["V3_carrier_2_", "V3_estimated_fastener_2", "Fill_vial_neighbour_2"]}

# Camera look for the stream: fixed for the whole run (one real camera). Low noise, because frame-to-frame
# noise would look like movement to the vial finder.
LOOK = {"exposure_gamma": 1.3, "exposure_gain": 0.87, "noise_sigma": 0.5, "blur": 1}


# ----------------------------------------------------------------------------------------------
# The plan: what happens when (pure Python, no Isaac Sim)
# ----------------------------------------------------------------------------------------------
def read_queue(path_or_rows):
    """Queue rows -> list of {vial, recipe_id, T200, T400, T100}. Accepts a CSV path (CSI's columns) or the
    dashboard's rows (orange_ml, blue_ml, clear_ml)."""
    rows = list(csv.DictReader(open(path_or_rows))) if isinstance(path_or_rows, (str, Path)) else path_or_rows
    out = []
    for n, r in enumerate(rows, 1):
        num = lambda k: float(r.get(k) or 0)                                    # noqa: E731
        ml = ({"T200": num("orange_ml"), "T400": num("blue_ml"), "T100": num("clear_ml")} if "orange_ml" in r
              else {"T200": num("t200_ml"), "T400": num("t400_ml"), "T100": num("t100_ml")})
        out.append({"vial": int(float(r.get("vial") or n)), "recipe_id": str(r["recipe_id"]), **ml})
    return out


def plan_run(queue, rules, seed):
    """Events on the twin clock. Returns (events, truth rows, run info)."""
    if not queue:
        raise ValueError("the recipe queue is empty")
    rng = random.Random(seed)
    station_leftover = round(rng.uniform(5.0, 30.0), 1)
    events, truth, t = [], [], 1.0
    path_mix = None                                              # what the mixing path holds (mL per tank)
    exiting_full = False
    for q in queue:
        cur = {k: q[k] for k in TANKS}
        problems = validate({"recipe_id": q["recipe_id"], **cur}, rules)
        if problems:
            raise ValueError(f"vial {q['vial']} (recipe {q['recipe_id']}): {', '.join(problems)}")
        total = sum(cur.values())
        if rng.random() < 0.1 and truth:                           # an empty carrier passes now and then
            d = rng.uniform(1.6, 2.4)
            events.append({"kind": "move", "t0": t, "t1": t + d, "exit_full": exiting_full, "stop": 0.0})
            t += d + rng.uniform(5.0, 8.0)                           # sits empty, under 10 s
            exiting_full = False
        d = rng.uniform(1.6, 2.4)
        stop = rng.uniform(-0.06, 0.06)                              # where this vial stops (twin units)
        events.append({"kind": "move", "t0": t, "t1": t + d, "exit_full": exiting_full, "stop": stop})
        t += d
        start = t
        prev = path_mix if path_mix else cur                         # the first vial: the path holds its recipe
        residue = min(max(station_leftover + rng.gauss(0, 1.5), 2.0), total / 2)
        keep = 1.0 - residue / total                                 # the rest of this mix stays in the path
        cum = {k: 0.0 for k in TANKS}
        t += rng.uniform(1.5, 3.5)                                   # wait before the first pour
        order = [k for k in rules["dispense_order"] if cur[k] > 0]
        for n, k in enumerate(order):
            start_cum = dict(cum)
            if n == 0:                                               # the leftover comes out with pour 1
                for j in TANKS:
                    cum[j] += residue * prev[j] / sum(prev.values())
            cum[k] += cur[k] * keep
            d = (sum(cum.values()) - sum(start_cum.values())) / rng.uniform(5.0, 7.0)
            events.append({"kind": "pour", "t0": t, "t1": t + d, "tank": k, "from": start_cum, "to": dict(cum)})
            t += d + (rng.uniform(1.5, 4.0) if n < len(order) - 1 else rng.uniform(3.0, 6.0))
        final_total = sum(cum.values())
        truth.append({"vial": q["vial"], "recipe_id": q["recipe_id"], "t200_ml": cur["T200"],
                      "t400_ml": cur["T400"], "t100_ml": cur["T100"],
                      "previous_recipe": truth[-1]["recipe_id"] if truth else q["recipe_id"],
                      "residue_ml": round(residue, 2),
                      "carryover": int(bool(truth) and any(prev[k] != cur[k] for k in TANKS)),
                      "actual_orange": round(cum["T200"] / final_total, 4),
                      "actual_blue": round(cum["T400"] / final_total, 4),
                      "under_nozzle_from_s": round(start, 2), "under_nozzle_to_s": round(t, 2)})
        path_mix, exiting_full = cur, True
    d = rng.uniform(1.6, 2.4)
    events.append({"kind": "move", "t0": t, "t1": t + d, "exit_full": exiting_full, "stop": 0.0})
    t += d + 2.0
    info = {"seed": seed, "station_leftover_ml": station_leftover, "twin_seconds": round(t, 1),
            "frames": int(t * FPS), "vials": len(queue)}
    return events, truth, info


def smooth(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def state_at(t, events):
    """Scene at twin time t: carrier offsets, liquid mL per tank, stream on/off."""
    s = {"dy": [0.0, 0.0, 0.0], "cum": None, "stream": False}
    stop = 0.0
    for ev in events:
        if ev["t0"] > t:
            break
        if ev["kind"] == "move":
            p = smooth((t - ev["t0"]) / (ev["t1"] - ev["t0"]))
            if t < ev["t1"]:                                     # moving: the vial under the nozzle leaves
                s["dy"] = [stop - 2 * SLOT * p, (ev["stop"] - SLOT) * p, -SLOT * p]
                s["cum"] = s["cum"] if ev["exit_full"] else None
                return s
            stop = ev["stop"]                                     # arrived: parts snap back, new empty vial
            s["dy"], s["cum"] = [stop, 0.0, 0.0], None
        else:
            p = min((t - ev["t0"]) / (ev["t1"] - ev["t0"]), 1.0)
            s["cum"] = {k: ev["from"][k] + (ev["to"][k] - ev["from"][k]) * p for k in TANKS}
            s["stream"] = t < ev["t1"]
            s["tank"] = ev["tank"]
    return s


# ----------------------------------------------------------------------------------------------
# Server: the dashboard sends the queue and reads frames
# ----------------------------------------------------------------------------------------------
class Twin:
    def __init__(self, rules):
        self.rules = rules
        self.lock, self.cancel = threading.Lock(), threading.Event()
        self.latest, self.video = b"", None             # newest frame (jpeg), finished video path
        self.queue, self.status = None, {"state": "idle", "error": None}

    def update(self, **kw):
        with self.lock:
            self.status.update(kw)

    def serve(self, host, port):
        twin = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, body, ctype, code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with twin.lock:
                    status, latest, video = dict(twin.status), twin.latest, twin.video
                if self.path in ("/", "/status"):
                    return self.send(json.dumps(status).encode(), "application/json")
                if self.path.startswith("/latest.jpg") and latest:
                    return self.send(latest, "image/jpeg")
                if self.path == "/video" and status["state"] == "done" and video:
                    return self.send(Path(video).read_bytes(), "video/mp4")
                self.send_error(404)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/cancel":
                    twin.cancel.set()
                    return self.send(b"{}", "application/json")
                if self.path != "/queue":
                    return self.send_error(404)
                with twin.lock:
                    if twin.status["state"] in ("starting", "rendering"):
                        return self.send(json.dumps({"error": "the twin is already rendering a queue"}).encode(),
                                         "application/json", 409)
                    try:
                        form = json.loads(body)
                        queue = read_queue(form["vials"])
                        plan_run(queue, twin.rules, 0)                 # checks the queue and recipe rules now
                        twin.queue = (queue, form.get("seed"))
                    except Exception as e:
                        return self.send(json.dumps({"error": str(e)}).encode(), "application/json", 400)
                    twin.cancel.clear()
                    twin.latest, twin.video = b"", None
                    twin.status = {"state": "starting", "error": None}
                self.send(b"{}", "application/json")

        server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"twin listening on http://{host}:{port} (POST /queue, GET /status, /latest.jpg, /video)", flush=True)


def camera_look(img, t, n, cv2, np):
    """The real camera's look: exposure, slight blur, low noise, and a timestamp from the twin clock."""
    x = np.clip(LOOK["exposure_gain"] * 255.0 * (img.astype(np.float32) / 255.0) ** LOOK["exposure_gamma"], 0, 255)
    if LOOK["blur"]:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    x = np.clip(x + np.random.default_rng(n).normal(0, LOOK["noise_sigma"], x.shape), 0, 255).astype(np.uint8)
    secs = 36000 + t
    text = f"{time.strftime('%Y-%m-%d')} {int(secs) // 3600:02d}:{int(secs) // 60 % 60:02d}:{secs % 60:09.6f}"
    for color, thick in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
        cv2.putText(x, text, (10, 467), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)
    return x


def scene_key(s):
    """What a picture depends on; the same key = the same picture."""
    return (tuple(round(v, 4) for v in s["dy"]),
            None if s["cum"] is None else tuple(round(s["cum"][k], 2) for k in TANKS), s["stream"])


def run_queue(twin, queue, seed, rules, out_root, scene, cv2, np):
    """One queue: plan it, render every frame into video.mp4, then offer the video."""
    events, truth, info = plan_run(queue, rules, seed)
    out = Path(out_root) / f"twin_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "queue.csv", [{"vial": q["vial"], "recipe_id": q["recipe_id"], "t100_ml": q["T100"] or "",
                                   "t200_ml": q["T200"] or "", "t400_ml": q["T400"] or ""} for q in queue])
    write_csv(out / "truth.csv", truth)
    states = [state_at(n / FPS, events) for n in range(info["frames"])]
    render_at, last = [], None                               # which frames need a new picture
    for n, s in enumerate(states):
        if last is None or (scene_key(s) != last and n % RENDER_EVERY == 0):
            render_at.append(n)
            last = scene_key(s)
    info["renders"] = len(render_at)
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"run {out.name}: {info}", flush=True)
    twin.update(state="rendering", run=out.name, vials=len(queue), twin_seconds=info["twin_seconds"],
                renders_planned=len(render_at), renders_done=0, progress=0.0, eta_s=None)
    path = out / "video.mp4"
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    todo, img, done, t_start = set(render_at), None, 0, time.time()
    for n, s in enumerate(states):
        if twin.cancel.is_set():
            video.release()
            twin.update(state="cancelled")
            print(f"cancelled at frame {n}", flush=True)
            return
        if n in todo:
            img = scene.render(s, n)
            done += 1
            per = (time.time() - t_start) / done
            twin.update(renders_done=done, progress=round(done / len(render_at), 4),
                        eta_s=round(per * (len(render_at) - done)), s_per_render=round(per, 2))
        bgr = np.ascontiguousarray(camera_look(img, n / FPS, n, cv2, np)[..., ::-1])
        video.write(bgr)
        if n in todo:
            with twin.lock:
                twin.latest = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            if done % 25 == 0:
                print(f"render {done}/{len(render_at)} ({per:.2f} s each, about "
                      f"{per * (len(render_at) - done) / 60:.0f} min left)", flush=True)
    video.release()
    with twin.lock:
        twin.video = str(path)
        twin.status.update(state="done", progress=1.0, eta_s=0)
    print(f"done: {path} ({done} renders in {(time.time() - t_start) / 60:.1f} min)", flush=True)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------
# Pictures: Isaac Sim, or simple drawings for --preview
# ----------------------------------------------------------------------------------------------
class PreviewScene:
    """No Isaac Sim: a gray picture with a vial, liquid and carrier drawn in, to test the connection."""
    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np

    def render(self, s, n):
        cv2, np = self.cv2, self.np
        img = np.full((480, 640, 3), 200, np.uint8)
        img[60:200, 60:250] = 235                                          # white panel
        xs = [base + int(s["dy"][k] / SLOT * 85) for k, base in enumerate([310, 395, 480])]   # 85 px per slot
        for x in xs:
            cv2.rectangle(img, (x - 25, 205), (x + 25, 265), (90, 90, 90), -1)          # carrier
            cv2.rectangle(img, (x - 15, 280), (x + 15, 445), (225, 225, 225), 2)        # vial
        total = sum(s["cum"].values()) if s["cum"] else 0.0
        if total > 0.05:
            diffuse, _ = liquid_color(s["cum"]["T200"] / total, s["cum"]["T400"] / total)
            top = int(445 - 160 * total / FULL_ML)
            x = xs[0]
            cv2.rectangle(img, (x - 14, top), (x + 14, 444), [int(255 * c) for c in diffuse], -1)
        if s["stream"]:
            color, _ = liquid_color(float(s["tank"] == "T200"), float(s["tank"] == "T400"))
            cv2.line(img, (xs[0], 150), (xs[0], 300), [int(255 * c) for c in color], 3)
        return img


class IsaacScene:
    """The organizers' fill station in Isaac Sim, edited in memory only (session layer)."""
    def __init__(self, args, cam):
        rw, rh = cam["render_px"]
        from isaacsim import SimulationApp
        self.kit = SimulationApp({"headless": True, "renderer": args.renderer, "width": rw, "height": rh})
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdShade
        self.np, self.rep, self.Gf, self.Sdf, self.Usd = np, rep, Gf, Sdf, Usd
        self.subframes, self.rw, self.rh = args.subframes, rw, rh
        ctx = omni.usd.get_context()
        if not ctx.open_stage(args.stage):
            raise RuntimeError(f"could not open stage: {args.stage}")
        while ctx.get_stage_loading_status()[2] > 0:
            self.kit.update()
        for _ in range(10):
            self.kit.update()
        stage = ctx.get_stage()
        cam_path = next(p for p in [cam["prim_in_station_file"], cam["prim_in_csi_line"]]
                        if stage.GetPrimAtPath(p).IsValid())
        self.bottom = liquid_bottom(stage)
        tc = Usd.TimeCode(LAST_FRAME)
        root = stage.GetPrimAtPath(LIQUID).GetParent()
        # the moving parts and where they are at the end of the twin's animation (carrier 2 has arrived)
        self.parts = {k: [] for k in CARRIER_PARTS}
        for p in root.GetChildren():
            for k, prefixes in CARRIER_PARTS.items():
                if any(p.GetName().startswith(x) for x in prefixes):
                    a = p.GetAttribute("xformOp:translate")
                    if a:
                        self.parts[k].append((a, a.Get(tc)))
        mesh = stage.GetPrimAtPath(LIQUID_MESH)
        settled = {n: mesh.GetAttribute(n).Get(tc) for n in ["points", "normals", "extent"]}
        liquid_xf = stage.GetPrimAtPath(LIQUID)
        self.scale_attr = liquid_xf.GetAttribute("xformOp:scale")
        self.base_scale = self.scale_attr.Get(tc)
        self.liquid_vis = liquid_xf.GetAttribute("visibility")
        self.stream_vis = stage.GetPrimAtPath(STREAM).GetAttribute("visibility")
        self.neighbour_vis = root.GetChild("Fill_vial_neighbour_2").GetAttribute("visibility")
        material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        self.shader = next(UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim()) if UsdShade.Shader(p))

        stage.SetEditTarget(stage.GetSessionLayer())                  # every change below stays in memory
        self.tc = tc
        for n, v in settled.items():                                   # the settled liquid shape, always
            if v is not None:
                mesh.GetAttribute(n).Set(v, tc)
        self.neighbour_vis.Set("inherited", tc)
        # the twin's stream is always blue: give it its own material, colored by the tank that is pouring
        stream_mat = UsdShade.Material.Define(stage, "/TwinLive/StreamMaterial")
        self.stream_shader = UsdShade.Shader.Define(stage, "/TwinLive/StreamMaterial/Surface")
        self.stream_shader.CreateIdAttr("UsdPreviewSurface")
        self.stream_shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(ORGANIZERS_OPACITY)
        self.stream_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.1)
        stream_mat.CreateSurfaceOutput().ConnectToSource(self.stream_shader.ConnectableAPI(), "surface")
        stream_mesh = stage.GetPrimAtPath(STREAM + "/Fill_dispensing_stream")
        UsdShade.MaterialBindingAPI.Apply(stream_mesh).Bind(stream_mat)
        self.rp = rep.create.render_product(cam_path, (rw, rh))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach([self.rp])
        rep.orchestrator.set_capture_on_play(False)
        self.timeline = omni.timeline.get_timeline_interface()
        self.tcps = stage.GetTimeCodesPerSecond()
        self.kit.update()

    def render(self, s, n):
        Gf, tc = self.Gf, self.tc
        for k, parts in self.parts.items():                            # carriers along the track (y)
            for attr, base in parts:
                attr.Set(Gf.Vec3d(base[0], base[1] + s["dy"][k], base[2]), tc)
        liquid_t = next(a for a, _ in self.parts[0] if a.GetPrim().GetName() == "Fill_liquid")
        base_t = next(b for a, b in self.parts[0] if a.GetPrim().GetName() == "Fill_liquid")
        total = sum(s["cum"].values()) if s["cum"] else 0.0
        if total > 0.05:
            sz = total / FULL_ML                                       # height ~ volume, bottom kept in place
            self.scale_attr.Set(Gf.Vec3f(self.base_scale[0], self.base_scale[1], self.base_scale[2] * sz), tc)
            liquid_t.Set(Gf.Vec3d(base_t[0], base_t[1] + s["dy"][0],
                                  base_t[2] + self.base_scale[2] * self.bottom * (1 - sz)), tc)
            diffuse, opacity = liquid_color(s["cum"]["T200"] / total, s["cum"]["T400"] / total)
            self.shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
            self.shader.CreateInput("opacity", self.Sdf.ValueTypeNames.Float).Set(float(opacity))
            self.liquid_vis.Set("inherited", tc)
        else:
            self.liquid_vis.Set("invisible", tc)
        self.stream_vis.Set("inherited" if s["stream"] else "invisible", tc)
        if s["stream"]:                                                # pure orange, blue or clear
            color, _ = liquid_color(float(s["tank"] == "T200"), float(s["tank"] == "T400"))
            self.stream_shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        a, b = STREAM_FRAMES                                           # the stream's own animation moves
        self.timeline.set_current_time((a + n % (b - a)) / self.tcps)
        self.kit.update()
        return capture(self.rep, self.ann, self.rh, self.rw, self.subframes, self.np)

    def idle(self):
        self.kit.update()


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", default=DEFAULT_TWIN, help="the organizers' twin folder (has cameras/ and stations/)")
    ap.add_argument("--stage", default=None, help="default: <twin>/stations/fill_station/fill_station.usda")
    ap.add_argument("--rules", default=os.path.join(SCRIPT_DIR, "recipe_rules.json"))
    ap.add_argument("--out-dir", default="twin_live_runs", help="where each run's video and truth file go")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=20700, help="YAAD team ports: 20700-20799")
    ap.add_argument("--seed", type=int, default=None, help="fixed seed = the same twin decisions every run")
    ap.add_argument("--subframes", type=int, default=32, help="render quality (32 = the training pictures)")
    ap.add_argument("--renderer", default="PathTracing", help="PathTracing (as training) or RayTracedLighting")
    ap.add_argument("--preview", action="store_true", help="drawn pictures instead of Isaac Sim (connection test)")
    ap.add_argument("--dry-run", action="store_true", help="with --queue: print the plan and truth, no pictures")
    ap.add_argument("--queue", help="--dry-run: recipe queue CSV")
    args = ap.parse_args()
    rules, _ = load_rules(args.rules)

    if args.dry_run:
        if not args.queue:
            ap.error("--dry-run needs --queue <recipe queue CSV>")
        events, truth, info = plan_run(read_queue(args.queue), rules, args.seed if args.seed is not None else 0)
        out = Path(args.out_dir) / "dry_run"
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "truth.csv", truth)
        print(json.dumps(info))
        for ev in events:
            extra = f"{ev['tank']} -> {sum(ev['to'].values()):.1f} mL" if ev["kind"] == "pour" else \
                ("full vial leaves" if ev["exit_full"] else "empty carrier leaves")
            print(f"  {ev['t0']:7.1f}-{ev['t1']:7.1f} s  {ev['kind']:4s}  {extra}")
        for r in truth:
            print(f"  vial {r['vial']:>2} recipe {r['recipe_id']:>3} after {r['previous_recipe']:>3}: leftover "
                  f"{r['residue_ml']:5.1f} mL, true mix {100 * r['actual_orange']:4.1f}% orange "
                  f"{100 * r['actual_blue']:4.1f}% blue, carryover {r['carryover']}")
        print(f"wrote {out / 'truth.csv'}")
        return

    import cv2
    import numpy as np
    args.twin = str(Path(args.twin).expanduser().resolve())
    args.stage = args.stage or os.path.join(args.twin, "stations", "fill_station", "fill_station.usda")
    twin = Twin(rules)
    twin.serve(args.host, args.port)
    if args.preview:
        scene = PreviewScene(cv2, np)
    else:
        scene = IsaacScene(args, json.load(open(os.path.join(args.twin, "cameras", "FillCam.json"))))
    print("ready: waiting for a recipe queue", flush=True)
    try:
        while True:
            with twin.lock:
                job, twin.queue = twin.queue, None
            if job is None:
                if hasattr(scene, "idle"):
                    scene.idle()
                time.sleep(0.1)
                continue
            queue, seed = job
            try:
                run_queue(twin, queue, seed if seed is not None else (args.seed if args.seed is not None
                                                                      else random.randrange(10 ** 6)),
                          rules, args.out_dir, scene, cv2, np)
            except Exception as e:                                         # shown in /status
                import traceback
                traceback.print_exc()
                twin.update(state="error", error=str(e))
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(scene, "kit"):
            scene.kit.close()


if __name__ == "__main__":
    main()

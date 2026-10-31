"""Generate labeled training pictures from the fill-station digital twin: one picture after each pour.

    # Plan only (no Isaac Sim; runs on a laptop). Checks recipes against the rules and writes labels.csv:
    python3 twin_tools/render_dataset.py --dry-run --count 20 --out-dir dataset_plan   (from version_1/)

    # Render (Isaac Sim 5.1 on YAAD): upload this file + recipe_rules.json into the workspace twin's tools/,
    # then from the twin folder:
    /isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 20 --variations 1 --out-dir /job/out
    # -> out/images/clean/*.png, out/images/camera/*.jpg, out/masks/*.png, out/labels.csv (one row per picture)

What one "sample" is: one vial. It gets a recipe (real 5-14, or a random new one that passes
tools/recipe_rules.json) and a previous vial. Every vial starts with `residue_ml` of the previous
vial's mix (the shared mixing path the organizers describe), and the same amount of its own mix stays
behind in the path (taken proportionally from every pour), so the total matches the recipe. The leftover is
capped at --max-residue-share of the vial. If the previous vial was the same recipe the sample is clean
(the finished vial matches the recipe, but early-pour pictures are a blend, like real vial 11); if it was
a different recipe the sample is a carryover sample.
Each vial variation gets one lighting setting and one camera look, shared by all its pours (--light-mode
random for training; neutral/dim/bright/warm/cool fixed). The camera never moves. Every render is saved
TWICE (--styles): "clean" (straight from Isaac Sim, lossless PNG, like fillcam_frame686.png) and "camera"
(the same picture with the real-camera look), so the model sees both kinds of rendering.
The liquid surface is always the settled finished-vial shape scaled to the mL so far (no splashing tops),
and the liquid material keeps the organizers' opacity 0.6 (--opacity-mode fixed).
For each pour (dispense order T200 -> T400 -> T100) the script sets the liquid height for the mL so far,
sets the liquid color for the mix so far, hides the pouring stream, renders FillCam, makes the image look
like the real camera, measures the white panel's color, and writes one row to labels.csv.

Nothing in the twin files is changed: all edits go to the session layer (in memory) and are never saved.
APPROXIMATE (calibrate before trusting): liquid colors, opacity, residue volume.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The organizers' twin is NOT copied into version_1; point to it with --twin.
# Default: the untouched original next to version_1. On YAAD the job passes --twin /job/in.
DEFAULT_TWIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "csi_digital_twin_v1"))
TANKS = ["T200", "T400", "T100"]                                       # orange, blue, clear
LIQUID = "/FillStation/FillStation_LocalRoot/Fill_liquid"
LIQUID_MESH = LIQUID + "/Fill_liquid"
STREAM = "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream"
FULL_ML = 70.0          # the twin's animation is recipe 7: 70 mL at frame 686
FIRST_FRAME, LAST_FRAME = 21, 686   # liquid is visible from frame 21

# Liquid color model (APPROXIMATE). Dye absorption fitted to the 20 clean real vials of video 5
# (changeover-watch, Beer-Lambert). The twin material uses diffuse = BRIGHTNESS * exp(-(o*A_ORANGE + b*A_BLUE)),
# where o, b are the orange and blue fractions. BRIGHTNESS matches the green channel of the organizers'
# blue for recipe 7 (diffuseColor 0.006, 0.2, 0.3).
A_ORANGE = (0.0, 4.503, 7.878)
A_BLUE = (6.36, 1.81, 0.508)
BRIGHTNESS = 0.73
COLOR_MODEL = "approx_v1"

# Class ids written into the mask PNGs (twin's own synthetic_class_id where it has one).
CLASS_IDS = {"vial": 1, "carrier": 4, "track": 5, "nozzle": 6, "background": 7, "label": 8,
             "liquid": 9, "stream": 10}

COLUMNS = ["render_style", "image", "mask", "sample_id", "variation", "recipe_id", "recipe_source", "t200_ml", "t400_ml",
           "t100_ml", "total_ml", "dispense_order", "previous_recipe", "carryover", "residue_ml", "stage",
           "stage_tank", "stages_total", "cum_ml", "cum_orange_ml", "cum_blue_ml", "cum_clear_ml",
           "orange_frac", "blue_frac", "settled", "frame", "liquid_scale_z", "diffuse_r", "diffuse_g",
           "diffuse_b", "opacity", "light_mode", "light_gain", "light_tint", "exposure_gamma", "exposure_gain",
           "noise_sigma", "blur", "dye_orange_scale", "dye_blue_scale", "dye_brightness", "shift_x", "shift_y",
           "panel_r", "panel_g", "panel_b", "vial_box", "color_model", "seed"]

# White-panel patch left of the vial (x0, y0, x1, y1). Clean white in both twin and real vial_fill frames,
# away from the timestamp. Its measured color says how bright/warm the light is in that picture.
PANEL_ROI = (60, 60, 250, 200)

# Lighting per vial. "random" = domain randomization for training; fixed modes = stress-test datasets.
LIGHT_MODES = {"random": None, "neutral": (1.0, 0.0), "dim": (0.6, 0.0), "bright": (1.4, 0.0),
               "warm": (1.0, 0.08), "cool": (1.0, -0.08)}

# Two renderings of every picture, from ONE render (same vial, pour, lighting):
#   clean  = the render as it comes out of Isaac Sim, saved lossless (like the organizers' fillcam_frame686.png)
#   camera = the same render with the real-camera look (darker, noise, blur, timestamp), saved as JPEG
STYLES = {"clean": ".png", "camera": ".jpg"}
ORGANIZERS_OPACITY = 0.6     # the organizers' liquid material (USD_blue_liquid_approximation) uses 0.6


# ----------------------------------------------------------------------------------------------
# Recipe logic (pure Python, no Isaac Sim)
# ----------------------------------------------------------------------------------------------
def load_rules(path):
    rules = json.load(open(path))
    real = [{"recipe_id": str(r["recipe_id"]), "source": "real", **{t: float(r.get(t, 0)) for t in TANKS}}
            for r in rules["real_recipes"]]
    for r in real:
        problems = validate(r, rules)
        if problems:
            raise ValueError(f"real recipe {r['recipe_id']} breaks the rules: {problems}")
    return rules, real


def validate(recipe, rules):
    """List of rule violations (empty = OK)."""
    p = []
    for t in rules["forbidden_tanks"]:
        if recipe.get(t, 0):
            p.append(f"uses forbidden tank {t}")
    used = [t for t in TANKS if recipe[t] > 0]
    if not any(rules["tanks"][t]["role"] == "dye" for t in used):
        p.append("no dye")
    if sum(recipe[t] for t in TANKS) > rules["max_total_ml"]:
        p.append(f"total over {rules['max_total_ml']} mL")
    for t in used:
        if recipe[t] < rules["min_ml_per_used_tank"]:
            p.append(f"{t} under {rules['min_ml_per_used_tank']} mL")
        if recipe[t] > rules["max_ml_per_tank"][t]:
            p.append(f"{t} over {rules['max_ml_per_tank'][t]} mL")
    return p


def random_recipe(rng, rules, n):
    """A new recipe that passes the rules (volumes in 5 mL steps)."""
    while True:
        r = {"recipe_id": f"N{n:03d}", "source": "random"}
        for t in TANKS:
            use = rng.random() < (0.8 if t != "T100" else 0.6)
            r[t] = float(rng.randrange(10, int(rules["max_ml_per_tank"][t]) + 1, 5)) if use else 0.0
        if not validate(r, rules):
            return r


def stages(cur, prev, residue_ml, rules):
    """Cumulative mL per tank after each pour. The shared mixing path is treated as well mixed (the docs:
    'residual liquid from the previous recipe's mixture'): residue_ml of the previous vial's mix comes out
    first, and the same amount of THIS recipe's mix stays behind, taken proportionally from every pour.
    So a clean vial ends exactly at its recipe, no dye is lost, and a carryover vial ends as
    (residue/total) old mix + the rest new recipe, the same model the changeover-watch explainer fits."""
    order = [t for t in rules["dispense_order"] if cur[t] > 0]
    keep = 1.0 - residue_ml / sum(cur[t] for t in TANKS)
    deliver = {t: cur[t] * keep for t in order}
    prev_total = sum(prev[t] for t in TANKS)
    cum = {t: residue_ml * prev[t] / prev_total for t in TANKS}
    out = []
    for k, t in enumerate(order):
        cum[t] += deliver[t]
        out.append({"stage": k + 1, "tank": t, "cum": dict(cum)})
    return out


def plan_samples(rules, real, args, rng):
    samples, n_random = [], 0
    for i in range(args.count):
        if rng.random() < args.real_share:
            cur = rng.choice(real)
        else:
            n_random += 1
            cur = random_recipe(rng, rules, n_random)
        carry = rng.random() < args.carryover_share
        prev = cur
        while carry and prev is cur or (carry and all(prev[t] == cur[t] for t in TANKS)):
            if rng.random() < 0.5:
                prev = rng.choice(real)
            else:
                n_random += 1
                prev = random_recipe(rng, rules, n_random)
        total = sum(cur[t] for t in TANKS)
        residue = min(round(rng.uniform(*args.residue_ml) * 2) / 2, args.max_residue_share * total)
        samples.append({"sample_id": i + 1, "cur": cur, "prev": prev, "carryover": int(carry),
                        "residue_ml": residue, "stages": stages(cur, prev, residue, rules)})
    return samples


def liquid_color(orange_frac, blue_frac, opacity_mode="fixed", dye=None):
    """dye = this vial's random scales from vial_conditions (--dye-jitter); None = the plain color model."""
    so, sb, sbr = (dye["_o"], dye["_b"], dye["_br"]) if dye else ((1, 1, 1), (1, 1, 1), 1.0)
    diffuse = [min(1.0, max(0.0, BRIGHTNESS * sbr * math.exp(-(orange_frac * ao * ko + blue_frac * ab * kb))))
               for ao, ab, ko, kb in zip(A_ORANGE, A_BLUE, so, sb)]
    if opacity_mode == "fixed":                    # same as the organizers' material (clean look)
        opacity = ORGANIZERS_OPACITY
    else:                                          # "by-dye": paler mixes more see-through (earlier versions)
        opacity = 0.15 + 0.45 * min(1.0, (orange_frac + blue_frac) / (50 / 70))
    return diffuse, opacity


# ----------------------------------------------------------------------------------------------
# Fill height: calm surface. Every picture uses the SETTLED finished-vial shape (frame 686, 70 mL) and
# scales it vertically to the mL so far, keeping the bottom in place. (Earlier versions borrowed
# mid-pour animation frames, whose surface was still splashing: the "foamy tops".)
# ----------------------------------------------------------------------------------------------
def liquid_bottom(stage):
    z = [p[2] for p in stage.GetPrimAtPath(LIQUID_MESH).GetAttribute("points").Get(LAST_FRAME)]
    return min(z)


def calm_height(cum_ml):
    """Frame and vertical scale for cum_ml (height ~ volume in the cylinder)."""
    return LAST_FRAME, cum_ml / FULL_ML


def vial_conditions(args, sample_id, v):
    """Lighting and camera look for one vial variation: the same for every pour of that vial."""
    rng = random.Random(f"{args.seed}-s{sample_id}-v{v}")
    if args.light_mode == "random":
        gain, tint = rng.uniform(0.6, 1.4), rng.uniform(-0.08, 0.08)
        mults = [rng.uniform(0.85, 1.15) for _ in range(16)]          # small per-light differences
    else:
        gain, tint = LIGHT_MODES[args.light_mode]
        mults = [1.0] * 16
    cond = {"light_mode": args.light_mode, "light_gain": round(gain, 3), "light_tint": round(tint, 3),
            "_light_mults": mults, "exposure_gamma": round(rng.uniform(1.1, 1.5), 3),
            "exposure_gain": round(rng.uniform(0.75, 1.0), 3), "noise_sigma": round(rng.uniform(1.5, 4.0), 2),
            "blur": int(rng.random() < 0.5)}
    # Dye shade and vial position: own random stream, so runs without them match earlier runs exactly.
    rng2 = random.Random(f"{args.seed}-s{sample_id}-v{v}-dye-pos")
    j = args.dye_jitter                        # each dye's absorption per channel x U(1-j, 1+j); brightness x U(1-j/2, 1+j/2)
    so = tuple(round(rng2.uniform(1 - j, 1 + j), 3) for _ in range(3))
    sb = tuple(round(rng2.uniform(1 - j, 1 + j), 3) for _ in range(3))
    sbr = round(rng2.uniform(1 - j / 2, 1 + j / 2), 3)
    p = args.pos_jitter                        # whole picture shifted by up to p pixels (vial stop position)
    cond.update(_o=so, _b=sb, _br=sbr, dye_orange_scale=" ".join(map(str, so)),
                dye_blue_scale=" ".join(map(str, sb)), dye_brightness=sbr,
                shift_x=rng2.randint(-p, p), shift_y=rng2.randint(-p, p))
    return cond


def rows_for(samples, args):
    """Expand samples into one row per variation x pour. A variation = the whole vial under one lighting
    and camera look; every pour of that vial shares it (render-only fields are filled in later)."""
    rows = []
    for s in samples:
        cur, prev = s["cur"], s["prev"]
        for v in range(1, args.variations + 1):
            cond = vial_conditions(args, s["sample_id"], v)
            for st in s["stages"]:
                cum = st["cum"]
                total = sum(cum.values())
                o, b = cum["T200"] / total, cum["T400"] / total
                diffuse, opacity = liquid_color(o, b, args.opacity_mode, cond)
                frame, scale_z = calm_height(total)
                name = f"s{s['sample_id']:04d}_p{st['stage']}_v{v}"
                rows.append({**cond,
                    "_name": name, "mask": f"masks/{name}.png", "sample_id": s["sample_id"],
                    "variation": v, "recipe_id": cur["recipe_id"], "recipe_source": cur["source"],
                    "t200_ml": cur["T200"], "t400_ml": cur["T400"], "t100_ml": cur["T100"],
                    "total_ml": sum(cur[t] for t in TANKS),
                    "dispense_order": ">".join(t for t in ["T200", "T400", "T100"] if cur[t] > 0),
                    "previous_recipe": prev["recipe_id"], "carryover": s["carryover"],
                    "residue_ml": s["residue_ml"], "stage": st["stage"], "stage_tank": st["tank"],
                    "stages_total": len(s["stages"]), "cum_ml": round(total, 2),
                    "cum_orange_ml": round(cum["T200"], 2), "cum_blue_ml": round(cum["T400"], 2),
                    "cum_clear_ml": round(cum["T100"], 2), "orange_frac": round(o, 4), "blue_frac": round(b, 4),
                    "settled": 1, "frame": frame, "liquid_scale_z": round(scale_z, 4),
                    "diffuse_r": round(diffuse[0], 4), "diffuse_g": round(diffuse[1], 4),
                    "diffuse_b": round(diffuse[2], 4), "opacity": round(opacity, 3),
                    "color_model": COLOR_MODEL, "seed": f"{args.seed}-{name}"})
    return rows


# ----------------------------------------------------------------------------------------------
# Rendering (Isaac Sim 5.1)
# ----------------------------------------------------------------------------------------------
def render(rows, args, cam):
    rw, rh = cam["render_px"]
    from isaacsim import SimulationApp
    kit = SimulationApp({"headless": True, "renderer": args.renderer, "width": rw, "height": rh,
                         "extra_args": args.kit_arg})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdShade

        ctx = omni.usd.get_context()
        if not ctx.open_stage(args.stage):
            raise RuntimeError(f"could not open stage: {args.stage}")
        while ctx.get_stage_loading_status()[2] > 0:
            kit.update()
        for _ in range(10):
            kit.update()
        stage = ctx.get_stage()
        cam_path = next((p for p in [cam["prim_in_station_file"], cam["prim_in_csi_line"]]
                         if stage.GetPrimAtPath(p).IsValid()), None)
        if cam_path is None:
            raise RuntimeError("FillCam not found; use the fill-station file as --stage")
        bottom = liquid_bottom(stage)

        stage.SetEditTarget(stage.GetSessionLayer())     # every change below stays in memory only
        liquid_xf = stage.GetPrimAtPath(LIQUID)
        stream = stage.GetPrimAtPath(STREAM)
        material = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(LIQUID_MESH)).ComputeBoundMaterial()[0]
        shader = next(UsdShade.Shader(p) for p in Usd.PrimRange(material.GetPrim()) if UsdShade.Shader(p))
        lights = [p for p in stage.Traverse() if p.GetTypeName().endswith("Light")]
        light_base = {}
        for p in lights:
            ia = p.GetAttribute("inputs:intensity") or p.GetAttribute("intensity")
            ca = p.GetAttribute("inputs:color") or p.GetAttribute("color")
            light_base[p.GetPath()] = (ia, ia.Get() if ia else None, ca, ca.Get() if ca else Gf.Vec3f(1, 1, 1))
        scale_attr = liquid_xf.GetAttribute("xformOp:scale")
        base_scale = scale_attr.Get(Usd.TimeCode(LAST_FRAME))   # authored once in the twin; read before edits
        translate_attr = liquid_xf.GetAttribute("xformOp:translate")
        base_translate = translate_attr.Get(Usd.TimeCode(LAST_FRAME))

        labels_on = not args.no_labels and apply_labels(stage)
        rp = rep.create.render_product(cam_path, (rw, rh))
        ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        ann_rgb.attach([rp])
        if labels_on:
            ann_box = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            ann_seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
            ann_box.attach([rp])
            ann_seg.attach([rp])
        rep.orchestrator.set_capture_on_play(False)
        kit.update()

        out = Path(args.out_dir)
        for style in args.styles:
            (out / "images" / style).mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        tl = omni.timeline.get_timeline_interface()
        tcps = stage.GetTimeCodesPerSecond()
        with open(out / "labels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            for i, r in enumerate(rows):
                f = int(r["frame"])
                tc = Usd.TimeCode(f)
                tl.set_current_time(f / tcps)
                # liquid height (settled shape scaled, bottom kept in place), visibility, color
                sz = r["liquid_scale_z"]
                scale_attr.Set(Gf.Vec3f(base_scale[0], base_scale[1], base_scale[2] * sz), tc)
                translate_attr.Set(Gf.Vec3d(base_translate[0], base_translate[1],
                                            base_translate[2] + base_scale[2] * bottom * (1 - sz)), tc)
                stream.GetAttribute("visibility").Set("invisible", tc)
                liquid_xf.GetAttribute("visibility").Set("inherited", tc)
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(r["diffuse_r"], r["diffuse_g"], r["diffuse_b"]))
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(r["opacity"]))
                # lights: this vial's brightness + warm/cool tint (same for all its pours)
                tint = r["light_tint"]
                for k, (ia, i0, ca, c0) in enumerate(light_base.values()):
                    if ia is not None and i0 is not None:
                        ia.Set(float(i0) * r["light_gain"] * r["_light_mults"][k % 16])
                    if ca is not None:
                        ca.Set(Gf.Vec3f(c0[0] * (1 + tint), c0[1], c0[2] * (1 - tint)))
                kit.update()

                img = capture(rep, ann_rgb, rh, rw, args.subframes, np)
                boxes = parse_boxes(ann_box.get_data(), rw) if labels_on else {}
                mask = semantic_mask(ann_seg.get_data(), np) if labels_on else None
                dx, dy = r["shift_x"], r["shift_y"]
                if dx or dy:                                   # vial stop position (--pos-jitter)
                    img = shift(img, dx, dy, cv2, np, cv2.INTER_LINEAR)
                    mask = None if mask is None else shift(mask, dx, dy, cv2, np, cv2.INTER_NEAREST)
                    boxes = {k: " ".join(str(int(c) + (dx, dy)[n % 2]) for n, c in enumerate(b.split()))
                             for k, b in boxes.items()}
                if mask is not None:
                    cv2.imwrite(str(out / r["mask"]), mask)
                r["vial_box"] = boxes.get("vial", "")
                if mask is None:
                    r["mask"] = ""
                for style in args.styles:                      # one render -> clean + camera pictures
                    pic = img if style == "clean" else realism(img, r, i, cv2, np)
                    x0, y0, x1, y1 = PANEL_ROI                 # what a real camera could measure too
                    row = {**r, "render_style": style, "image": f"images/{style}/{r['_name']}{STYLES[style]}"}
                    row["panel_r"], row["panel_g"], row["panel_b"] = [
                        round(float(c), 1) for c in pic[y0:y1, x0:x1].reshape(-1, 3).mean(0)]
                    if style == "clean":
                        cv2.imwrite(str(out / row["image"]), pic[..., ::-1])
                    else:
                        cv2.imwrite(str(out / row["image"]), pic[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow(row)
                fh.flush()
                print(f"[{i + 1}/{len(rows)}] {r['_name']} ({'+'.join(args.styles)}) recipe {r['recipe_id']} pour "
                      f"{r['stage']}/{r['stages_total']} {r['cum_ml']} mL carryover={r['carryover']}", flush=True)
        print("wrote", out / "labels.csv", flush=True)
    except BaseException:
        traceback.print_exc()
        kit.close(exit_code=1)
        raise
    kit.close()


def capture(rep, ann, rh, rw, subframes, np):
    """Step the renderer until the RGB annotator returns a real frame (first frames can come back empty)."""
    for attempt in range(40):
        rep.orchestrator.step(rt_subframes=subframes, delta_time=0.0)
        if attempt == 0:
            continue
        data = np.asarray(ann.get_data())
        if data.ndim == 3 and data.shape[:2] == (rh, rw) and data.shape[2] >= 3:
            return data[..., :3].copy()
    raise RuntimeError("renderer returned no image after 40 steps; check GPU/RTX startup errors")


def apply_labels(stage):
    """Turn the twin's userProperties:synthetic_class tags into Replicator labels; tag liquid and stream."""
    tagged = []
    for p in stage.Traverse():
        a = p.GetAttribute("userProperties:synthetic_class")
        cls = a.Get() if a and a.IsValid() else None
        if str(p.GetPath()) == LIQUID:
            cls = "liquid"
        elif str(p.GetPath()) == STREAM:
            cls = "stream"
        if cls:
            tagged.append((p, cls))
    try:
        try:
            from isaacsim.core.utils.semantics import add_labels
            for p, c in tagged:
                add_labels(p, labels=[c], instance_name="class")
        except ImportError:
            from isaacsim.core.utils.semantics import add_update_semantics
            for p, c in tagged:
                add_update_semantics(p, c)
        print(f"labels: tagged {len(tagged)} parts", flush=True)
        return True
    except Exception as e:  # labeling API differs between Isaac Sim versions: keep rendering without labels
        print(f"WARNING: could not add labels ({e}); rendering pictures without masks/boxes", flush=True)
        return False


def _label_of(info, sid):
    lab = info.get("idToLabels", {}).get(str(sid), "")
    return lab.get("class", "") if isinstance(lab, dict) else str(lab)


def parse_boxes(d, rw):
    """'x0 y0 x1 y1' for the vial under the nozzle (closest to image center). No liquid box (decision:
    not needed now; the see-through liquid is left out of Replicator's labels anyway)."""
    out, best = {}, None
    for row in d.get("data", []):
        cls = _label_of(d.get("info", {}), row["semanticId"])
        box = f"{int(row['x_min'])} {int(row['y_min'])} {int(row['x_max'])} {int(row['y_max'])}"
        if cls == "vial":
            dist = abs((row["x_min"] + row["x_max"]) / 2 - rw / 2)
            if best is None or dist < best:
                best, out["vial"] = dist, box
    return out


def semantic_mask(d, np):
    ids = np.asarray(d["data"])
    mask = np.zeros(ids.shape[:2], np.uint8)
    for sid in np.unique(ids):
        mask[ids == sid] = CLASS_IDS.get(_label_of(d.get("info", {}), sid), 0)
    return mask


def shift(img, dx, dy, cv2, np, interp):
    """Move the picture dx, dy pixels (edges repeat), like the vial stopping a little off its usual spot."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=interp,
                          borderMode=cv2.BORDER_REPLICATE)


def realism(img, r, i, cv2, np):
    """Make the render look like the real vial_fill camera. Exposure, brightness, blur and noise level come
    from the vial (same for all its pours); only the noise grain itself differs per picture."""
    x = np.clip(r["exposure_gain"] * 255.0 * (img.astype(np.float32) / 255.0) ** r["exposure_gamma"], 0, 255)
    if r["blur"]:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    grain = random.Random(r["seed"])                     # per-picture grain
    x = np.clip(x + np.random.default_rng(grain.randrange(2 ** 31)).normal(0, r["noise_sigma"], x.shape),
                0, 255).astype(np.uint8)
    secs = 71229 + 2 * i
    text = f"2026-09-23 {secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}.{grain.randrange(10 ** 6):06d}"
    for color, thick in [((0, 0, 0), 3), ((255, 255, 255), 1)]:
        cv2.putText(x, text, (10, 467), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)
    return x


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", default=DEFAULT_TWIN, help="the organizers' twin folder (has cameras/ and stations/)")
    ap.add_argument("--stage", default=None, help="default: <twin>/stations/fill_station/fill_station.usda")
    ap.add_argument("--rules", default=os.path.join(SCRIPT_DIR, "recipe_rules.json"))
    ap.add_argument("--out-dir", default="dataset_out")
    ap.add_argument("--count", type=int, default=20, help="number of vials (samples)")
    ap.add_argument("--variations", type=int, default=2,
                    help="versions of each vial, each under a new lighting and camera look (all pours together)")
    ap.add_argument("--real-share", type=float, default=0.5, help="share of samples using real recipes 5-14")
    ap.add_argument("--carryover-share", type=float, default=0.5, help="share of samples after a recipe change")
    ap.add_argument("--residue-ml", type=float, nargs=2, default=[5.0, 30.0],
                    help="range of previous-vial liquid that comes out first (APPROXIMATE)")
    ap.add_argument("--max-residue-share", type=float, default=0.5,
                    help="leftover is capped at this share of the vial (real carryover fits: ~40-60%%)")
    ap.add_argument("--light-mode", choices=list(LIGHT_MODES), default="random",
                    help="random per vial (training) or a fixed dim/bright/warm/cool setting (stress tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subframes", type=int, default=32, help="path-tracing samples per image (quality)")
    ap.add_argument("--renderer", default="PathTracing", help="PathTracing or RayTracedLighting")
    ap.add_argument("--no-labels", action="store_true", help="skip masks and boxes")
    ap.add_argument("--styles", default="clean,camera",
                    help="which renderings to save per picture: clean (like fillcam_frame686.png), camera, or both")
    ap.add_argument("--opacity-mode", choices=["fixed", "by-dye"], default="fixed",
                    help="fixed = organizers' 0.6 (clean look); by-dye = paler mixes more see-through (old)")
    ap.add_argument("--dye-jitter", type=float, default=0.0,
                    help="random dye shade per vial: absorption x U(1-j, 1+j) per channel (e.g. 0.2); 0 = off")
    ap.add_argument("--pos-jitter", type=int, default=0,
                    help="random picture shift per vial, up to this many pixels (e.g. 4); 0 = off")
    ap.add_argument("--dry-run", action="store_true", help="plan recipes and write labels.csv; no Isaac Sim")
    ap.add_argument("--kit-arg", action="append", default=[], help="extra Isaac Sim argument (repeatable)")
    args = ap.parse_args()
    args.twin = str(Path(args.twin).expanduser().resolve())
    if args.stage is None:
        args.stage = os.path.join(args.twin, "stations", "fill_station", "fill_station.usda")
    args.stage = str(Path(args.stage).expanduser().resolve())

    rules, real = load_rules(args.rules)
    rng = random.Random(args.seed)
    samples = plan_samples(rules, real, args, rng)
    args.styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    bad = [x for x in args.styles if x not in STYLES]
    if bad or not args.styles:
        ap.error(f"--styles must be from {list(STYLES)}")
    rows = rows_for(samples, args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "color_model": COLOR_MODEL, "A_ORANGE": A_ORANGE, "A_BLUE": A_BLUE,
               "BRIGHTNESS": BRIGHTNESS, "samples": len(samples), "renders": len(rows),
               "pictures": len(rows) * len(args.styles)},
              open(out / "run_info.json", "w"), indent=2)
    n_carry = sum(s["carryover"] for s in samples)
    n_real = sum(s["cur"]["source"] == "real" for s in samples)
    print(f"{len(samples)} vials ({n_real} real recipes, {len(samples) - n_real} random), {n_carry} carryover, "
          f"{len(rows)} renders x {len(args.styles)} styles = {len(rows) * len(args.styles)} pictures planned")
    if args.dry_run:
        with open(out / "labels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                for style in args.styles:
                    w.writerow({**r, "render_style": style,
                                "image": f"images/{style}/{r['_name']}{STYLES[style]}"})
        print("dry run: wrote", out / "labels.csv")
        return
    cam = json.load(open(os.path.join(args.twin, "cameras", "FillCam.json")))
    render(rows, args, cam)


if __name__ == "__main__":
    main()

"""Run the chosen model (model 2: image + recipe, trained on the twin) on a real fill-camera video.

    # from version_1/ (needs .venv: numpy, pandas, opencv, torch, torchvision):
    .venv/bin/python model/real_check.py --video <video.mp4> --recipes <recipe queue CSV>
    # final validation on video 5:
    .venv/bin/python model/real_check.py \
        --video ../csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4 \
        --recipes ../csi-vial-fill-dataset/vial_fill/labels/video5_vial_recipes.csv
    # -> model/results/real_<video name>/vials.csv, metrics.json, contact_sheet.jpg, snapshots/

Recipe queue (the assumed runtime input): one row per vial, in order, with CSI's columns
`vial, recipe_id, t100_ml, t200_ml, t400_ml` (blank = tank not used). If it also has CSI's time columns
`start_s, end_s` (the dispense window in video seconds), each vial is matched to the detected vial window that
overlaps it most, and the run is SCORED. Without them, vials are matched in order (the counts must agree) and
the script only reports each vial's estimate and flag.

Steps (nothing is tuned on the video being checked):
  1. Find vials (settings chosen by looking at videos 1-4). 5 frames per second. A vial's time under the nozzle
     lies between two big motions of the carrier (the next vial arriving). A pour is a stretch of motion inside
     the vial; back-to-back tanks with no pause count as one pour.
  2. Final snapshot FINAL_BEFORE_S before the vial leaves, judged as the last pour (a final-vial check).
  3. Model 2 estimates the orange and blue shares from the snapshot + recipe inputs (computed as for the twin).
  4. Flag when the estimate is more than the model's threshold (from twin validation) away from the expected mix
     of a clean vial. No calibration.
Scoring ground truth (an assumption, from CSI's run plan of 3 vials per recipe): the first vial after a recipe
change is a carryover vial, the others are clean. The first vial in the queue is not scored (its previous recipe
is unknown). Video 5 is the held-out validation set: run it once, at the end.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))
from build_table import NOMINAL_RESIDUE_ML, clean_vial_shares, expected_shares        # noqa: E402
from train_cnn import (CENTER, LOAD, PANEL_ROI, RECIPE_COLS, RECIPE_SCALE, build_net,  # noqa: E402
                       network_input, scores)

CHECKPOINT = MODEL_DIR / "results" / "model2" / "out" / "model.pt"
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
    smooth = pd.Series(mv).rolling(3, center=True, min_periods=1).median().values
    vials = []
    for (_, a), (b, _) in zip(edges[:-1], edges[1:]):
        c0, c1 = a + 0.5, b - 0.5
        m = (t >= c0) & (t <= c1)
        if c1 - c0 < 3 or not m.any():
            continue
        pours = []
        for tk, on in zip(t[m], smooth[m] > POUR_MOTION):
            if on:
                if pours and tk - pours[-1][1] <= POUR_GAP_S + dt:
                    pours[-1][1] = tk
                else:
                    pours.append([tk, tk])
        pours = [p for p in pours if p[1] - p[0] >= POUR_MIN_S]
        final = c1 - FINAL_BEFORE_S
        if not pours and (c1 - c0 < 10 or sat[np.argmin(np.abs(t - final))] < LIQUID_SAT):
            continue                                             # empty, or cut off by the end of the video
        vials.append({"start_s": round(c0, 1), "end_s": round(c1, 1), "pours_seen": len(pours),
                      "final_s": round(final, 2)})
    return vials


def frame_at(cap, seconds):
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, f = cap.read()
    return f if ok else None


def match_queue(queue, detected):
    """Recipe-queue row index -> detected vial. By time overlap if the queue has start_s/end_s, else in order."""
    if {"start_s", "end_s"} <= set(queue.columns):
        match = {}
        for i, q in enumerate(queue.itertuples()):
            best, j = max((min(d["end_s"], q.end_s) - max(d["start_s"], q.start_s), j) for j, d in enumerate(detected))
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


def estimate(frames, rec):
    import torch
    import torchvision
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    net, _ = build_net(torch, torchvision, pretrained=False, n_recipe=len(ckpt["recipe_cols"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    x0, y0, x1, y1 = LOAD
    px0, py0, px1, py1 = PANEL_ROI
    rgb = [f[..., ::-1] for f in frames]
    crops = np.stack([f[y0:y1, x0:x1] for f in rgb])
    panels = np.stack([np.median(f[py0:py1, px0:px1].reshape(-1, 3), axis=0) for f in rgb]).astype(np.float32)
    recipe_in = (rec[RECIPE_COLS].astype(float).values / np.array(RECIPE_SCALE, float)).astype(np.float32)
    with torch.no_grad():
        est = net(torch.from_numpy(network_input(crops, panels, [CENTER] * len(frames))),
                  torch.from_numpy(recipe_in)).numpy()
    return est, ckpt["threshold"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="fill-camera video (640x480, the vial_fill view)")
    ap.add_argument("--recipes", required=True, help="recipe queue CSV (CSI's video5_vial_recipes.csv format)")
    ap.add_argument("--out-dir", default=None, help="default: model/results/real_<video name>")
    args = ap.parse_args()
    video = Path(args.video).expanduser().resolve()
    out = Path(args.out_dir or MODEL_DIR / "results" / f"real_{video.stem}")
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
    est, threshold = estimate(frames, rec)
    df = pd.DataFrame({"vial": queue.vial, "recipe_id": queue.recipe_id,
                       "detected_start_s": [v["start_s"] for v in vials], "detected_end_s": [v["end_s"] for v in vials],
                       "snapshot_s": [v["final_s"] for v in vials], "pours_seen": [v["pours_seen"] for v in vials],
                       "snapshot": [f"snapshots/vial{int(v):02d}_final.jpg" for v in queue.vial],
                       "expected_orange": rec.expected_clean_orange.round(3),
                       "expected_blue": rec.expected_clean_blue.round(3),
                       "est_orange": est[:, 0].round(3), "est_blue": est[:, 1].round(3)})
    df["gap"] = np.maximum(np.abs(df.est_orange - df.expected_orange), np.abs(df.est_blue - df.expected_blue)).round(3)
    df["flag"] = (df.gap > threshold).astype(int)

    metrics = {"info": {"video": str(video), "recipes": str(Path(args.recipes).resolve()),
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

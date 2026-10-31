"""Break twin pictures down into a table: one row per picture, ready for the color model.

    # from version_1/ (needs opencv-python + numpy, e.g. changeover-watch/.venv/bin/python):
    python model/build_table.py twin_runs/train_camera_s1001/out
    # -> model/tables/train_camera_s1001.csv

Several runs can be given at once; vial_uid = "<run seed>-<sample_id>" keeps vials from different runs apart.

What is measured from each picture (the same code can run on real camera frames; the camera never moves):
  liq_*    median color of a fixed patch at the bottom of the vial (inside the liquid even at 10 mL)
  panel_*  median color of the white panel left of the vial (shows the lighting)
  liqn_*   liquid color divided by the panel color (lighting-corrected)
  liq_h/s/v  hue, saturation, brightness of the liquid color (OpenCV scale: hue 0-179)
  band{k}_r/g/b  median color of 8 stacked strips down the visible vial below the carrier (band0 = top,
           y 280-300; band7 = bottom, y 420-440), divided by the panel color. Shows the color through
           the whole liquid column and where the liquid ends (the real camera sees the same).

Column roles (see docs/DATASET_GENERATION.md, the leakage rule):
  X  = X_COLS: recipe info the line knows at runtime + the measured colors.
       expected_orange/blue = the dye shares the recipe says the vial should have after this pour
       (plain recipe math, no leftover).
       expected_clean_orange/blue = what a normal clean vial of this recipe looks like after this pour, with
       the station's typical leftover (NOMINAL_RESIDUE_ML of the same recipe's mix). Clean vials blend at
       pour 1, so this is the fair reference. On the real line the typical leftover is a station constant
       learned in the calibration period; here it is the middle of the twin's 5-30 mL range.
  y  = Y_COLS: actual_orange/blue (the true mix, which the model learns to predict), off_spec (main flag),
       carryover. off_spec = 1 when the actual orange or blue share differs by more than OFF_SPEC_POINTS
       from a normal clean vial of the same recipe at the same pour (same leftover, same-recipe mix).
  split = train/test, 80/20 by vial (all pours of a vial on the same side). Fixed seed, so it never changes.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

TANKS = ["T200", "T400", "T100"]          # orange, blue, clear; also the dispense order
LIQ_ROI = (293, 400, 327, 435)            # x0, y0, x1, y1 in the 640x480 FillCam picture
PANEL_ROI = (60, 60, 250, 200)            # same patch the generator measures
BAND_X = (295, 326)                       # vial strip below the carrier
BANDS = [(y, y + 20) for y in range(280, 440, 20)]
OFF_SPEC_POINTS = 0.05                    # 5 percentage points of orange or blue share
NOMINAL_RESIDUE_ML = 17.5                 # station's typical leftover (twin: 5-30 mL, uniform)
TEST_SHARE = 0.2
SPLIT_SEED = 0

ID_COLS = ["vial_uid", "run", "image", "split"]
X_COLS = ["recipe_id", "t200_ml", "t400_ml", "t100_ml", "total_ml", "stage", "stage_tank", "stages_total",
          "expected_orange", "expected_blue", "expected_clean_orange", "expected_clean_blue",
          "liq_r", "liq_g", "liq_b", "panel_r", "panel_g", "panel_b",
          "liqn_r", "liqn_g", "liqn_b", "liq_h", "liq_s", "liq_v"] + \
         [f"band{k}_{c}" for k in range(len(BANDS)) for c in "rgb"]
Y_COLS = ["actual_orange", "actual_blue", "off_spec", "off_spec_points", "carryover"]


def median_rgb(img, roi):
    x0, y0, x1, y1 = roi
    b, g, r = np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    return float(r), float(g), float(b)


def recipe_ml(row):
    return {"T200": float(row["t200_ml"]), "T400": float(row["t400_ml"]), "T100": float(row["t100_ml"])}


def shares(cum):
    total = sum(cum.values())
    return cum["T200"] / total, cum["T400"] / total


def expected_shares(row):
    """What the recipe says after this pour: the tanks poured so far, nothing else."""
    cur = recipe_ml(row)
    poured = [t for t in TANKS if cur[t] > 0][:int(row["stage"])]
    return shares({t: (cur[t] if t in poured else 0.0) for t in TANKS})


def clean_vial_shares(row, residue=None):
    """A clean vial of this recipe at this pour: same well-mixed leftover model as stages() in
    twin_tools/render_dataset.py, with previous = this recipe. With residue=None it uses the vial's true
    residue_ml (label side only); the X side passes NOMINAL_RESIDUE_ML."""
    cur = recipe_ml(row)
    total = sum(cur.values())
    residue = float(row["residue_ml"]) if residue is None else min(residue, total / 2)
    cum = {t: residue * cur[t] / total for t in TANKS}
    for t in [t for t in TANKS if cur[t] > 0][:int(row["stage"])]:
        cum[t] += cur[t] * (1.0 - residue / total)
    return shares(cum)


def assign_splits(vial_uids):
    uids = sorted(set(vial_uids))
    random.Random(SPLIT_SEED).shuffle(uids)
    n_test = round(len(uids) * TEST_SHARE)
    return {u: ("test" if i < n_test else "train") for i, u in enumerate(uids)}


def measure(img):
    """The measured color columns for one BGR picture (twin render or real camera frame)."""
    liq, panel = median_rgb(img, LIQ_ROI), median_rgb(img, PANEL_ROI)
    h, s, v = cv2.cvtColor(np.uint8([[liq[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0]
    return dict(liq_r=liq[0], liq_g=liq[1], liq_b=liq[2],
                panel_r=panel[0], panel_g=panel[1], panel_b=panel[2],
                liqn_r=round(liq[0] / panel[0], 4), liqn_g=round(liq[1] / panel[1], 4),
                liqn_b=round(liq[2] / panel[2], 4), liq_h=int(h), liq_s=int(s), liq_v=int(v),
                **{f"band{k}_{c}": round(val / p, 4)
                   for k, (y0, y1) in enumerate(BANDS)
                   for c, val, p in zip("rgb", median_rgb(img, (BAND_X[0], y0, BAND_X[1], y1)), panel)})


def table_rows(run_dir):
    run_dir = Path(run_dir)
    seed = json.loads((run_dir / "run_info.json").read_text())["args"]["seed"]
    run = run_dir.parent.name if run_dir.name == "out" else run_dir.name
    for row in csv.DictReader(open(run_dir / "labels.csv")):
        img = cv2.imread(str(run_dir / row["image"]))
        if img is None:
            print(f"  skipped (unreadable): {row['image']}", file=sys.stderr)
            continue
        exp_o, exp_b = expected_shares(row)
        act_o, act_b = float(row["orange_frac"]), float(row["blue_frac"])
        nom_o, nom_b = clean_vial_shares(row, NOMINAL_RESIDUE_ML)
        cln_o, cln_b = clean_vial_shares(row)
        points = max(abs(act_o - cln_o), abs(act_b - cln_b))
        out = {"vial_uid": f"{seed}-{row['sample_id']}", "run": run, "image": row["image"]}
        out.update({k: row[k] for k in ["recipe_id", "t200_ml", "t400_ml", "t100_ml", "total_ml",
                                        "stage", "stage_tank", "stages_total", "carryover"]})
        out.update(expected_orange=round(exp_o, 4), expected_blue=round(exp_b, 4),
                   expected_clean_orange=round(nom_o, 4), expected_clean_blue=round(nom_b, 4),
                   **measure(img),
                   actual_orange=round(act_o, 4), actual_blue=round(act_b, 4),
                   off_spec=int(points > OFF_SPEC_POINTS), off_spec_points=round(points, 4))
        yield out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="twin run folders (each with labels.csv, run_info.json, images/)")
    ap.add_argument("--out", help="output CSV (default: model/tables/<first run>.csv)")
    args = ap.parse_args()

    rows = []
    for run_dir in args.runs:
        print(f"reading {run_dir}")
        rows.extend(table_rows(run_dir))
    splits = assign_splits(r["vial_uid"] for r in rows)
    for r in rows:
        r["split"] = splits[r["vial_uid"]]

    out = Path(args.out) if args.out else Path(__file__).parent / "tables" / f"{rows[0]['run']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ID_COLS + X_COLS + Y_COLS)
        w.writeheader()
        w.writerows(rows)

    n_vials = len(splits)
    n_test = sum(s == "test" for s in splits.values())
    print(f"wrote {out}: {len(rows)} rows, {n_vials} vials ({n_vials - n_test} train / {n_test} test), "
          f"off_spec {sum(r['off_spec'] for r in rows)}, carryover {sum(int(r['carryover']) for r in rows)}")


if __name__ == "__main__":
    main()

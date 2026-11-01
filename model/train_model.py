"""Train and test the vial color model on digital-twin pictures.

What the model does: it looks at a picture of the vial (taken after a pour) together with the recipe, and
estimates how much of the liquid is orange dye and how much is blue dye. A simple rule then compares that
estimate with what a correct vial of the recipe should look like after that pour, and raises an alert when the
difference is bigger than a threshold (picked here, on held-out twin vials).

    # On a GPU (YAAD, NVIDIA PyTorch container nvcr.io/nvidia/pytorch:25.10-py3), from version_1/:
    pip install pandas                     # not included in that container
    python model/train_model.py --table model/tables/train_all.csv --pretrained \
        --run train_camera_s1001=<folder of that twin run> --run ...     # one --run per twin run in the table
    # -> model/results/twin_test/: model.pt (the trained model, not in git), metrics.json, test_predictions.csv
    # Quick check on a laptop (a few vials, 1 epoch): add --limit-vials 40 --epochs 1 --device cpu

Data: the table from build_table.py says which pictures are for training and which for testing (split by vial,
so all pictures of one vial are on the same side). 15% of the training vials are held back to pick the best
training round and the alert threshold. The test vials are used once, at the end.

Inputs: the picture and what the line knows from the recipe (see RECIPE_COLS). Never the answers.
Picture preparation (the same on real camera frames): cut out the area around the vial and divide every pixel
by the color of the white panel next to it, so darker or warmer lighting cancels out.
During training only, each picture gets a random shift of up to 6 pixels (the vial stops in a slightly
different spot) and a random liquid shade (each color channel of the vial pixels x 0.85-1.15), so the model
can't depend on the twin's exact dye colors. The vial pixels come from the twin's masks (training only).
Color-shift test: the test pictures are scored again with the liquid or the whole picture recolored (SHIFTS).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VERSION_DIR = Path(__file__).resolve().parent.parent
PANEL_ROI = (60, 60, 250, 200)                 # same patch as build_table.py
LOAD = (230, 145, 390, 480)                    # x0, y0, x1, y1 loaded around the vial (room for the offset)
CROP_W, CROP_H = 140, 320                      # network input (before resizing)
CENTER = (10, 7)                               # crop offset inside LOAD for evaluation
MAX_OFFSET = 6
VIAL_CLASS = 1                                 # mask id of the vial (glass + what's seen through it)
RECIPE_COLS = ["expected_clean_orange", "expected_clean_blue", "expected_orange", "expected_blue",
               "stage", "stages_total", "t200_ml", "t400_ml", "t100_ml", "total_ml"]
RECIPE_SCALE = [1, 1, 1, 1, 3, 3, 80, 80, 80, 80]
TARGETS = ["actual_orange", "actual_blue"]
# color-shift test: (name, (r, g, b) gains, liquid only?)
SHIFTS = [("none", (1, 1, 1), True),
          ("liquid green x0.9", (1, 0.9, 1), True), ("liquid green x1.1", (1, 1.1, 1), True),
          ("liquid red x0.9", (0.9, 1, 1), True), ("liquid red x1.1", (1.1, 1, 1), True),
          ("liquid blue x0.9", (1, 1, 0.9), True), ("liquid blue x1.1", (1, 1, 1.1), True),
          ("liquid darker x0.85", (0.85, 0.85, 0.85), True), ("liquid lighter x1.15", (1.15, 1.15, 1.15), True),
          ("camera darker x0.85", (0.85, 0.85, 0.85), False), ("camera warmer r1.1 b0.9", (1.1, 1, 0.9), False)]
SEED = 0


# ----------------------------------------------------------------------------------------------
# Data (numpy only)
# ----------------------------------------------------------------------------------------------
def load_pictures(df, run_dirs):
    """Crops (N, H, W, 3) uint8, vial masks (N, H, W) bool, panel colors (N, 3) float."""
    x0, y0, x1, y1 = LOAD
    px0, py0, px1, py1 = PANEL_ROI
    crops = np.zeros((len(df), y1 - y0, x1 - x0, 3), np.uint8)
    masks = np.zeros((len(df), y1 - y0, x1 - x0), bool)
    panels = np.zeros((len(df), 3), np.float32)
    for i, (run, image) in enumerate(zip(df.run, df.image)):
        d = run_dirs[run]
        img = np.asarray(Image.open(d / image).convert("RGB"))
        crops[i] = img[y0:y1, x0:x1]
        panels[i] = np.median(img[py0:py1, px0:px1].reshape(-1, 3), axis=0)
        mask_path = d / "masks" / (Path(image).stem + ".png")
        if mask_path.exists():
            masks[i] = np.asarray(Image.open(mask_path))[y0:y1, x0:x1] == VIAL_CLASS
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{len(df)} pictures", flush=True)
    return crops, masks, panels


def recipe_vector(df):
    """Recipe inputs, scaled to roughly 0-1."""
    return (df[RECIPE_COLS].astype(float).values / np.array(RECIPE_SCALE, float)).astype(np.float32)


def apply_shift(crops, masks, panels, gains, liquid_only):
    """Pictures as if the liquid (vial pixels) or the whole picture had these r, g, b gains."""
    g = np.array(gains, np.float32)
    c = crops.astype(np.float32)
    if liquid_only:
        c = np.where(masks[..., None], c * g, c)
        p = panels
    else:
        c, p = c * g, panels * g
    return np.clip(c, 0, 255).astype(np.uint8), np.clip(p, 1, 255)


def network_input(crops, panels, offsets):
    """Crop at the given offsets and divide by the panel color -> float32 (N, 3, H, W)."""
    out = np.zeros((len(crops), 3, CROP_H, CROP_W), np.float32)
    for i, (ox, oy) in enumerate(offsets):
        patch = crops[i, oy:oy + CROP_H, ox:ox + CROP_W].astype(np.float32) / panels[i]
        out[i] = patch.transpose(2, 0, 1)
    return out


def augment(crops, masks, panels, rng, shade):
    """Training batch: random liquid shade per picture and random crop offset."""
    c = crops.astype(np.float32)
    gains = rng.uniform(1 - shade, 1 + shade, size=(len(c), 1, 1, 3)).astype(np.float32)
    c = np.where(masks[..., None], c * gains, c)
    c = np.clip(c, 0, 255).astype(np.uint8)
    offsets = [(CENTER[0] + rng.integers(-MAX_OFFSET, MAX_OFFSET + 1),
                CENTER[1] + rng.integers(-MAX_OFFSET, MAX_OFFSET + 1)) for _ in range(len(c))]
    return network_input(c, panels, offsets)


# ----------------------------------------------------------------------------------------------
# Scores (numpy only)
# ----------------------------------------------------------------------------------------------
def roc_auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(s).rank().values                 # ties get the average rank
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def scores(y, score, flag):
    y, flag = np.asarray(y).astype(int), np.asarray(flag).astype(int)
    tp, fp = int(((flag == 1) & (y == 1)).sum()), int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"n": int(len(y)), "positives": int(y.sum()), "accuracy": float((flag == y).mean()),
           "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "false_alarm_rate": float(flag[y == 0].mean()) if (y == 0).any() else None,
           "roc_auc": roc_auc(y, score)}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def best_threshold(y, gap):
    return max(np.arange(0.02, 0.41, 0.01), key=lambda t: scores(y, gap, gap > t)["f1"])


def mix_gap(df, est):
    return np.maximum(np.abs(est[:, 0] - df.expected_clean_orange.values),
                      np.abs(est[:, 1] - df.expected_clean_blue.values))


def vial_level(df, flag):
    v = pd.DataFrame({"vial": df.vial_uid.values, "flag": flag, "off": df.off_spec.values,
                      "carry": df.carryover.values}).groupby("vial").max()
    return {"vials": int(len(v)),
            "off_spec_vials_caught": f"{int(((v.flag == 1) & (v.off == 1)).sum())} of {int(v.off.sum())}",
            "carryover_vials_caught": f"{int(((v.flag == 1) & (v.carry == 1)).sum())} of {int(v.carry.sum())}",
            "clean_vials_falsely_flagged": f"{int(((v.flag == 1) & (v.carry == 0)).sum())} of {int((v.carry == 0).sum())}"}


# ----------------------------------------------------------------------------------------------
# Network (torch)
# ----------------------------------------------------------------------------------------------
def build_net(torch, torchvision, pretrained, n_recipe=len(RECIPE_COLS)):
    nn = torch.nn
    weights = None
    if pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            torchvision.models.resnet18(weights=weights)          # triggers the download now, if needed
        except Exception as e:                                     # e.g. no internet on the GPU machine
            print(f"WARNING: pretrained weights unavailable ({e}); training from scratch", flush=True)
            weights = None

    class MixNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = torchvision.models.resnet18(weights=weights)
            self.cnn.fc = nn.Identity()
            self.head = nn.Sequential(nn.Linear(512 + n_recipe, 128), nn.ReLU(), nn.Linear(128, 2))

        def forward(self, img, recipe):
            return torch.sigmoid(self.head(torch.cat([self.cnn(img), recipe], dim=1)))

    return MixNet(), weights is not None


def predict(torch, net, imgs, recipes, device, batch=128):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            out.append(net(torch.from_numpy(imgs[i:i + batch]).to(device),
                           torch.from_numpy(recipes[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="table from build_table.py")
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR for a run's images (repeatable)")
    ap.add_argument("--out-dir", default=str(VERSION_DIR / "model" / "results" / "twin_test"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shade-aug", type=float, default=0.15, help="liquid shade augmentation, x U(1-a, 1+a)")
    ap.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (downloads them)")
    ap.add_argument("--limit-vials", type=int, default=0, help="use only this many vials (smoke test)")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    args = ap.parse_args()

    import torch
    import torchvision
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}, torch {torch.__version__}", flush=True)

    df = pd.read_csv(args.table)
    if args.limit_vials:
        keep = pd.Series(df.vial_uid.unique()).sample(args.limit_vials, random_state=SEED)
        df = df[df.vial_uid.isin(keep)]
    df = df.reset_index(drop=True)
    run_dirs = {r: VERSION_DIR / "twin_runs" / r / "out" for r in df.run.unique()}
    for spec in args.run:
        name, d = spec.split("=", 1)
        run_dirs[name] = Path(d).expanduser().resolve()

    train_vials = sorted(df[df.split == "train"].vial_uid.unique())
    random.Random(SEED).shuffle(train_vials)
    val_vials = set(train_vials[:max(1, round(len(train_vials) * 0.15))])
    part = np.where(df.split == "test", "test", np.where(df.vial_uid.isin(val_vials), "val", "fit"))
    print(f"{len(df)} pictures: fit {(part == 'fit').sum()}, val {(part == 'val').sum()}, "
          f"test {(part == 'test').sum()}", flush=True)

    t0 = time.time()
    crops, masks, panels = load_pictures(df, run_dirs)
    print(f"loaded pictures in {time.time() - t0:.0f} s; vial masks found for {masks.any(axis=(1, 2)).mean():.0%}",
          flush=True)
    recipes = recipe_vector(df)
    y = df[TARGETS].values.astype(np.float32)
    center = [CENTER] * len(df)
    idx = {p: np.where(part == p)[0] for p in ["fit", "val", "test"]}
    val_imgs = network_input(crops[idx["val"]], panels[idx["val"]], center[:len(idx["val"])])

    net, pretrained = build_net(torch, torchvision, args.pretrained, recipes.shape[1])
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best = (np.inf, None, 0)
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(idx["fit"])
        losses = []
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            imgs = augment(crops[b], masks[b], panels[b], rng, args.shade_aug)
            pred = net(torch.from_numpy(imgs).to(device), torch.from_numpy(recipes[b]).to(device))
            loss = torch.nn.functional.l1_loss(pred, torch.from_numpy(y[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_mae = float(np.abs(predict(torch, net, val_imgs, recipes[idx["val"]], device) - y[idx["val"]]).mean())
        if val_mae < best[0]:
            best = (val_mae, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
        print(f"epoch {epoch}/{args.epochs}: train L1 {np.mean(losses):.4f}, val MAE {val_mae:.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    net.load_state_dict(best[1])

    val_df = df.iloc[idx["val"]]
    t = best_threshold(val_df.off_spec.values,
                       mix_gap(val_df, predict(torch, net, val_imgs, recipes[idx["val"]], device)))
    test = df.iloc[idx["test"]].reset_index(drop=True)
    tc, tm, tp, tr = crops[idx["test"]], masks[idx["test"]], panels[idx["test"]], recipes[idx["test"]]

    robust, est = {}, None
    for name, gains, liquid_only in SHIFTS:
        c, p = apply_shift(tc, tm, tp, gains, liquid_only)
        e = predict(torch, net, network_input(c, p, [CENTER] * len(c)), tr, device)
        gap = mix_gap(test, e)
        r = scores(test.off_spec, gap, gap > t)
        robust[name] = {"mix_rule_cnn": {k: r[k] for k in ["roc_auc", "f1", "recall", "false_alarm_rate"]}}
        if name == "none":
            est, test_gap = e, gap
    flag = (test_gap > t).astype(int)
    results = {
        "info": {"table": args.table, "runs": {k: str(v) for k, v in run_dirs.items()}, "device": device,
                 "pretrained": pretrained, "epochs": args.epochs, "best_epoch": best[2],
                 "shade_aug": args.shade_aug, "fit_pictures": len(idx["fit"]), "val_pictures": len(idx["val"]),
                 "test_pictures": len(idx["test"]), "test_vials": int(test.vial_uid.nunique()),
                 "minutes": round((time.time() - t0) / 60, 1),
                 "note": "scored on TWIN test vials only; twin colors are uncalibrated (approx_v1)"},
        "mix_rule_cnn": {
            "threshold_points": round(float(t), 2),
            "mix_error_mae": {"orange": round(float(np.abs(est[:, 0] - test.actual_orange).mean()), 4),
                              "blue": round(float(np.abs(est[:, 1] - test.actual_blue).mean()), 4)},
            "test": scores(test.off_spec, test_gap, flag),
            "test_by_stage": {int(s): scores(g.off_spec, test_gap[g.index], flag[g.index])
                              for s, g in test.groupby("stage")},
            "test_vials": vial_level(test, flag)},
        "robustness": robust}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.concat([test[["vial_uid", "run", "image", "recipe_id", "stage", "carryover", "off_spec", "off_spec_points",
                     "actual_orange", "actual_blue", "expected_clean_orange", "expected_clean_blue"]],
               pd.DataFrame({"est_orange": est[:, 0].round(4), "est_blue": est[:, 1].round(4),
                             "gap": test_gap.round(4), "flag": flag})], axis=1).to_csv(
        out / "test_predictions.csv", index=False)
    torch.save({"state_dict": best[1], "threshold": float(t), "recipe_cols": RECIPE_COLS,
                "recipe_scale": RECIPE_SCALE, "load": LOAD, "center": CENTER, "crop": (CROP_W, CROP_H),
                "panel_roi": PANEL_ROI}, out / "model.pt")

    s = results["mix_rule_cnn"]["test"]
    print(f"\ntwin test: AUC {s['roc_auc']}, F1 {s['f1']}, recall {s['recall']}, "
          f"false alarms {s['false_alarm_rate']}, threshold {t:.2f}")
    print(f"mix error (MAE): {results['mix_rule_cnn']['mix_error_mae']}")
    print(f"vials: {results['mix_rule_cnn']['test_vials']}")
    print("robustness (AUC / F1 / false alarms):")
    for name, r in robust.items():
        r = r["mix_rule_cnn"]
        print(f"  {name:26s} " + "  ".join("  n/a" if r[k] is None else f"{r[k]:.3f}"
                                           for k in ["roc_auc", "f1", "false_alarm_rate"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

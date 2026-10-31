"""Model 1: simple color measurements -> "is the mix wrong for this pour?"  Trained and scored on the twin table.

    # from version_1/ (needs pandas + scikit-learn, e.g. changeover-watch/.venv/bin/python):
    python model/build_table.py twin_runs/train_camera_s1001/out
    python model/train_color_model.py model/tables/train_camera_s1001.csv
    # -> model/results/model1/metrics.json, test_predictions.csv

Models compared (all chosen and tuned on the TRAIN vials only, with 5-fold cross-validation grouped by vial;
the TEST vials are scored once at the end):
  mix_rule    two steps: a regressor estimates the orange and blue shares from the measured colors only,
              then flags the pour when the estimate is more than a threshold away from expected_clean_*.
              The threshold is picked on the train folds. This is the explainable version
              ("estimated 66% orange, a clean vial would be 92%").
  direct_gb   one gradient-boosted classifier: colors + recipe info -> off_spec.
  direct_lr   the same inputs, logistic regression (the simplest model).
  end_only    baseline: direct_gb trained and scored on the LAST pour of each vial only.
  recipe_only bias check: direct_gb WITHOUT any image columns. Should be near chance (AUC ~0.5);
              if not, the twin generation leaks the answer through the recipe.
Robustness test: the trained mix_rule and direct_gb are re-scored on the test vials with the measured colors
shifted, to see what happens when the real camera's colors differ from the twin's (thresholds unchanged):
  liquid shifts  only the liquid gets a different shade (e.g. green x0.9): the twin's dye colors are wrong
  camera shifts  the whole picture, white panel included (e.g. x0.85): different exposure or white balance
Measured columns are recomputed from the shifted colors (liquid/panel ratios, hue/saturation/brightness).
Inputs follow the leakage rule in docs/DATASET_GENERATION.md: only the X columns from build_table.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLOR_COLS = ["liq_r", "liq_g", "liq_b", "panel_r", "panel_g", "panel_b",
              "liqn_r", "liqn_g", "liqn_b", "liq_h", "liq_s", "liq_v"] + \
             [f"band{k}_{c}" for k in range(8) for c in "rgb"]
RECIPE_COLS = ["t200_ml", "t400_ml", "t100_ml", "total_ml", "stage", "stages_total",
               "expected_orange", "expected_blue", "expected_clean_orange", "expected_clean_blue"]
# recipe_id is left out on purpose: random recipes (N...) each appear once, so it can only be memorized.
# stage_tank is covered by stage + the tank mL.
TARGET = "off_spec"
SEED = 0
BAND_COLS = [f"band{k}_{c}" for k in range(8) for c in "rgb"]
# (name, (r, g, b) gains, liquid only?)
SHIFTS = [("none", (1, 1, 1), True),
          ("liquid green x0.9", (1, 0.9, 1), True), ("liquid green x1.1", (1, 1.1, 1), True),
          ("liquid red x0.9", (0.9, 1, 1), True), ("liquid red x1.1", (1.1, 1, 1), True),
          ("liquid blue x0.9", (1, 1, 0.9), True), ("liquid blue x1.1", (1, 1, 1.1), True),
          ("liquid darker x0.85", (0.85, 0.85, 0.85), True), ("liquid lighter x1.15", (1.15, 1.15, 1.15), True),
          ("camera darker x0.85", (0.85, 0.85, 0.85), False), ("camera warmer r1.1 b0.9", (1.1, 1, 0.9), False)]


def scores(y, prob, flag):
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "accuracy": accuracy_score(y, flag), "precision": precision_score(y, flag, zero_division=0),
           "recall": recall_score(y, flag), "f1": f1_score(y, flag),
           "false_alarm_rate": float(flag[y == 0].mean()) if (y == 0).any() else None}
    out["roc_auc"] = roc_auc_score(y, prob) if len(set(y)) > 1 else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def gb_classifier():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)


def lr_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def cv_probs(make, X, y, groups):
    """Out-of-fold probabilities on the train vials (used to pick thresholds without touching test)."""
    prob = np.zeros(len(y))
    for fit_i, val_i in GroupKFold(n_splits=5).split(X, y, groups):
        prob[val_i] = make().fit(X.iloc[fit_i], y.iloc[fit_i]).predict_proba(X.iloc[val_i])[:, 1]
    return prob


def best_threshold(y, score, candidates):
    return max(candidates, key=lambda t: f1_score(y, score > t))


def run_classifier(name, make, cols, train, test, results, preds, fitted=None):
    Xtr, ytr, Xte, yte = train[cols], train[TARGET], test[cols], test[TARGET]
    t = best_threshold(ytr, cv_probs(make, Xtr, ytr, train.vial_uid), np.arange(0.1, 0.91, 0.05))
    model = make().fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    results[name] = {"inputs": cols, "threshold": round(float(t), 2), "test": scores(yte, prob, prob > t)}
    preds[f"{name}_prob"] = pd.Series(prob, index=test.index)
    preds[f"{name}_flag"] = pd.Series((prob > t).astype(int), index=test.index)
    if fitted is not None:
        fitted[name] = lambda df: model.predict_proba(df[cols])[:, 1], t


def fit_mix(train_X, train_y):
    return {k: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED)
            .fit(train_X, train_y[k]) for k in ["actual_orange", "actual_blue"]}


def predict_mix(reg, X):
    return np.clip(reg["actual_orange"].predict(X), 0, 1), np.clip(reg["actual_blue"].predict(X), 0, 1)


def mix_estimates(train_X, train_y, X):
    return predict_mix(fit_mix(train_X, train_y), X)


def hsv_opencv(r, g, b):
    """Hue 0-179, saturation, brightness of colors, computed exactly as build_table.py does (8-bit, OpenCV)."""
    bgr = np.stack([b, g, r], axis=-1).astype(np.uint8)[None]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0]
    return hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)


def shifted(df, gains, liquid_only):
    """The table as if the liquid (or the whole picture) had come out with these r, g, b gains."""
    d = df.copy()
    for c, k in zip("rgb", gains):
        d[f"liq_{c}"] = np.clip(df[f"liq_{c}"] * k, 0, 255)
        if not liquid_only:
            d[f"panel_{c}"] = np.clip(df[f"panel_{c}"] * k, 0, 255)
        d[f"liqn_{c}"] = d[f"liq_{c}"] / d[f"panel_{c}"]
        for b in range(8):                           # bands are already divided by the panel
            if liquid_only:
                d[f"band{b}_{c}"] = df[f"band{b}_{c}"] * k
    d["liq_h"], d["liq_s"], d["liq_v"] = hsv_opencv(*(d[f"liq_{c}"].values for c in "rgb"))
    return d


def robustness(test, fitted):
    out = {}
    for name, gains, liquid_only in SHIFTS:
        d = shifted(test, gains, liquid_only)
        out[name] = {}
        for model, (score_fn, t) in fitted.items():
            s = score_fn(d)
            r = scores(d[TARGET], s, s > t)
            out[name][model] = {k: r[k] for k in ["roc_auc", "f1", "recall", "false_alarm_rate"]}
    return out


def mix_gap(df, est_o, est_b):
    return np.maximum(np.abs(est_o - df.expected_clean_orange.values), np.abs(est_b - df.expected_clean_blue.values))


def run_mix_rule(train, test, results, preds, fitted):
    ys = ["actual_orange", "actual_blue"]
    gap_oof = np.zeros(len(train))
    for fit_i, val_i in GroupKFold(n_splits=5).split(train, groups=train.vial_uid):
        f, v = train.iloc[fit_i], train.iloc[val_i]
        gap_oof[val_i] = mix_gap(v, *mix_estimates(f[COLOR_COLS], f[ys], v[COLOR_COLS]))
    t = best_threshold(train[TARGET], gap_oof, np.arange(0.02, 0.41, 0.01))
    reg = fit_mix(train[COLOR_COLS], train[ys])
    fitted["mix_rule"] = lambda df: mix_gap(df, *predict_mix(reg, df[COLOR_COLS])), t
    est_o, est_b = predict_mix(reg, test[COLOR_COLS])
    gap = mix_gap(test, est_o, est_b)
    results["mix_rule"] = {
        "inputs": COLOR_COLS + ["expected_clean_orange", "expected_clean_blue (compared, not learned)"],
        "threshold_points": round(float(t), 2),
        "mix_error_mae": {"orange": round(mean_absolute_error(test.actual_orange, est_o), 4),
                          "blue": round(mean_absolute_error(test.actual_blue, est_b), 4)},
        "test": scores(test[TARGET], gap, gap > t)}
    preds["est_orange"], preds["est_blue"] = est_o.round(4), est_b.round(4)
    preds["mix_rule_gap"], preds["mix_rule_flag"] = gap.round(4), (gap > t).astype(int)


def vial_level(test, preds, flag_col):
    """A vial is flagged if any of its pours is flagged."""
    v = pd.DataFrame({"vial": test.vial_uid, "flag": preds[flag_col], "off": test[TARGET],
                      "carry": test.carryover}).groupby("vial").max()
    return {"vials": int(len(v)),
            "off_spec_vials_caught": f"{int(((v.flag == 1) & (v.off == 1)).sum())} of {int(v.off.sum())}",
            "carryover_vials_caught": f"{int(((v.flag == 1) & (v.carry == 1)).sum())} of {int(v.carry.sum())}",
            "clean_vials_falsely_flagged": f"{int(((v.flag == 1) & (v.carry == 0)).sum())} of {int((v.carry == 0).sum())}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="table from build_table.py")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results" / "model1"))
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    results, preds, fitted = {}, pd.DataFrame(index=test.index), {}

    run_mix_rule(train, test, results, preds, fitted)
    run_classifier("direct_gb", gb_classifier, COLOR_COLS + RECIPE_COLS, train, test, results, preds, fitted)
    run_classifier("direct_lr", lr_classifier, COLOR_COLS + RECIPE_COLS, train, test, results, preds)
    run_classifier("recipe_only", gb_classifier, RECIPE_COLS, train, test, results, preds)

    last = lambda d: d[d.stage == d.stages_total].reset_index(drop=True)
    end_preds = pd.DataFrame(index=last(test).index)
    run_classifier("end_only", gb_classifier, COLOR_COLS + RECIPE_COLS, last(train), last(test), results, end_preds)

    for name in ["mix_rule", "direct_gb", "direct_lr"]:
        results[name]["test_by_stage"] = {
            int(s): scores(g[TARGET], preds.loc[g.index, f"{name}_gap" if name == "mix_rule" else f"{name}_prob"],
                           preds.loc[g.index, f"{name}_flag"]) for s, g in test.groupby("stage")}
        results[name]["test_vials"] = vial_level(test, preds, f"{name}_flag")
    results["end_only"]["test_vials"] = vial_level(last(test), end_preds, "end_only_flag")
    results["robustness"] = robustness(test, fitted)

    info = {"table": args.table, "train_rows": len(train), "test_rows": len(test),
            "train_vials": train.vial_uid.nunique(), "test_vials": test.vial_uid.nunique(),
            "note": "scored on TWIN test vials only; twin colors are uncalibrated (approx_v1)"}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"info": info, **results}, indent=2))
    pd.concat([test[["vial_uid", "image", "recipe_id", "stage", "carryover", TARGET, "off_spec_points",
                     "actual_orange", "actual_blue", "expected_clean_orange", "expected_clean_blue"]], preds],
              axis=1).to_csv(out / "test_predictions.csv", index=False)

    print(f"train {info['train_vials']} vials / test {info['test_vials']} vials")
    print(f"{'model':12s} {'AUC':>6s} {'F1':>6s} {'recall':>7s} {'prec':>6s} {'false-alarm':>11s}  vials")
    for name in ["mix_rule", "direct_gb", "direct_lr", "end_only", "recipe_only"]:
        s = results[name]["test"]
        print(f"{name:12s} {s['roc_auc']:6.3f} {s['f1']:6.3f} {s['recall']:7.3f} {s['precision']:6.3f} "
              f"{s['false_alarm_rate']:11.3f}  {results[name].get('test_vials', '')}")
    print(f"mix error (MAE, share points): {results['mix_rule']['mix_error_mae']}")
    print(f"\nrobustness (test vials, colors shifted; thresholds unchanged)")
    print(f"{'shift':26s} " + "  ".join(f"{m + ' AUC/F1/false-alarm':>34s}" for m in fitted))
    for name, r in results["robustness"].items():
        print(f"{name:26s} " + "  ".join(f"{r[m]['roc_auc']:>18.3f} {r[m]['f1']:6.3f} {r[m]['false_alarm_rate']:8.3f}"
                                         for m in fitted))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

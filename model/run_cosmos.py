"""Model 3: NVIDIA Cosmos Reason (vision-language model), no training. It gets the picture + the recipe and the
mix a clean vial should have after this pour, and answers with its estimate of the orange and blue shares,
match / no match, and a short reason. Scored with the same rule and metrics as models 1 and 2.

    # GPU (YAAD, nvcr.io/nvidia/pytorch:25.10-py3), from version_1/:
    pip install pandas transformers accelerate
    python model/run_cosmos.py --table model/tables/train_all.csv --limit 200 --out-dir /job/out
    # add --blind to leave the recipe and expected mix out of the prompt (fairer test of its color reading)
    # -> answers.csv (one row per picture: raw answer + parsed values), metrics.json

--limit picks a fixed, balanced sample of TEST pictures (half off-spec, half fine; same seed every time), because
a 7B model takes seconds per picture. Models 1 and 2 can be re-scored on exactly the same pictures from
answers.csv. --images-csv runs on any other pictures (e.g. real snapshots) with the same columns minus labels.
The prompt gives only what the line knows at runtime (recipe, pour, expected_clean_*), never the labels.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

VERSION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "nvidia/Cosmos-Reason1-7B"
SEED = 0

SYSTEM = ("You are a quality inspector at a liquid filling station. A fixed camera looks at a clear glass vial "
          "under a filling nozzle. The vial holds a mix of orange dye, blue dye and clear water. Mixed together, "
          "orange and blue make brown, olive or dark green shades depending on the amounts.")

PROMPT = """The vial in the center, under the nozzle, is being filled with recipe {recipe_id}: {t200} mL orange, {t400} mL blue, {t100} mL clear, poured in that order.
This picture was taken after pour {stage} of {stages_total}.
A correctly filled vial should now be about {exp_o}% orange dye, {exp_b}% blue dye and {exp_c}% clear water.
Leftover liquid from the previous recipe can change the color.

Look at the liquid's color. Estimate its actual orange and blue percentages, and say whether it matches what is expected.
Answer with only this JSON on one line:
{{"orange_pct": <number>, "blue_pct": <number>, "matches": <true or false>, "reason": "<one short sentence>"}}"""

# --blind: no recipe and no expected mix in the prompt, so the answer can't copy them; the flag comes only from
# comparing the estimate with expected_clean_* afterwards (the same rule as models 1 and 2).
PROMPT_BLIND = """The vial in the center, under the nozzle, holds a mix of orange dye, blue dye and clear water.
Look only at the liquid's color. Estimate what percentage of the liquid is orange dye and what percentage is blue dye
(the rest is clear water).
Answer with only this JSON on one line:
{"orange_pct": <number>, "blue_pct": <number>, "reason": "<one short sentence describing the color>"}"""


def prompt_for(r, blind=False):
    if blind:
        return PROMPT_BLIND
    exp_o, exp_b = round(100 * r.expected_clean_orange), round(100 * r.expected_clean_blue)
    return PROMPT.format(recipe_id=r.recipe_id, t200=int(r.t200_ml), t400=int(r.t400_ml), t100=int(r.t100_ml),
                         stage=int(r.stage), stages_total=int(r.stages_total), exp_o=exp_o, exp_b=exp_b,
                         exp_c=100 - exp_o - exp_b)


def parse(answer):
    """Last JSON object in the answer (the model may think out loud first)."""
    for m in reversed(re.findall(r"\{[^{}]*\}", answer)):
        try:
            d = json.loads(m)
            o, b = float(d["orange_pct"]) / 100, float(d["blue_pct"]) / 100
            m = d.get("matches")
            matches = m if isinstance(m, bool) or m is None else str(m).lower() == "true"
            return o, b, matches, str(d.get("reason", ""))
        except (ValueError, KeyError, TypeError):
            continue
    return None, None, None, ""


def sample_test(df, n):
    test = df[df.split == "test"]
    if not n or n >= len(test):
        return test
    off, ok = test[test.off_spec == 1], test[test.off_spec == 0]
    k = min(n // 2, len(off), len(ok))
    return pd.concat([off.sample(k, random_state=SEED), ok.sample(n - k, random_state=SEED)]).sort_index()


def scores(y, score, flag):
    y, flag, score = np.asarray(y).astype(int), np.asarray(flag).astype(int), np.asarray(score, float)
    tp, fp = int(((flag == 1) & (y == 1)).sum()), int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    prec, rec = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    ranks = pd.Series(score).rank().values
    auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else None
    out = {"n": int(len(y)), "positives": int(pos), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "false_alarm_rate": float(flag[y == 0].mean()) if neg else None, "roc_auc": auc}
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", help="table from build_table.py (scored on its test pictures)")
    ap.add_argument("--images-csv", help="instead of --table: pictures to answer for (no scoring)")
    ap.add_argument("--run", action="append", default=[], help="NAME=DIR for a run's images (repeatable)")
    ap.add_argument("--limit", type=int, default=200, help="balanced sample of test pictures (0 = all)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--blind", action="store_true", help="prompt without recipe or expected mix (see PROMPT_BLIND)")
    ap.add_argument("--out-dir", default=str(VERSION_DIR / "model" / "results" / "model3"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if args.table:
        rows = sample_test(pd.read_csv(args.table), args.limit)
    else:
        rows = pd.read_csv(args.images_csv)
    run_dirs = {r: VERSION_DIR / "twin_runs" / r / "out" for r in rows.run.unique()}
    for spec in args.run:
        name, d = spec.split("=", 1)
        run_dirs[name] = Path(d).expanduser().resolve()

    t0 = time.time()
    print(f"loading {args.model}, torch {torch.__version__}, cuda {torch.cuda.is_available()}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"loaded in {time.time() - t0:.0f} s; {len(rows)} pictures", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answers = []
    for i, r in enumerate(rows.itertuples()):
        image = Image.open(run_dirs[r.run] / r.image).convert("RGB")
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt_for(r, args.blind)}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        o, b, matches, reason = parse(answer)
        answers.append({"vial_uid": getattr(r, "vial_uid", ""), "run": r.run, "image": r.image,
                        "recipe_id": r.recipe_id, "stage": r.stage,
                        "expected_clean_orange": r.expected_clean_orange, "expected_clean_blue": r.expected_clean_blue,
                        "off_spec": getattr(r, "off_spec", ""), "carryover": getattr(r, "carryover", ""),
                        "actual_orange": getattr(r, "actual_orange", ""), "actual_blue": getattr(r, "actual_blue", ""),
                        "est_orange": o, "est_blue": b, "matches": matches, "reason": reason, "answer": answer})
        print(f"[{i + 1}/{len(rows)}] {r.image} -> orange {o}, blue {b}, matches {matches} "
              f"({time.time() - t0:.0f} s)", flush=True)
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            pd.DataFrame(answers).to_csv(out / "answers.csv", index=False)     # keep partial results

    a = pd.DataFrame(answers)
    info = {"model": args.model, "blind": args.blind, "pictures": len(a), "parsed": int(a.est_orange.notna().sum()),
            "minutes": round((time.time() - t0) / 60, 1), "prompt": PROMPT_BLIND if args.blind else PROMPT, "system": SYSTEM}
    results = {"info": info}
    if args.table:
        ok = a[a.est_orange.notna()].reset_index(drop=True)
        gap = np.maximum(np.abs(ok.est_orange - ok.expected_clean_orange), np.abs(ok.est_blue - ok.expected_clean_blue))
        results["gap_rule_0.08"] = scores(ok.off_spec, gap, gap > 0.08)          # model 2's threshold
        if not args.blind:
            said = ok.matches.fillna(True).astype(bool)
            results["says_no_match"] = scores(ok.off_spec, (~said).astype(int), ~said)
        results["mix_error_mae"] = {"orange": round(float((ok.est_orange - ok.actual_orange).abs().mean()), 4),
                                    "blue": round(float((ok.est_blue - ok.actual_blue).abs().mean()), 4)}
        print(json.dumps(results, indent=1)[:1500])
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

# Generating Training Pictures from the Digital Twin

**What this makes:** labeled pictures of the fill-station vial from the twin's FillCam (the copy of the real vial_fill camera), **one render after each pour** of a recipe, **saved in two styles**, plus a CSV describing every picture:
- **clean:** straight from Isaac Sim, lossless PNG, crisp like the organizers' `fillcam_frame686.png`
- **camera:** the same render with the real-camera look (darker, noise, slight blur, timestamp), JPEG

So the model sees two kinds of rendering of the same vial and pour.

**Files** (all inside `version_1/twin_tools/`). The organizers' twin is **not copied** into version_1; the script points to it with `--twin` (default: the untouched original at `~/Desktop/Hackathon/csi_digital_twin_v1`):
| File | What it is |
|---|---|
| `recipe_rules.json` | The recipe rules and the 10 real recipes, each rule with its documentation source. **Edit rules here, not in the script.** |
| `render_dataset.py` | The picture generator (Isaac Sim 5.1) |
| `render_example.py` | The organizers' original one-picture script. **Unchanged**, kept as the known-working reference |

The twin's scene files are never changed. Every edit (liquid color and height, lights, camera) is made in memory while rendering and thrown away afterwards.

---

## How it works, in plain words
For each **vial** (a "sample"):
1. **Pick a recipe:** about half are the 10 real recipes (5–14), and half are random new recipes. Every recipe is checked against `recipe_rules.json` (≤80 mL, ≥10 mL per tank used, ≤50 mL orange, no T300).
2. **Pick the vial before it:**
   - **Same recipe** = a *clean* sample.
   - **Different recipe** = a *carryover* sample (about half).
3. **Start with leftover liquid:** every vial begins with 5–30 mL of the previous vial's mix (capped at half the vial). This is the organizers' "shared mixing path," treated as well mixed. An equal amount of **this** vial's own mix stays behind in the path for the next vial, taken proportionally from every pour, so the total always matches the recipe and no dye is ever lost.
   - **Clean samples** (previous vial = same recipe): the **finished** vial matches the recipe exactly, but **early-pour pictures are a blend, not pure dye.** Example: after the orange pour, the vial already holds some of the previous vial's full mix. That's intended; real vial 11 showed the same thing.
   - **Carryover samples** (previous vial = different recipe): the finished vial is *(leftover ÷ vial size)* old mix + the rest new recipe. Typically ~30% old mix, at most 50%. That's the contamination the model must catch.
4. **Set this vial's conditions once:** light brightness and warmth (random per vial, or a fixed `--light-mode`), plus the camera look (darkness, blur, noise level). **Every pour of the vial shares them,** as on the real line, where lights don't change mid-fill. The camera never moves.
5. **For each pour,** in the real order (orange → blue → clear, skipping unused tanks):
   - set the liquid **height** to the mL so far, using the twin's **settled finished-vial shape** (frame 686) scaled to the right height, with the bottom kept in place. This gives a **calm surface**; earlier versions borrowed mid-pour frames, which caused "foamy" tops.
   - set the liquid **color** to the mix so far, keeping the organizers' opacity of 0.6 so the liquid stays clean and see-through like their render
   - **hide the pouring stream** (a "pour finished, liquid settled" picture)
   - **render FillCam once,** then save it twice:
     - **clean** (`images/clean/…png`)
     - **camera** (`images/camera/…jpg`): the vial's camera look plus fresh noise grain and the timestamp text
   - **measure the white panel** (a patch left of the vial) in **each** saved picture → `panel_r/g/b`. This shows the lighting the way a real camera could measure it
   - save the **pictures**, a **mask** (which pixels are vial, carrier, nozzle and so on; the see-through liquid isn't included), and **one CSV row per saved picture** (so 2 rows per render)

**Example:** recipe 8 (20 orange + 20 blue + 20 clear) gives 3 renders per variation (after orange, after blue, after clear), so **6 pictures**: 3 clean + 3 camera.

---

## How to run it

**1. Plan only, on your laptop (free, no Isaac Sim).** This checks the recipes and writes the CSV, without pictures:
```bash
cd ~/Desktop/Hackathon/version_1 && python3 twin_tools/render_dataset.py --dry-run --count 5 --variations 1 --seed 4 --out-dir dataset_plan
```

**2. Small test render on YAAD (do this before anything big).** The workspace twin folder must hold `cameras/` and `stations/fill_station/` (already uploaded). Upload `twin_tools/render_dataset.py` and `twin_tools/recipe_rules.json` into its `tools/` folder, then submit a job with `input_path` = that twin folder:
- **image:** `nvcr.io/nvidia/isaac-sim:5.1.0`
- **env:** `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y`
- **command:** `cd /job/in && /isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 5 --variations 1 --seed 4 --out-dir /job/out`
  (seed 4 gives a good test mix: 3 real recipes, 2 random, 3 carryover, 11 pictures)
- **memory:** 48 GB · **cpus:** 8 · **max_minutes:** 60

Then fetch `/job/out` (over FortiClient) and **look at every picture** before going bigger.

**3. Bigger runs.** Raise `--count` (vials) and `--variations` (versions of each whole vial, each under new lighting and camera look). Renders ≈ count × ~2.2 pours × variations; pictures = renders × 2 styles. 1,000 vials × 1 variation ≈ 2,200 renders = 4,400 pictures. Timing test (Oct 29, 32 subframes): ~105 s startup + ~2 s per render, so about 74 min for 1,000 vials. Use a new `--seed` for each big run.

**Options:**
| Option | Default | Meaning |
|---|---|---|
| `--count` | 20 | Number of vials |
| `--variations` | 2 | Versions of each vial; each re-renders all its pours under a new lighting and camera look |
| `--light-mode` | random | `random` per vial (training), or fixed `neutral` (organizers' lights) / `dim` / `bright` / `warm` / `cool` |
| `--real-share` | 0.5 | Share of vials using real recipes 5–14 |
| `--carryover-share` | 0.5 | Share of vials that follow a different recipe |
| `--residue-ml` | 5 30 | Range of leftover mL from the previous vial (APPROXIMATE) |
| `--max-residue-share` | 0.5 | Leftover is capped at this share of the vial |
| `--seed` | 0 | Same seed = same plan, so results can be repeated |
| `--subframes` | 32 | Render quality; lower is faster but noisier |
| `--renderer` | PathTracing | Or `RayTracedLighting` if PathTracing gives empty images |
| `--no-labels` | off | Skip masks and boxes |
| `--styles` | `clean,camera` | Which renderings to save: `clean`, `camera`, or both. Both cost the same GPU time (one render) |
| `--opacity-mode` | `fixed` | `fixed` = organizers' 0.6 (clean look); `by-dye` = paler mixes more see-through (the older look) |
| `--dye-jitter` | 0 (off) | Random dye shade **per vial**: each dye's absorption is scaled per color channel by a random factor in 1 ± j, brightness by 1 ± j/2 (e.g. `0.2`). So the model can't memorize one exact twin shade. 0 = identical to earlier runs |
| `--pos-jitter` | 0 (off) | Random shift of the whole picture **per vial**, up to this many pixels (e.g. `4`), like the vial stopping slightly off its usual spot. Masks and `vial_box` shift with it |
| `--dry-run` | off | Plan only; no Isaac Sim |
| `--twin` | `../../csi_digital_twin_v1` | The organizers' twin folder (on YAAD: `/job/in`) |

---

## Outputs
```
out/
├── labels.csv      one row per picture
├── run_info.json   settings used (for repeating the run)
├── images/clean/   s0001_p1_v1.png  = sample 1, pour 1, variation 1, clean
├── images/camera/  s0001_p1_v1.jpg  = the same render, camera look
└── masks/          s0001_p1_v1.png  (one per render, shared by both styles); pixel values = class ids
```
**Mask class ids:** 1 vial · 4 carrier · 5 track · 6 nozzle · 7 background · 8 label (barcode) · 9 liquid · 10 stream · 0 other.

### `labels.csv`: one row = one saved picture (one render style) of one pour
| Column | Meaning | Example |
|---|---|---|
| `render_style` | `clean` or `camera` | `clean` |
| `image`, `mask` | File paths | `images/clean/s0002_p1_v1.png`, `masks/s0002_p1_v1.png` |
| `sample_id`, `variation` | Which vial, and which lighting/camera version | `2`, `1` |
| `recipe_id`, `recipe_source` | Recipe (`5`–`14` or `N001`…) and `real` / `random` | `5`, `real` |
| `t200_ml`, `t400_ml`, `t100_ml`, `total_ml` | The recipe's intended mL | `10, 10, 10, 30` |
| `dispense_order` | Pour order | `T200>T400>T100` |
| `previous_recipe`, `carryover`, `residue_ml` | The vial before, whether it differs (1 = carryover), and mL of leftover | `N001`, `1`, `12.0` |
| `stage`, `stage_tank`, `stages_total` | Which pour this picture is after | `1`, `T200`, `3` |
| `cum_ml`, `cum_orange_ml`, `cum_blue_ml`, `cum_clear_ml` | Liquid in the vial so far, leftover included | `18.0`, `6.0`, `5.0`, `7.0` |
| `orange_frac`, `blue_frac` | Dye shares in the vial so far | `0.333`, `0.278` |
| `settled` | Always 1: liquid shown fully mixed | `1` |
| `frame`, `liquid_scale_z` | Always frame `686` (settled shape); vertical scale = mL so far ÷ 70 | `686`, `0.257` |
| `diffuse_r/g/b`, `opacity` | Liquid material color used; opacity fixed at the organizers' 0.6 | `0.125, 0.098, 0.046`, `0.6` |
| `light_mode`, `light_gain`, `light_tint` | This vial's lighting (same for all its pours) | `random`, `1.199`, `-0.012` |
| `exposure_gamma`, `exposure_gain`, `noise_sigma`, `blur` | This vial's camera look, applied **only to the camera style** (same for all its pours; grain differs per picture) | `1.3, 0.85, 2.4, 1` |
| `dye_orange_scale`, `dye_blue_scale`, `dye_brightness` | This vial's random dye shade (`--dye-jitter`); `1.0 1.0 1.0` / `1.0` when off | `1.057 0.999 1.102` |
| `shift_x`, `shift_y` | This vial's picture shift in pixels (`--pos-jitter`); 0 when off | `-1`, `2` |
| `panel_r`, `panel_g`, `panel_b` | Measured color of the white panel (x 60–250, y 60–200) in **this** picture (clean and camera differ). Shows the lighting | filled when rendered (≈ `210, 211, 209` in the seed-4 test) |
| `vial_box` | Vial box `x0 y0 x1 y1` in pixels (blank if labels were off). No liquid box (decision: not needed now) | `272 27 348 459` |
| `color_model`, `seed` | Color model version; seed to remake this picture | `approx_v1` |

---

## 🎯 Intended model: check after EVERY pour

**Question the model answers:** *"Does the mix so far match what it should be at this pour?"*
- **Input:** the picture after a pour + the recipe + which pour just finished (`recipe_id`, tank mL, `stage`, `stage_tank`)
- **Output:** match / doesn't match (y = `carryover`)
- **The final-vial check is simply the last pour.** One model covers every pour, including the end.
- **Why:** it catches carryover at pour 1 and says which pour went wrong.
- **Baseline to compare against:** an end-only model that judges only the finished vial.

## ⚠️ Using this data for machine learning: inputs (X), answers (y), and what the model must NEVER see

**X = the images** (or the color features the changeover-watch code measures from them). **y = labels from `labels.csv`.** Not every column is a label. Some columns describe *how the picture was made*, and feeding them to the model gives away the answer ("leakage").

| Group | Columns | Rule |
|---|---|---|
| **y: targets** | `carryover` (0/1, **the main target**) · strength = `residue_ml` ÷ `cum_ml` (as a target only) · `previous_recipe` (for the cause) · `vial_box`, `mask` (for a box/outline detector, later) | What the model learns to predict |
| **Allowed inputs (known at runtime)** | `recipe_id`, `t200_ml`, `t400_ml`, `t100_ml`, `total_ml`, `dispense_order`, `stage`, `stage_tank`, `stages_total` · **`panel_r/g/b`** (measured from the picture, which the real camera can do too) | The real line knows these from the recipe queue, so the model may use them with the image |
| **⛔ Never inputs** | `diffuse_r/g/b`, `opacity`, `orange_frac`, `blue_frac`, `cum_ml`, `cum_orange_ml`, `cum_blue_ml`, `cum_clear_ml`, `residue_ml`, `frame`, `liquid_scale_z`, `render_style`, `light_mode`, `light_gain`, `light_tint`, `exposure_*`, `noise_sigma`, `blur`, `dye_*`, `shift_x`, `shift_y`, `color_model`, `seed`, `sample_id`, `variation` | Rendering settings. They leak the answer. Use them only for error analysis and reproducing pictures |

**Example:** recipe 7, after pour 1.
- **X:** the image + "recipe 7, pour 1 (T400)"
- **y:** `carryover = 1`, `previous_recipe = 6`
- **Must not see:** `diffuse_*` or `residue_ml`

**Splitting and testing rules:**
- **Split by `sample_id`.** All pours, variations **and both render styles** of one vial go in the same split. Never spread one vial across train and test.
- **The final score always comes from the REAL video-5 vials**, never from twin pictures.

---

## Known limits: check these in the first test render
- **Liquid colors are approximate.** They're based on dye values fitted to real video-5 vials, anchored to the organizers' blue for recipe 7. Check them by measuring test pictures with the changeover-watch color code, and compare with real clean vials.
- **The leftover amount (5–30 mL, capped at 50% of the vial) is an estimate,** from the real fill speed (~6 mL/s), how long vial 7's orange lasted (~4 s), and carryover fits of ~40–60% (probably inflated by color-model error). Pin it down by asking CSI how much the mixing path holds, or by running a pale recipe right after a dark one.
- **Liquid is always fully mixed.** Real fresh dye swirls for several seconds after a pour, so real mid-recipe frames look patchier.
- **Liquid height** comes from scaling the settled 70 mL shape up or down, an approximation of the real meniscus.
- **The liquid isn't in the masks.** It's tagged, but the renderer leaves see-through materials out of the labels. The vial and other parts are in. A liquid box or mask can be added later if a detector needs it.
- **The labeling code differs between Isaac Sim versions.** If it fails, the script prints a warning and saves pictures without masks and boxes instead of crashing.
- **One fill animation (recipe 7)** supplies every height, so the liquid's surface shape is the same for every recipe.

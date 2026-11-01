# How the Training Pictures Are Made (Digital Twin)

The model is trained only on pictures from the **digital twin**, the organizers' 3D copy of the fill station in NVIDIA Isaac Sim. The twin's fill camera (`FillCam`) sees the vial from the same angle as the real `vial_fill` camera.

The generator renders **one picture after each pour** of a recipe. It writes a CSV row for every picture with the right answers (the exact dye shares in the vial), which a real camera could never provide.

**Files** (in `version_1/twin_tools/`):

| File | What it is |
|---|---|
| `render_dataset.py` | The picture generator (runs in Isaac Sim 5.1) |
| `recipe_rules.json` | The recipe rules, the tank assignment and CSI's 10 real recipes, each rule with its source in CSI's documents. **Rules are changed here, not in the script.** |

The organizers' twin is **not copied** into this repo; the script points to it with `--twin`. Its scene files are never changed: every edit (liquid color and height, lights) is made in memory while rendering and thrown away afterwards.

---

## What happens for each vial

1. **A recipe is picked:** about half are CSI's 10 real recipes (5–14), half are random new recipes. Every recipe must pass `recipe_rules.json`: at most 80 mL in total, at least 10 mL from each tank used, at most 50 mL orange, no T300 (out of service).
2. **The vial before it is picked:**
   - **The same recipe** gives a *clean* vial.
   - **A different recipe** gives a *carryover* vial. About half the vials are carryover.
3. **Leftover liquid:** the station has a shared mixing path, so every vial starts with **5–30 mL of the previous vial's mix** (at most half the vial). The same amount of this vial's own mix stays behind for the next vial, taken evenly from every pour, so the total still matches the recipe.
   - **Clean vial:** the finished vial matches the recipe exactly. After the first pours it is a blend: after the orange pour it already holds some of the previous (same-recipe) mix.
   - **Carryover vial:** the finished vial is part old mix and part new recipe. For example, 20 mL of leftover in a 60 mL vial means one third is the previous recipe.
4. **The vial's look is set once:**
   - Lighting: brightness × 0.6–1.4 and a slight warm or cool tint.
   - Camera look: exposure, blur, noise level.
   - Dye shade (`--dye-jitter`) and vial position (`--pos-jitter`).

   Every pour of the vial shares these, as on a real line where the lights don't change mid-fill. The camera never moves.
5. **For each pour,** in CSI's order (orange T200 → blue T400 → clear T100, skipping tanks the recipe doesn't use):
   - The liquid **height** is set to the mL so far. The twin's finished-vial liquid shape (70 mL) is scaled down, so the surface is always flat and settled.
   - The liquid **color** is set to the mix so far, with the organizers' see-through level (opacity 0.6).
   - The **pouring stream is hidden** (the picture shows the pour finished and the liquid settled).
   - The fill camera renders the picture.
   - The **camera** style adds the real camera's look: darker exposure, slight blur, noise and a timestamp. The **clean** style keeps the render as it comes out of Isaac Sim.
   - The generator measures the **white panel** left of the vial (its color shows the lighting, as a real camera could measure it) and saves the picture, a **mask** (which pixels are vial, carrier, nozzle and so on) and **one CSV row**.

**Example:** recipe 8 (20 mL orange + 20 mL blue + 20 mL clear) gives 3 pictures: after orange, after blue, after clear.

---

## The data used to train the model

Four runs of 1,000 vials each, all in the camera style:

| Run | Seed | Dye shade varies | Position varies | Pictures |
|---|---|---|---|---|
| `train_camera_s1001` | 1001 | no | no | 2,255 |
| `train_jitter_s2001` | 2001 | yes (±20%) | yes (±4 px) | 2,209 |
| `train_jitter_s2002` | 2002 | yes (±20%) | yes (±4 px) | 2,206 |
| `train_jitter_s2003` | 2003 | yes (±20%) | yes (±4 px) | 2,234 |

**Total: 4,000 vials and 8,904 pictures.** The table `model/tables/train_all.csv` holds all of them (made by `model/build_table.py`). The pictures are on the YAAD workspace, not in git, because of their size.

---

## How to run it

**1. Plan only, on a laptop (no Isaac Sim).** This checks the recipes and writes the CSV without pictures:
```bash
cd version_1 && python3 twin_tools/render_dataset.py --dry-run --count 5 --variations 1 --seed 4 --out-dir dataset_plan
```

**2. Render on YAAD.** The workspace twin folder must hold the organizers' `cameras/` and `stations/fill_station/`. Put `render_dataset.py` and `recipe_rules.json` in its `tools/` folder, then submit a job with `input_path` = that twin folder:
- **Image:** `nvcr.io/nvidia/isaac-sim:5.1.0`
- **Environment:** `ACCEPT_EULA=Y`, `PRIVACY_CONSENT=Y`
- **Command (one training run):**
  ```bash
  cd /job/in && /isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 1000 --variations 1 --seed 2001 --styles camera --dye-jitter 0.2 --pos-jitter 4 --out-dir /job/out
  ```
- **Resources:** 48 GB memory, 8 CPUs.
- **Time limit:** up to 240 minutes (the slower Spark machines need it).

Start with a small test (`--count 5`) and look at every picture before rendering more.

**Speed:** about 105 s to start, then about 2 s per picture at the default quality (32 subframes). 1,000 vials is about 2,200 pictures, or 75 minutes on a fast machine.

**Options:**

| Option | Default | Meaning |
|---|---|---|
| `--count` | 20 | Number of vials |
| `--variations` | 2 | Versions of each vial, each under a new lighting and camera look |
| `--seed` | 0 | The same seed gives the same plan, so a run can be repeated. Use a new seed for each run |
| `--real-share` | 0.5 | Share of vials using CSI's real recipes 5–14 |
| `--carryover-share` | 0.5 | Share of vials that follow a different recipe |
| `--residue-ml` | 5 30 | Range of leftover mL from the previous vial |
| `--max-residue-share` | 0.5 | Leftover is at most this share of the vial |
| `--light-mode` | random | `random` per vial, or a fixed `neutral` / `dim` / `bright` / `warm` / `cool` |
| `--styles` | clean,camera | Which pictures to save: `clean`, `camera` or both (both cost one render) |
| `--dye-jitter` | 0 (off) | Random dye shade per vial: each dye's strength per color channel × a random 1 ± j (e.g. `0.2`) |
| `--pos-jitter` | 0 (off) | Random shift of the whole picture per vial, up to this many pixels (e.g. `4`); masks and the vial box shift too |
| `--subframes` | 32 | Render quality; lower is faster but noisier |
| `--renderer` | PathTracing | Or `RayTracedLighting` if PathTracing gives empty pictures |
| `--no-labels` | off | Skip masks and boxes |
| `--opacity-mode` | fixed | `fixed` = the organizers' 0.6; `by-dye` = paler mixes more see-through (not used) |
| `--dry-run` | off | Plan only; no Isaac Sim |
| `--twin` | `../../csi_digital_twin_v1` | The organizers' twin folder (on YAAD: `/job/in`) |

---

## What comes out

```
out/
├── labels.csv      one row per picture
├── run_info.json   the settings used (to repeat the run)
├── images/camera/  s0001_p1_v1.jpg = vial 1, pour 1, variation 1
├── images/clean/   (only if the clean style is on)
└── masks/          s0001_p1_v1.png: each pixel's value says what it shows
```
**Mask values:** 1 vial · 4 carrier · 5 track · 6 nozzle · 7 background · 8 label (barcode) · 9 liquid · 10 stream · 0 other. The see-through liquid is left out by the renderer, so the vial pixels (1) cover the glass and what is seen through it.

### `labels.csv`: one row per picture

| Column | Meaning | Example |
|---|---|---|
| `render_style` | `camera` or `clean` | `camera` |
| `image`, `mask` | File paths | `images/camera/s0002_p1_v1.jpg` |
| `sample_id`, `variation` | Which vial, and which lighting/camera version | `2`, `1` |
| `recipe_id`, `recipe_source` | Recipe (`5`–`14` or `N001`…) and `real` / `random` | `5`, `real` |
| `t200_ml`, `t400_ml`, `t100_ml`, `total_ml` | The recipe's mL of orange, blue, clear and in total | `10, 10, 10, 30` |
| `dispense_order` | Pour order | `T200>T400>T100` |
| `previous_recipe`, `carryover`, `residue_ml` | The vial before, carryover yes (1) or no (0), and mL of leftover | `N001`, `1`, `12.0` |
| `stage`, `stage_tank`, `stages_total` | Which pour this picture is after | `1`, `T200`, `3` |
| `cum_ml`, `cum_orange_ml`, `cum_blue_ml`, `cum_clear_ml` | Liquid in the vial so far, leftover included | `18.0`, `6.0`, `5.0`, `7.0` |
| `orange_frac`, `blue_frac` | **The true dye shares in the vial so far** (what the model learns to estimate) | `0.333`, `0.278` |
| `settled`, `frame`, `liquid_scale_z` | Always settled; the finished-vial shape (frame 686) scaled to mL so far ÷ 70 | `1`, `686`, `0.257` |
| `diffuse_r/g/b`, `opacity` | The liquid color given to the renderer | `0.125, 0.098, 0.046`, `0.6` |
| `light_mode`, `light_gain`, `light_tint` | This vial's lighting | `random`, `1.199`, `-0.012` |
| `exposure_gamma`, `exposure_gain`, `noise_sigma`, `blur` | This vial's camera look (camera style only) | `1.3, 0.85, 2.4, 1` |
| `dye_orange_scale`, `dye_blue_scale`, `dye_brightness` | This vial's dye shade; `1.0 1.0 1.0` / `1.0` when off | `1.057 0.999 1.102` |
| `shift_x`, `shift_y` | This vial's picture shift in pixels; 0 when off | `-1`, `2` |
| `panel_r`, `panel_g`, `panel_b` | Measured color of the white panel (x 60–250, y 60–200) in this picture | `210, 211, 209` |
| `vial_box` | The vial's box `x0 y0 x1 y1` in pixels | `272 27 348 459` |
| `color_model`, `seed` | Color model version, and the seed to remake this picture | `approx_v1` |

---

## Rules for using this data (no answer leakage)

Some columns describe *how the picture was made*. A model that sees them could read the answer from them instead of from the picture.

| Group | Columns | Rule |
|---|---|---|
| **Answers** | `orange_frac`, `blue_frac` (the main target), `carryover`, `previous_recipe`, `residue_ml` | Used only to train and score the model |
| **Allowed inputs** (known on a real line) | The picture; the recipe (`recipe_id`, `t200_ml`, `t400_ml`, `t100_ml`, `total_ml`, `dispense_order`, `stage`, `stage_tank`, `stages_total`); `panel_r/g/b` (measured from the picture) | The line knows the recipe from its queue, so the model may use it with the picture |
| **Never inputs** | `diffuse_r/g/b`, `opacity`, `cum_*`, `frame`, `liquid_scale_z`, `render_style`, `light_*`, `exposure_*`, `noise_sigma`, `blur`, `dye_*`, `shift_x`, `shift_y`, `color_model`, `seed`, `sample_id`, `variation` | Rendering settings. They give away the answer |

**Splitting:** all pictures of one vial stay on the same side (train or test). The final real-world check is on CSI's real video 5, which is never used for training or tuning.

---

## Known limits

- **The liquid colors are approximate.** They come from a dye model fitted to the 20 clean real vials of video 5, anchored to the organizers' blue for recipe 7. The twin's orange renders redder than the real camera. Random dye shades in training reduce the effect.
- **The leftover amount (5–30 mL) is an estimate.** It is based on the real fill speed (about 6 mL/s) and how long the leftover color shows in the real videos. CSI could confirm how much the mixing path holds.
- **The liquid is always fully mixed and settled.** Real fresh dye swirls for a few seconds after a pour.
- **The liquid height comes from one shape** (the twin's 70 mL fill, recipe 7) scaled up or down, so the surface looks the same for every recipe.
- **The liquid isn't in the masks.** The renderer leaves see-through materials out of its labels.
- **Labeling code differs between Isaac Sim versions.** If it fails, the script warns and saves pictures without masks instead of stopping.

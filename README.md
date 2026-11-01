# Vial Color Check for Fill Stations

**Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station 3**

A camera-based add-on for vial filling machines. It looks at each vial after filling, estimates how much orange and blue dye is in it, and compares that with the recipe. If the color is off, it raises an **ALERT**.

The main problem it catches is **carryover**. The station's tanks share one mixing path, so when the recipe changes, liquid left over from the previous recipe ends up in the next vial.

The model was trained only on pictures from a **digital twin** (a 3D simulation of the station). It was then tested once on a **real video** from CSI's test line (CSI = the UWM Connected Systems Institute, which runs the line). The model never trained on real video; the only real-video input was an early estimate of the twin's dye colors (see the disclosures under Results).

**Results at a glance:**

| | Digital-twin test (800 vials) | Real video 5 (29 scored vials) |
|---|---|---|
| Carryover vials caught | 336 of 377 (89%) | **8 of 9** |
| Clean vials wrongly flagged | 175 of 423 (41%) | **9 of 20** |
| How well it separates good from bad (AUC, 1.0 = perfect, 0.5 = guessing) | 0.90 | **0.82** |

---

## How it works

1. **Find each vial.** The program watches the fill camera. When the carrier (the holder that moves each vial along the track) moves, vials are changing. Movement inside the vial means a pour.
2. **Take a picture of the finished vial,** 1 second before the carrier moves it away.
3. **Estimate the mix.** A neural network looks at the picture and the recipe and estimates the share of orange dye and blue dye in the vial. For example: *about 35% orange, 34% blue*.
4. **Compare with the recipe.** The **gap** is the larger difference between the estimate and what a correct vial of that recipe looks like, for orange or for blue. Above **8 percentage points** the vial gets an **ALERT**; otherwise **OK**. For example: *"recipe 6 should be 25% orange and 0% blue; measured 35% / 34%; gap 34 points: ALERT."*

## What it needs on a real line

| Input | Format | Where it comes from |
|---|---|---|
| **Fill camera** | 640×480 video of the fill position, the same view as CSI's `vial_fill` camera (USB camera, network stream, or a recorded file) | A camera on the fill station |
| **Recipe queue** | A CSV with one row per vial, in order: `vial, recipe_id, t100_ml, t200_ml, t400_ml` (mL from each tank; blank = not used). This is the same format as CSI's `video5_vial_recipes.csv` | The operator, the production schedule or a production system export |
| **Tank assignment** | Which tank holds which color. On CSI's line: T200 = orange, T400 = blue, T100 = clear (diluent), T300 = out of service | Set once at installation; chosen on the dashboard's setup screen |

- **No connection to the machine's controls is needed.** It only needs the camera, the recipe queue and the tank assignment.
- **Any ordinary computer can run it.** The model checks a vial in well under a second on a normal laptop processor; no graphics card is needed at the line.
- **The recipe queue format is an assumption.** It was not confirmed with CSI, so the program uses CSI's own label format.

```
Fill camera ──────────────────┐
                              ├─► Computer running vial_check.py ─► Dashboard in a web browser: OK / ALERT per vial
Recipe queue (CSV) ───────────┤                                     Predictions saved to a CSV file
Tank assignment (setup) ──────┘
```

## The operator dashboard

```bash
cd version_1
.venv/bin/python model/vial_check.py --live
```
Then open **http://127.0.0.1:8000** in a web browser. To open it from another screen on the same network (for example, a tablet at the line), add `--host 0.0.0.0`.

**Setup screen:**
1. **Video:** choose **Upload a video** (an .mp4 from the fill camera), **Connect a camera** (a USB camera number such as `0`, or a stream address such as `rtsp://...`), or **Digital twin** (optional, see [below](#optional-the-digital-twin-as-a-video-source)).
2. **Recipe queue:** upload the CSV.
3. **Tank assignment:** pick which tank holds orange, blue and clear.
4. Click **Start check**.

**While it runs:**
- **The live camera picture,** with the vial area marked.
- **The vial being filled:** its recipe, the pours seen so far, and the status (pouring, waiting, carrier moving).
- **The last finished vial:** a green **OK** or red **ALERT: possible carryover** banner, its picture, orange and blue bars (measured vs expected) and the gap.
- **Totals:** vials checked and alerts.
- **A log** of every vial, newest first.
- **Stop** ends the run early; **New run** goes back to setup.

**Saved results:** every run gets its own folder, `model/results/live_<video name>_<date-time>/`:
- `vials.csv`: one row per vial with the measured and expected mix, the gap and OK/ALERT. It is updated after every vial, so a stopped run keeps everything up to that point.
- `snapshots/`: the picture of each vial.
- `recipe_queue.csv`: the queue that was used.

Uploaded videos play at real speed; add `--speed 10` to play them 10 times faster. Twin videos always play at normal speed.

**Scoring (optional):** if the recipe CSV also has CSI's `start_s, end_s` columns (when each vial was filled, in video seconds), vials are matched by time and the dashboard also shows carryover vials caught and clean vials flagged. Without them, vials are matched in the order they arrive.

## Using the alerts on the line

The program **only advises; it never controls the machine.** This is required for testing on CSI's line, where the plant controls must not be touched. People act on the alerts.

**For the operator at the station:**
1. **A red ALERT names the vial:** its number in the queue, its recipe, and how far its color is from the recipe (for example, *"vial 4, recipe 6: measured 35% orange / 34% blue, expected 25% / 0%"*).
2. **The operator pulls that vial** at the next stop (or marks it) and sets it aside for a second look, instead of letting it go on to capping and packing.
3. **The picture of the vial** is saved with the alert, so a supervisor or quality inspector can review the decision later.

**For the quality process:**
- **Every run leaves a record:** `vials.csv` lists each vial with OK/ALERT, the measured and expected mix and the gap. It can be handed to whoever decides what happens to flagged vials: re-test, rework, or discard. CSI's documents don't say what the line does with off-color vials, so that decision stays with the plant's own process.
- **Alerts point to the cause:** on CSI's line, alerts cluster on the first vial after a recipe change, which points to leftover liquid in the shared mixing path. A plant could use this to add a short flush between recipes, or to set aside the first vial after every change as a routine step.
- **Later, with the plant's approval:** the same OK/ALERT signal could drive a stack light or a reject gate through the machine's controller. That is not part of this project.

## Optional: the digital twin as a video source

Instead of a camera, the dashboard can ask the digital twin to **make a fill-camera video for any recipe queue**. This is an extra feature for testing and demos, not part of the main results.

1. The twin (`twin_tools/render_live_video.py`) runs in Isaac Sim 5.1 (NVIDIA's 3D simulation software). It needs an NVIDIA GPU, so here it runs in a workspace on YAAD (the GPU lab provided for the hackathon). The setup commands are at the top of the script.
   - **Before starting a run,** open the secure connection from the laptop to YAAD (an SSH tunnel), with the VPN on, and leave it open:
     ```bash
     ssh -N -L 20700:127.0.0.1:20700 <user>@<workspace address>
     ```
   - The dashboard reaches the twin at `http://localhost:20700` (already filled in on the setup screen).
2. On the setup screen, choose **Digital twin** and upload a recipe queue (5 examples are in `twin_tools/demo_queues/`).
3. **The twin decides on its own what happens on the line**, as the real station would:
   - **Carryover comes from the leftover physics:** the station keeps one leftover amount in its shared mixing path (5–30 mL, chosen per run), and the first vial after a recipe change picks up the previous recipe's mix.
   - **Pour speeds and waits vary** from vial to vial.
   - **The vial stops in a slightly different spot** each time.
   - **Now and then an empty carrier passes.**
4. The twin renders the whole video. This takes about 2 minutes per vial on the YAAD GPU, and the dashboard shows the progress. Then the video plays at normal speed with OK/ALERT for each vial.
5. The twin also saves `truth.csv` (the true mix and carryover of every vial) in its run folder on YAAD. It is **not** sent to the dashboard, so the check runs blind, and it can be compared with the dashboard's `vials.csv` afterwards.

---

## Results

### Real video 5 (final validation)

The trained model was run once on CSI's real video 5:
- 30 vials and 10 recipes, each recipe on 3 vials in a row.
- No adjustment to the real camera.
- Nothing was trained or tuned on this video. The one exception is the twin's dye colors, which were estimated from it earlier (see Disclosures).

**Scoring rule (an assumption):** CSI's labels don't say which vials have carryover. The first vial after a recipe change is counted as carryover and the next two as clean. The very first vial has nothing before it and is not scored. That leaves **29 scored vials: 9 carryover, 20 clean.**

| Measure | Result |
|---|---|
| Vials found | 30 of 30 (plus 1 empty carrier move, ignored because no recipe matched it) |
| Carryover vials caught | **8 of 9** |
| Clean vials wrongly flagged | **9 of 20** |
| AUC | **0.82** |
| Typical gap: carryover vs clean vials | **28 vs 7 points** |
| Flagged vials that really were carryover | **8 of 17 (47%)**, against 9 of 29 (31%) for vials picked at random |

**By recipe** (each recipe: the vial right after the change, then 2 clean vials)

| Recipe (mL orange / blue / clear) | Vial after the change | Clean vials flagged |
|---|---|---|
| 5 (10 / 10 / 10) | first vial, not scored | 0 of 2 |
| 6 (10 / 0 / 30) | caught | 2 of 2 |
| 7 (0 / 50 / 20) | caught | 2 of 2 |
| 8 (20 / 20 / 20) | caught | 0 of 2 |
| 9 (35 / 35 / 10) | **missed** (gap 7.3 points, just under 8) | 0 of 2 |
| 10 (15 / 15 / 45) | caught | 2 of 2 |
| 11 (10 / 50 / 10) | caught | 0 of 2 |
| 12 (50 / 10 / 10) | caught | 0 of 2 |
| 13 (50 / 20 / 0) | caught | 1 of 2 |
| 14 (0 / 10 / 60) | caught | 2 of 2 |

**What this shows:**
- **Carryover stands out on a real camera, without any real training pictures:** a typical gap of 28 points for carryover vials against 7 for clean vials.
- **The false alarms are a color bias for certain recipes, not random noise.** 8 of the 9 come from recipe 7 (blue only) and from recipes 6, 10 and 14, which are mostly clear water. On these the model reads more dye than the vial holds, on every vial of the recipe.
  - The mixed recipes 5, 8, 11 and 12 have no false alarms.
  - A short adjustment using a few known-good vials per recipe might reduce this bias. This was not tested and is not part of this version.
- **The one miss** is recipe 9 right after recipe 8. Both have equal orange and blue, so the leftover barely changes the color.

Every vial's estimate, gap and result: [`model/results/video5_validation/vials.csv`](model/results/video5_validation/vials.csv). All 30 vial pictures with their results: [`contact_sheet.jpg`](model/results/video5_validation/contact_sheet.jpg).

**Disclosures:**
- The carryover labels above are an assumption (the first vial after a recipe change), not CSI ground truth.
- Matching vials to the queue uses the times in CSI's label file.
- The twin's dye colors were estimated from video 5's clean vials before video 5 was set aside for validation. Random dye shades in training reduce the effect.
- During setup of the matching step, the results for 6 vials were seen once before the matching rule was fixed. The model, threshold and alert rule were not changed.

### Digital-twin test (800 vials never used in training)

The twin test runs on the **picture after every pour**, not just the finished vial.

| Measure | Result |
|---|---|
| Mix estimate error (orange / blue) | **2.5 / 2.1 percentage points** on average |
| AUC (every picture) | **0.90** |
| Off-recipe pictures caught | 82% |
| Carryover vials caught (any pour flagged) | **336 of 377 (89%)** |
| Clean vials wrongly flagged (any pour) | 175 of 423 (41%) |
| AUC with the liquid or the whole picture recolored (worst of 10 color shifts) | **0.90** (no real drop) |

**By pour:**

| After pour | Off-recipe pictures caught | Clean pictures wrongly flagged |
|---|---|---|
| 1 | 88% | 40% |
| 2 | 80% | 8% |
| 3 | 69% | 2% |

- **Most false alarms come after pour 1.** At that point the unknown amount of leftover blurs what "correct" looks like.
- **Ignoring pour-1 alerts** (counting only vials with 2 or more pours) gives 225 of 323 carryover vials caught (70%) with only 30 of 386 clean vials flagged (8%).

Full numbers: [`model/results/twin_test/metrics.json`](model/results/twin_test/metrics.json); every test prediction: [`test_predictions.csv`](model/results/twin_test/test_predictions.csv).

---

## The model

**Type:** a ResNet-18 image network (pretrained on the general ImageNet photo collection, then trained on the twin vials), combined with the recipe.

**Inputs:**
- **The picture:** one 640×480 frame of the vial.
- **The recipe:** 10 numbers, all known from the recipe queue:
  - the dye shares a correct vial should have, with a typical 17.5 mL of same-recipe leftover;
  - the plain recipe shares;
  - which pour, and how many pours in all;
  - mL from each tank, and the total.

**Picture preparation** (the same in training and on the real camera):
1. Cut out a 140×320-pixel area around the vial (the camera doesn't move).
2. Divide every pixel by the color of the white panel left of the vial. Darker or warmer lighting changes both equally, so it cancels out.

**How it's built:**
```
vial picture ──► ResNet-18 ──► 512 picture features ─┐
                                                     ├─► 128 ─► 2 numbers: orange share, blue share (0–1)
recipe (10 numbers) ─────────────────────────────────┘
```

**Training:**
- **Data:** 8,904 twin pictures of 4,000 vials, split by vial: 80% to learn from and 20% for the final test. 15% of the learning vials are held back to pick the best round and the 8-point threshold.
- **Variety added during training:** each picture's liquid shade is changed at random (each color × 0.85–1.15) and the vial is shifted up to 6 pixels, so the model can't depend on the twin's exact colors or position.
- **Settings:** 25 rounds through the data; the network is adjusted to reduce the average error of its estimates (AdamW optimizer, learning rate 3e-4).
- **Hardware and time:** one NVIDIA DGX Spark GPU computer on YAAD, about 7 minutes.

---

## Known limits

- **Twin colors don't exactly match the real camera.** The twin's orange renders redder. Random shades in training help, but on video 5 this shows up as a color bias on pale and blue-only recipes (see above).
- **Only the finished vial is checked on real video.** The twin test covers every pour; the real check uses the finished-vial picture, which the video gives reliably for every vial.
- **One real video** (30 vials, 10 recipes) was used for validation. CSI's videos 1–4 have no recipe queue, so they can't be scored.
- **In the twin, the liquid is always fully mixed and settled,** and carryover is the only fault simulated.
- **The camera view is fixed:** the program expects the 640×480 `vial_fill` view, with the vial in the same place.

---

## Running everything

Python 3 with `numpy`, `pandas`, `opencv-python`, `torch` and `torchvision`. The project's environment is `version_1/.venv`.

```bash
# 1. Make twin pictures (Isaac Sim 5.1, e.g. on YAAD; details in docs/DATASET_GENERATION.md).
#    Copy twin_tools/render_dataset.py and recipe_rules.json into the twin folder's tools/ first, then from the twin folder:
/isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 1000 --variations 1 --seed 2001 \
    --styles camera --dye-jitter 0.2 --pos-jitter 4 --out-dir /job/out

# 2. Turn the pictures into one table (recipe information, measured colors, right answers, train/test split)
.venv/bin/python model/build_table.py <twin run folders...> --out model/tables/train_all.csv

# 3. Train and test the model (GPU) -> model/results/twin_test/
python model/train_model.py --table model/tables/train_all.csv --pretrained --run <name>=<folder> ...

# 4a. Operator dashboard
.venv/bin/python model/vial_check.py --live

# 4b. Or check a whole recorded video at once (summary + picture sheet of all vials)
.venv/bin/python model/vial_check.py --video <video.mp4> --recipes <recipe queue CSV> --out-dir <folder>
```

The trained model file (`model/results/twin_test/model.pt`, 43 MB) is not in git. Step 3 recreates it, or it is available on request.

## What's in this repo

| Path | What it is |
|---|---|
| `model/vial_check.py` | **The program:** finds vials, runs the model, and shows the operator dashboard |
| `model/train_model.py` | Trains and tests the model on twin pictures |
| `model/build_table.py` | Turns twin pictures into the training table |
| `model/tables/train_all.csv` | The training table: 8,904 pictures of 4,000 twin vials |
| `model/results/twin_test/` | Twin test results: metrics, every test prediction, the training log |
| `model/results/video5_validation/` | Real video 5 results: every vial's result, pictures, summary |
| `twin_tools/render_dataset.py` | Makes training pictures with the digital twin (Isaac Sim) |
| `twin_tools/render_live_video.py` | Optional: the digital twin as a video source for the dashboard (makes a fill-camera video for a recipe queue) |
| `twin_tools/demo_queues/` | 5 example recipe queues (CSI's recipes 5–14) for the twin |
| `twin_tools/recipe_rules.json` | Recipe rules, tank assignment and CSI's 10 real recipes |
| `docs/DATASET_GENERATION.md` | How the twin pictures are made, and every column of their labels |
| `samples/` | 24 example twin pictures; `samples.csv` lists recipe, pour and carryover for each |

## Data

- **CSI's data and digital twin** (`csi-vial-fill-dataset`, `csi_digital_twin_v1`): from the Innovation Lab data folder.
- **Real videos are for testing only, never training.** Video 5 is the final validation.
- **Twin pictures** (8,904 pictures from 4 runs of 1,000 vials, with masks and labels): on the YAAD workspace, not in git because of their size. Available on request.

## Credits

- **Data, digital twin and line:** Rockwell Automation, NVIDIA, UWM Connected Systems Institute (CSI), MSOE AI Club.
- **Compute:** YAAD DGX Spark cluster.
- **Tools:** NVIDIA Isaac Sim 5.1 and Omniverse Replicator; NVIDIA PyTorch container; ResNet-18 (torchvision, ImageNet weights); PyTorch, OpenCV, pandas, NumPy.

### AI assistance

**Claude Code** (Anthropic) was used as an AI coding assistant:
- **Code:** it wrote and tested the code in this repo (twin picture generator, table builder, model training, vial check and dashboard), following the project's decisions.
- **Checks:** it reviewed rendered pictures and data quality, enforced the project rules (no answer leakage into model inputs, split by vial, video 5 kept for final validation), and re-checked results.
- **Docs:** it helped write the documentation and the YAAD job settings.

Project decisions (what to build, the data plan, which model to use, which runs to launch) were made, and all GPU jobs were run, by the project author.

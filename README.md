# Pour-by-Pour Color Check for Vial Filling

**Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station 3**

A camera-only add-on for older filling machines. After each pour, it checks whether the vial's color matches what the recipe should look like at that point, and alerts the operator when it doesn't. The main fault it targets is **carryover**: leftover liquid from the previous recipe in the shared mixing path that ends up in the next vial.

> Status: work in progress (branch `ap`). Trained and tested on digital-twin data; validated once on real video 5 (see [Validation on real video 5](#validation-on-real-video-5)).

---

## How it works

1. **Training data from the digital twin.** The organizers' Isaac Sim fill station is rendered from the same angle as the real fill camera, with **one picture after each pour**. Each vial gets a real or random valid recipe, and about half follow a different recipe (carryover). Lighting, camera look, dye shade and vial position vary per vial. Every picture is labeled for free.
2. **An image network estimates the mix.** For each picture after a pour, the model looks at the vial and the recipe and estimates the orange and blue shares in the vial.
3. **Compare with the recipe.** If the estimate is far from what a clean vial of that recipe should have at that pour, it **flags it**. For example: *"pour 1: about 66% orange, a clean vial would be 92%"*.
4. **On the real video**, a motion detector finds each vial and takes a snapshot of the finished vial. No calibration is used: the model trained on the twin runs on the real camera as is.

## What it needs on a real line (assumptions)

The model reads the mix from the camera. To say whether that mix is *wrong*, the program also needs to know what each vial should be. These inputs were not confirmed with CSI and are **assumed** to arrive as follows:

| Input | Assumed format | Real-world source | Set up |
|---|---|---|---|
| **Vial camera** | 640×480 video of the fill position, the same view as CSI's `vial_fill` camera | A camera on the fill station | Once |
| **Recipe queue:** which recipe each vial gets, in order | CSV rows of `vial, recipe_id, t100_ml, t200_ml, t400_ml, dispense_order`, the same columns as CSI's `video5_vial_recipes.csv` | The operator, the production schedule or an MES export | Per batch |
| **Tank assignment:** which tank holds which color | `T100` = clear (diluent), `T200` = orange, `T400` = blue, `T300` = out of service; kept in `twin_tools/recipe_rules.json` | Set at install | Once |

- **No connection to the machine's controls (PLC) is needed:** one camera, the recipe queue and the tank assignment.
- **A tank camera is not needed.** It is an optional upgrade: it could warn about low or empty tanks, catch a tank refilled with the wrong dye, and recalibrate the expected colors automatically. The digital twin doesn't model the tanks, so this is future work.

## The model: image network + recipe

A **ResNet-18** convolutional neural network (pre-trained on ImageNet, fine-tuned on the twin vials) looks at the vial crop, combines it with the recipe, and estimates the orange and blue shares. A rule then compares the estimate with the expected mix. It was chosen after comparing it with a color-measurement model, an image-only version of the same network and NVIDIA Cosmos Reason on the same test pictures: it had the best overall results and fits the solution, since the line already knows each vial's recipe.

### Inputs

| Input | What it is |
|---|---|
| **Picture** | One 640×480 frame taken after a pour |
| **Recipe vector (10 numbers)** | Expected orange and blue shares for a clean vial (with a typical 17.5 mL leftover); the plain-recipe expected shares; pour number ÷ 3; total pours ÷ 3; mL from T200, T400, T100 and the total, each ÷ 80. All known from the recipe at runtime. |
| **Target (what it learns)** | The true orange and blue shares in the vial, known exactly in the twin |

### Picture transformations

1. **Cut out the vial region:** a fixed box around the vial (x 230–390, y 145–480); the camera doesn't move.
2. **Measure the white panel:** the median color of a white patch left of the vial (x 60–250, y 60–200).
3. **Crop 140×320 pixels** around the vial (no resizing).
4. **Divide every pixel by the panel color,** per channel. A darker or warmer camera shifts the vial and the panel equally, so lighting and white balance cancel out.
5. **During training only**, new random changes every time a picture is used:
   - **Random liquid shade:** each color channel of the vial pixels × a random factor between 0.85 and 1.15, so the network can't memorize the twin's exact dye colors (vial pixels come from the twin's mask, used for training only).
   - **Random shift:** the crop moves up to ±6 pixels, like a vial stopping slightly off its usual spot.
6. **Result:** a 3×320×140 array of numbers (roughly 0–1.5).

### Network

```
vial crop (3×320×140) ──► ResNet-18 ──► 512 image features ─┐
                                                           ├─► 522 ─► Linear(128) ─► ReLU ─► Linear(2) ─► sigmoid
recipe vector (10) ────────────────────────────────────────┘
                                                                        ↓
                                                    estimated orange share, blue share (0–1)
```

### Training

- **Data:** 4 twin runs, 4,000 vials, 8,904 pictures; 3 runs have random dye shades. Split **80/20 by vial** (all pours of a vial on one side), with 15% of the training vials held out for validation: **6,055 pictures to fit, 1,035 to validate, 1,814 to test.**
- **Loss:** mean absolute error between estimated and true shares.
- **Optimizer:** AdamW, learning rate 3e-4 on a cosine schedule, batches of 64, 25 epochs. The epoch with the best validation error is kept.
- **Hardware:** one YAAD DGX Spark GPU, NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:25.10-py3`): about 15 seconds per epoch, **7 minutes in total.**

### From estimate to flag

1. **Gap** = the larger of |estimated orange − expected orange| and |estimated blue − expected blue|, against a clean vial of that recipe at that pour.
2. **Flag** when the gap is above the **threshold**, picked on the validation pictures (best F1): **8 points.**

## Results (twin test set: 800 vials, 1,814 pictures never used in training)

| Measure | Result |
|---|---|
| AUC (how well the gap separates bad from fine pictures) | **0.90** |
| Mix estimate error (orange / blue) | **2.5 / 2.1 percentage points** |
| Off-spec pictures caught (recall) | 82% |
| Carryover vials caught (any pour flagged) | **336 of 377 (89%)** |
| Clean vials falsely flagged (any pour) | 175 of 423 (41%) |
| AUC with the liquid's colors shifted (worst case) | **0.90** (no real drop) |

**By pour:**

| Pour | Off-spec caught | Clean pictures falsely flagged |
|---|---|---|
| 1 | 88% | 40% |
| 2 | 80% | 8% |
| 3 | 69% | 2% |

**Pours 2+ only** (pour 1 left to the end check): **225 of 323 carryover vials caught (70%)**, with **30 of 386 clean vials falsely flagged (8%)**.

- **Checking after every pour catches more carryover** than checking only the finished vial.
- **Most false alarms come from pour 1,** where the unknown leftover amount blurs what "clean" looks like. Skipping pour-1 alarms gives the best balance; knowing the station's real leftover volume would cut them further.
- **The recipe alone can't predict the answer** (a model with no image scored AUC 0.54), so the twin data doesn't give it away.

Full metrics and every test prediction: `model/results/model2/`. The model's weights (`model.pt`, 43 MB) are not in git.

## Validation on real video 5

The frozen model (trained only on twin data, threshold 8 points, **no calibration**) was run once on CSI's real video 5 with `real_check.py`. Video 5 was not used to train, tune or pick anything, apart from the early dye-color estimate noted under Disclosures.

**Setup**
- **Vials:** a motion detector finds each vial (carrier move, then pours) and takes a snapshot 1 s before the carrier moves the finished vial away. The check is on the **finished vial** (the last pour), since that is the picture the video gives reliably for every vial.
- **Recipe queue:** CSI's `video5_vial_recipes.csv` (30 vials, 10 recipes, each recipe on 3 vials in a row).
- **Matching:** each queue vial is matched to the detected vial whose time window overlaps its label window the most.
- **Ground truth (assumption):** CSI's labels don't say which vials have carryover. The first vial after a recipe change is counted as carryover and the others as clean. The very first vial has no previous recipe and is not scored, so **29 vials are scored: 9 carryover, 20 clean.**

**Results**

| Measure | Result |
|---|---|
| Vials found and matched | 30 of 30 (plus 1 false detection, dropped by matching) |
| Carryover vials caught | **8 of 9** |
| Clean vials falsely flagged | **9 of 20** |
| AUC (gap separating carryover from clean) | **0.82** |
| Precision / recall / F1 | 0.47 / 0.89 / 0.62 |
| Median gap: carryover vs clean | **28 vs 7 points** |

**By recipe** (each recipe: vial 1 after the change, then 2 clean vials)

| Recipe (mL orange / blue / clear) | Carryover vial | Clean vials flagged |
|---|---|---|
| 5 (10 / 10 / 10) | first vial, not scored | 0 of 2 |
| 6 (10 / 0 / 30) | caught | 2 of 2 |
| 7 (0 / 50 / 20) | caught | 2 of 2 |
| 8 (20 / 20 / 20) | caught | 0 of 2 |
| 9 (35 / 35 / 10) | **missed** (gap 7.3, just under 8) | 0 of 2 |
| 10 (15 / 15 / 45) | caught | 2 of 2 |
| 11 (10 / 50 / 10) | caught | 0 of 2 |
| 12 (50 / 10 / 10) | caught | 0 of 2 |
| 13 (50 / 20 / 0) | caught | 1 of 2 |
| 14 (0 / 10 / 60) | caught | 2 of 2 |

- **The model separates carryover from clean on real video** without seeing any real data in training (AUC 0.82; the twin test gave 0.90).
- **False alarms are a per-recipe color bias, not random noise.** 8 of the 9 come from recipes 6, 10 and 14 (mostly clear water) and recipe 7 (blue only): the model reads more dye than these vials hold, on every vial of the recipe. The mixed recipes 5, 8, 11 and 12 have no false alarms.
- **The missed vial** is recipe 9 right after recipe 8. Both have equal orange and blue, so the leftover barely changes the color.
- A per-recipe calibration with a few clean vials could reduce these false alarms; it was not used here.

Every vial's estimate, gap and flag: `model/results/real_vial_fill_video_5/vials.csv`; snapshots of all 30 vials: `contact_sheet.jpg` in the same folder.

**Disclosures**
- The carryover labels above are an assumption (first vial after a recipe change), not CSI ground truth.
- Matching uses the start and end times in CSI's label file. On a real line the queue would give the order only; the script falls back to order matching when the times are missing.
- During development of the matching step, the results for 6 vials were seen once before the matching rule was fixed (the first rule matched too few vials). The model, threshold and flag rule were not changed.
- The twin's dye colors were first estimated from video-5 vials, before video 5 was reserved for validation (see Known limits).

## Known limits

- **Twin colors aren't calibrated to the real camera** (orange renders too red). Random dye shades in training make the model hold up under color shifts, but no calibration to the real camera is used. On video 5 this shows up as a per-recipe color bias (see Validation on real video 5); a calibration step with a few clean vials per recipe is possible future work.
- The twin's dye colors were first estimated from video-5 vials, before video 5 was reserved for validation. Random dye shades dilute this, and it will be reported with the final results.
- In the twin, liquid is always fully mixed and settled; carryover is the only fault simulated.
- Real-video validation covers one video (30 vials, 10 recipes) and checks only the finished vial.

## How to run

Python 3 with `numpy`, `pandas`, `opencv-python`, `torch`, `torchvision` (training tested in the `nvcr.io/nvidia/pytorch:25.10-py3` container on a GPU; that container needs `pip install pandas`).

```bash
# 1. Plan a twin dataset on a laptop (no Isaac Sim; writes labels.csv only)
python twin_tools/render_dataset.py --dry-run --count 20 --out-dir dataset_plan

# 2. Render it (Isaac Sim 5.1, e.g. on YAAD, from the twin folder)
/isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 1000 --variations 1 --seed 2001 \
    --styles camera --dye-jitter 0.2 --pos-jitter 4 --out-dir /job/out

# 3. Pictures -> table (split, recipe inputs and labels)
python model/build_table.py twin_runs/<run>/out [more runs...] --out model/tables/train_all.csv

# 4. Train and test the model (GPU) -> model/results/model2/
python model/train_cnn.py --table model/tables/train_all.csv --pretrained

# 5. Check a real video with the trained model -> model/results/real_<video name>/
python model/real_check.py --video <video.mp4> --recipes <recipe queue CSV>
```

`real_check.py` works on any video from the fill camera: only `--video` and `--recipes` change. If the recipe queue has `start_s`/`end_s` columns (as CSI's labels do), vials are matched by time and the run is scored; otherwise vials are matched in order and it only reports estimates and flags.

| Folder | Contents |
|---|---|
| `twin_tools/` | `render_dataset.py` (twin picture generator), `recipe_rules.json` (recipe rules, tank assignment, the 10 real recipes) |
| `model/` | `build_table.py` (pictures → table), `train_cnn.py` (the model), `real_check.py` (runs the model on a real video); `tables/` and `results/` |
| `samples/` | 24 example twin pictures; `samples.csv` lists recipe, pour and carryover for each |
| `docs/DATASET_GENERATION.md` | How the twin data is made, every label column, and which columns may be model inputs |
| `PROCESS.md` | Plan and status |

## Data

- **The organizers' data and digital twin** (`csi-vial-fill-dataset`, `csi_digital_twin_v1`): the Innovation Lab data folder. Real videos are used for testing only, never for training.
- **Generated twin pictures** (8,904 pictures from 4 runs of 1,000 vials, plus masks and labels): on the YAAD workspace, not in git because of their size. Access on request.
- **Data plan:** train/test on twin pictures (80/20, split by vial); final validation once on real video 5. Real videos 1–4 have no recipe queue, so they are not scored.

## Credits

- **Data, digital twin and line:** Rockwell Automation, NVIDIA, UWM Connected Systems Institute (CSI), MSOE AI Club.
- **Compute:** YAAD DGX Spark cluster.
- **Tools and models:** NVIDIA Isaac Sim 5.1 and Omniverse Replicator; NVIDIA PyTorch container; ResNet-18 (torchvision, ImageNet weights); NVIDIA Cosmos-Reason1-7B (compared); scikit-learn, OpenCV, pandas, NumPy.

### AI assistance

**Claude Code** (Anthropic) was used as an AI coding assistant. Its role:
- **Wrote and tested the code** in this repo (the twin generator, table builder and models), following the project's decisions on the data plan, labels and rules.
- **Checked data quality:** reviewed rendered pictures, found and fixed problems (for example, an unfair "expected color" reference at pour 1 and a too-small color patch), and flagged biases in the twin data.
- **Enforced the project rules:** no label leakage into model inputs, split by vial, and video 5 kept for final validation.
- **Ran independent verification:** separate AI reviewers re-computed the table, labels and metrics.
- **Helped plan and document:** process outline, docs and YAAD job settings.

Project decisions (what to build, the data plan, which model to use, which runs to launch) and all GPU jobs were made and run by the project author, not the AI.

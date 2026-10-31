# Pour-by-Pour Color Check for Vial Filling

**Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station 3**

A camera-only add-on for older filling machines. After each pour, it checks whether the vial's color matches what the recipe should look like at that point, and alerts the operator when it doesn't. The main fault it targets is **carryover**: leftover liquid from the previous recipe in the shared mixing path that ends up in the next vial.

> Status: work in progress (branch `ap`). Results below are on digital-twin test data only. Real-video testing is next.

---

## How it works

1. **Training data from the digital twin.** The organizers' Isaac Sim fill station is rendered from the same angle as the real fill camera, with **one picture after each pour**. Each vial gets a real or random valid recipe, and about half follow a different recipe (carryover). Lighting, camera look, dye shade and vial position vary per vial. Every picture is labeled for free.
2. **Break each picture into a table.** The liquid color is measured in 8 strips down the vial, divided by the white panel's color so lighting cancels out, and combined with the recipe and pour number.
3. **Estimate the mix, then compare.** A model estimates the orange and blue shares in the vial from the colors. If the estimate is far from what a clean vial of that recipe should have at that pour, it **flags it**. For example: *"pour 1: about 66% orange, a clean vial would be 92%"*.
4. **On the real line** (planned), a pour-stop detector takes the snapshot after each pour. A short calibration period with a few clean vials per recipe lines the model up with the real camera.

## The models

Both models answer the same question, *"is the mix wrong for this pour?"*, and use the same flag rule: estimate the orange and blue shares, then compare them with what a clean vial of that recipe should have at that pour. They differ in **how they look at the picture**.

| | **Model 1: color model** | **Model 2: image network (CNN)** |
|---|---|---|
| What it sees | About 40 hand-picked measurements: the average color of 8 strips down the vial, a bottom patch, and the white panel | The whole vial crop (140×320 pixels) |
| What decides what matters | The chosen measurements: anything not measured can't be used | The network learns its own features (color, liquid height, shading, edges) |
| Model | Gradient boosting on a table (scikit-learn) | ResNet-18, pre-trained on everyday photos (ImageNet), then trained on the twin vials |
| Handling color differences | Trained on runs with random dye shades | The same, plus fresh random shades and vial shifts on every training pass |
| Runs on | A laptop, about 1 minute | A GPU (YAAD DGX Spark), about 1–2 hours |
| Explainable | Yes: the measured colors are visible | Less so, but it still outputs "estimated 66% orange vs 92% expected" |

**How they're built:**
1. Twin pictures are turned into a table (`build_table.py`). The 4,000 vials are split 80/20 **by vial**, so every pour of a vial is on the same side.
2. Each model learns the true orange and blue shares from the training vials. The flag threshold is picked on training data only (model 1 uses cross-validation; model 2 holds out 15% of the training vials).
3. Test vials are scored once.
4. **Robustness test:** the test pictures are re-scored with the liquid's colors shifted (each color channel ±10%, darker or lighter) and with camera-wide shifts. This mimics the real camera seeing different colors than the twin.
5. **Bias check:** a model that sees only the recipe, never the image, should score near chance.

Model inputs follow a strict rule: only the image and recipe information the line knows at runtime. Columns that describe how a picture was rendered are never inputs (see `docs/DATASET_GENERATION.md`).

## Results so far (twin test set: 800 vials never used in training)

Both models trained on all 4 runs (4,000 vials, 8,904 pictures; 3 of the runs have random dye shades) and scored on the **same** 800 test vials:

| Model | AUC | Mix estimate error (orange / blue) | Carryover vials caught | Clean vials falsely flagged |
|---|---|---|---|---|
| **Model 2: CNN (ResNet-18) + rule** | **0.90** | **2.5 / 2.1 points** | **336 of 377** | 175 of 423 |
| Model 1: gradient boosting (direct) | 0.87 | — | 317 of 377 | 145 of 423 |
| Model 1: mix estimate + rule | 0.87 | 4.3 / 3.5 points | 329 of 377 | 165 of 423 |
| End-only baseline (last pour only) | 0.87 | — | 222 of 377 | 34 of 423 |
| Recipe only, no image (bias check) | 0.54 | — | — | — |

**Robustness** (AUC with the liquid's colors shifted; worst case shown):

| Model | No shift | Worst liquid shift |
|---|---|---|
| Model 1, run 1 only (no random dye shades) | 0.90 | 0.72 (green ×0.9) |
| Model 1, all runs, mix estimate + rule | 0.86 | 0.76 (green ×0.9) |
| Model 1, all runs, gradient boosting | 0.87 | 0.84 |
| **Model 2, all runs** | **0.90** | **0.90** |

**Pours 2+ only** (pour 1 left to the end check):

| Model | Clean vials falsely flagged | Carryover vials caught |
|---|---|---|
| **Model 2** | **30 of 386 (8%)** | **225 of 323 (70%)** |
| Model 1, mix estimate + rule | 54 of 386 (14%) | 223 of 323 (69%) |

- **Model 2 is the best so far:** the most accurate mix estimate, the highest AUC, and almost no loss under color shifts. It trains in about 7 minutes on one GPU.
- **Training on random dye shades made both models more robust to color differences.** Caveat: the robustness test shifts colors the same way model 2's training augmentation does, so its flat result partly reflects what it practiced. Real video is the true test.
- **Checking after every pour catches more carryover** than checking only the finished vial (336 vs 222 of 377), at the cost of more false alarms.
- **Most false alarms come from pour 1,** where the unknown leftover amount blurs what "clean" looks like (40% of clean first pours are flagged by model 2). Skipping pour-1 alarms gives the best balance. Knowing the station's real leftover volume would cut them further.
- The recipe alone can't predict the answer (AUC 0.54), so the twin data doesn't leak it.
- Model 1's results were independently re-checked three times by separate AI reviewers (see "AI assistance").

Full metrics: `model/results/model2/` (model 2), `model/results/model1_all/` (model 1, all runs) and `model/results/model1/` (model 1, run 1 only). Model 2's weights (`model.pt`, 43 MB) are not in git.

## Known limits

- **Twin colors aren't calibrated to the real camera** (orange renders too red). Random dye shades in training reduce the damage from color differences (see Robustness), and a short calibration period on the real line is planned to line the colors up.
- The twin's dye colors were first estimated from video-5 vials, before video 5 was reserved for validation. Random dye shades dilute this, and it will be reported with the final results.
- In the twin, liquid is always fully mixed and settled; carryover is the only fault simulated.
- Not yet tested on real video. Twin vials without random dye shades are easier for model 2 (1.4 vs 3.0 points of mix error), a reminder that real colors will be harder.

## How to run

Python 3 with `numpy`, `pandas`, `opencv-python`, `scikit-learn`. Model 2 also needs `torch` and `torchvision` (tested with the `nvcr.io/nvidia/pytorch:25.10-py3` container on a GPU).

```bash
# 1. Plan a twin dataset on a laptop (no Isaac Sim; writes labels.csv only)
python twin_tools/render_dataset.py --dry-run --count 20 --out-dir dataset_plan

# 2. Render it (Isaac Sim 5.1, e.g. on YAAD, from the twin folder)
/isaac-sim/python.sh tools/render_dataset.py --twin /job/in --count 1000 --variations 1 --seed 2001 \
    --styles camera --dye-jitter 0.2 --pos-jitter 4 --out-dir /job/out

# 3. Pictures -> table
python model/build_table.py twin_runs/<run>/out

# 4. Model 1 (+ robustness test) -> model/results/model1/
python model/train_color_model.py model/tables/<run>.csv

# 5. Model 2 (GPU) -> model/results/model2/
python model/train_cnn.py --table model/tables/<run>.csv --pretrained
```

| Folder | Contents |
|---|---|
| `twin_tools/` | `render_dataset.py` (twin picture generator), `recipe_rules.json` (recipe rules + the 10 real recipes) |
| `model/` | `build_table.py`, `train_color_model.py`, `train_cnn.py`; `tables/` and `results/` |
| `samples/` | 24 example twin pictures; `samples.csv` lists recipe, pour and carryover for each |
| `docs/DATASET_GENERATION.md` | How the twin data is made, every label column, and which columns may be model inputs |
| `PROCESS.md` | Plan and status |

## Data

- **The organizers' data and digital twin** (`csi-vial-fill-dataset`, `csi_digital_twin_v1`): the Innovation Lab data folder. Real videos are used for testing only, never for training.
- **Generated twin pictures** (~9,000 pictures from 4 runs of 1,000 vials, plus masks and labels): on the YAAD workspace, not in git because of their size. Access on request.
- **Data plan:** train/test on twin pictures (80/20, split by vial); extra test on real videos 1–4; final validation once on real video 5.

## Credits

- **Data, digital twin and line:** Rockwell Automation, NVIDIA, UWM Connected Systems Institute (CSI), MSOE AI Club.
- **Compute:** YAAD DGX Spark cluster.
- **Tools and models:** NVIDIA Isaac Sim 5.1 and Omniverse Replicator; NVIDIA PyTorch container; ResNet-18 (torchvision, ImageNet weights); scikit-learn, OpenCV, pandas, NumPy.

### AI assistance

**Claude Code** (Anthropic) was used as an AI coding assistant. Its role:
- **Wrote and tested the code** in this repo (the twin generator, table builder and both models), following the project's decisions on the data plan, labels and rules.
- **Checked data quality:** reviewed rendered pictures, found and fixed problems (for example, an unfair "expected color" reference at pour 1 and a too-small color patch), and flagged biases in the twin data.
- **Enforced the project rules:** no label leakage into model inputs, split by vial, and video 5 kept for final validation.
- **Ran independent verification:** three separate AI reviewers re-computed the table, labels and metrics.
- **Helped plan and document:** process outline, docs and YAAD job settings.

Project decisions (what to build, the data plan, which runs to launch) and all GPU jobs were made and run by the project author, not the AI.

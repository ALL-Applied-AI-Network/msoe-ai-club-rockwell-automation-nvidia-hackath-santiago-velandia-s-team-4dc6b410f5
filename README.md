# Pour-by-Pour Color Check for Vial Filling

**Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station 3**

A camera-only add-on for older filling machines. After each pour, it checks whether the vial's color matches what the recipe should look like at that point, and alerts the operator when it doesn't. The main fault it targets is **carryover**: leftover liquid from the previous recipe in the shared mixing path that ends up in the next vial.

> Status: work in progress (branch `ap`). Results below are on digital-twin test data only. Real-video testing is next.

---

## How it works

1. **Training data from the digital twin.** We render the organizers' Isaac Sim fill station from the same angle as the real fill camera, taking **one picture after each pour**. Each vial gets a real or random valid recipe, and about half follow a different recipe (carryover). Lighting, camera look, dye shade and vial position vary per vial. Every picture is labeled for free.
2. **Break each picture into a table.** We measure the liquid color in 8 strips down the vial, divided by the white panel's color so lighting cancels out, and add the recipe and pour number.
3. **Estimate the mix, then compare.** A model estimates the orange and blue shares in the vial from the colors. If the estimate is far from what a clean vial of that recipe should have at that pour, we **flag it**. For example: *"pour 1: about 66% orange, a clean vial would be 92%"*.
4. **On the real line** (planned), a pour-stop detector takes the snapshot after each pour. A short calibration period with a few clean vials per recipe lines the model up with the real camera.

## Results so far (twin test set: 200 vials never used in training)

| Model | AUC | Carryover vials caught | Clean vials falsely flagged |
|---|---|---|---|
| **Model 1: color strips + mix estimate + rule** | **0.90** | **84 of 100** | 32 of 100 |
| End-only baseline (last pour only) | 0.83 | 66 of 100 | 27 of 100 |
| Recipe only, no image (bias check) | 0.54 | — | — |

- Checking after every pour catches more carryover than checking only the finished vial.
- **Most false alarms come from pour 1,** where the unknown leftover amount blurs what "clean" looks like. Judging pours 2+ only, false alarms drop to 5 of 89 clean vials.
- The recipe alone can't predict the answer, so the twin data doesn't leak it.
- Results were independently re-checked three times by separate AI reviewers (see "AI assistance").
- **Model 2** (ResNet-18 image network on YAAD) is built and in testing. Its results are coming.

## Known limits

- **Twin colors aren't calibrated to the real camera** (orange renders too red). A 10% shift in the liquid's green channel drops model 1's AUC from 0.90 to 0.72. Training with random dye shades and the real-line calibration period address this.
- The twin's dye colors were first estimated from video-5 vials, before we reserved video 5 for validation. Random dye shades dilute this, and we'll report it with the final results.
- In the twin, liquid is always fully mixed and settled; carryover is the only fault simulated.
- Not yet tested on real video.

## How to run

Python 3 with `numpy`, `pandas`, `opencv-python`, `scikit-learn`. Model 2 also needs `torch` and `torchvision` (we use the `nvcr.io/nvidia/pytorch:25.10-py3` container on a GPU).

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
- **Our twin pictures** (~9,000 pictures from 4 runs of 1,000 vials, plus masks and labels): on the YAAD team workspace, not in git because of their size. Ask the team for access.
- **Data plan:** train/test on twin pictures (80/20, split by vial); extra test on real videos 1–4; final validation once on real video 5.

## Credits

- **Data, digital twin and line:** Rockwell Automation, NVIDIA, UWM Connected Systems Institute (CSI), MSOE AI Club.
- **Compute:** YAAD DGX Spark cluster.
- **Tools and models:** NVIDIA Isaac Sim 5.1 and Omniverse Replicator; NVIDIA PyTorch container; ResNet-18 (torchvision, ImageNet weights); scikit-learn, OpenCV, pandas, NumPy.

### AI assistance

We used **Claude Code** (Anthropic) as an AI coding assistant. Its role:
- **Wrote and tested the code** in this repo (the twin generator, table builder and both models), following the team's decisions on the data plan, labels and rules.
- **Checked data quality:** reviewed rendered pictures, found and fixed problems (for example, an unfair "expected color" reference at pour 1 and a too-small color patch), and flagged biases in the twin data.
- **Enforced the rules we set:** no label leakage into model inputs, split by vial, and video 5 kept for final validation.
- **Ran independent verification:** three separate AI reviewers re-computed the table, labels and metrics.
- **Helped plan and document:** process outline, docs and YAAD job settings.

The team made the decisions (what to build, the data plan, which runs to launch) and launched and monitored all GPU jobs.

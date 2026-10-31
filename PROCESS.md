# Version 1: Process Outline

Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station.
This file says what we are building, what has been done, and what comes next. Anything not decided yet is listed as **open**.

---

## 1. Goal
A **camera-only add-on for older filling machines.** Mount a camera, let the program learn the station during a calibration period, and it flags vials whose color doesn't match the recipe. The main cause it should catch is **carryover**: leftover fluid from the previous recipe mixing into the next vial.

## 2. The system has two parts
| Part | What it does | Status |
|---|---|---|
| **A. Pour-stop detector** | Watches the video, detects when each pour ends, and takes a snapshot | **Later** (not started in version 1) |
| **B. ML model** | Snapshot + recipe + pour number → *"Does the mix so far match what it should be at this pour?"* The final-vial check is just the last pour | **Now** |

## 3. Data plan
| Use | Data | Status |
|---|---|---|
| **Train / test** | Digital-twin pictures, **80% / 20%, split by vial** (all pictures of one vial stay on the same side) | Now |
| **Extra test** | Real videos 1–4 | Later. Their recipe labels are an open item |
| **Validation** | Real video 5, used **once, at the very end** | Last |

## 4. Steps

### Step 1: Access the twin ✅ done
- The twin is the organizers' 3D copy of the fill station (`~/Desktop/Hackathon/csi_digital_twin_v1`). It's **not copied** into version 1; scripts point to it.
- It runs in **Isaac Sim 5.1** on **YAAD** (the shared DGX Spark computers). Isaac Sim 6.0 gave empty images there.
- The organizers' `twin_tools/render_example.py` rendered one picture successfully.

### Step 2: Generate twin pictures ✅ generator built and tested
- **Files:**
  - `twin_tools/render_dataset.py` (the generator)
  - `twin_tools/recipe_rules.json` (the recipe rules)
  - `docs/DATASET_GENERATION.md` (full guide and column list)
- **One picture after each pour**, from the camera that matches the real fill camera.
- **Recipes:** the 10 real recipes (5–14) plus random new recipes. All follow the rules: ≤80 mL, ≥10 mL per tank used, ≤50 mL orange, **no T300**.
- **Carryover:** about half the vials. Every vial starts with 5–30 mL of the previous vial's mix, capped at half the vial.
- **Lighting and camera look** are random **per vial** and shared by all its pours. The camera never moves.
- **Every render is saved in two styles:** **clean** (crisp, like the organizers' `fillcam_frame686.png`) and **camera** (with the real-camera look), so the model sees both. Same GPU time.
- **Calm liquid surface:** the settled finished-vial shape, scaled to each pour's height.
- **Liquid material** keeps the organizers' opacity (0.6).
- **Each picture also gets:** the measured **white panel** color, a mask, a vial box (no liquid box), and one CSV row of labels.
- **Tested** with a 5-vial run (seed 4). The changes requested after the first version all worked.

### Step 3: Generate the training set ⏭ next
- ✅ **Timing test** done (Oct 29): about 105 s startup + about 2 s per render, so about **74 min for 1,000 vials** (one job). Note: one render now gives two pictures (clean + camera) at the same GPU time.
- Then decide the final size.

### Step 4: Break down the images ⏭
Turn every picture into a **table row** (the model's input matrix), with columns such as the measured liquid color and the white panel color.

### Step 5: Build and compare models ⏭ (in this order, each reported)
1. **Simple color measurements** + a simple classifier
2. **TAO image model** (NVIDIA's training toolkit, named in the organizers' docs)
3. **Cosmos Reason** (NVIDIA's vision-language model, named in the organizers' docs)

### Later
- Pour-stop detector (part A)
- Extra test on real videos 1–4
- Validation on video 5

## 5. Rules we follow
- **ML inputs vs. answers:** the CSV columns describing *how the picture was made* (colors, fractions, leftover mL, lighting, seed and so on) are **never model inputs.** The full table is in `docs/DATASET_GENERATION.md`.
- **Video 5 is validation only.** Nothing is built or tuned on it.
- **The organizers' files are never edited.**

## 6. Open items
- **Twin colors are not calibrated** to the real camera yet. In the test, the twin's liquid looked lighter and more see-through than real. Calibrating needs real vials with known recipes.
- **Videos 1–4 have no recipe labels.** We'll handle that when we get there.
- **Training-set size**: waiting on the timing test.

## 7. Other folders (not part of version 1)
- `Oct_19_Explore/`: where the generator was first built and tested, including the test renders in `twin_runs/`. Left as it is.
- `changeover-watch/`: an earlier prototype. It was built using video 5, so its results are not part of version 1. Left as it is.

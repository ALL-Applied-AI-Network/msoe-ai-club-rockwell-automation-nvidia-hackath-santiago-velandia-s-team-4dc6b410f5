# Vial Color Check for Fill Stations

**Rockwell Automation + NVIDIA Innovation Lab, Fall 2026 · Dataset 2: Fluid Mixing Station 3**
**Repository:** github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-santiago-velandia-s-team-4dc6b410f5 (branch `ap`)

## Problem
At CSI's Fluid Mixing Station 3, three tanks (orange dye, blue dye, clear water) share one mixing path. When the recipe changes, leftover liquid from the previous recipe ends up in the next vial (**carryover**), so that vial is the wrong color. The line's existing color check only asks whether a color is one it knows, not whether it is right for this vial's recipe.

## Solution
A camera-based add-on that checks every vial against its own recipe:
1. **Find each vial** in the fill-camera video, from the carrier and pour movement, and take a picture of the finished vial.
2. **Estimate the mix:** a neural network (ResNet-18 plus the recipe) estimates the share of orange and blue dye in the vial.
3. **Compare with the recipe:** if the estimate is more than 8 percentage points from a correct vial of that recipe, the vial gets an **ALERT**.
4. **Show the operator:** a browser dashboard shows the live camera, OK or ALERT per vial, measured vs expected mix, and saves every vial's picture and result to a file.

**Inputs:** one camera (640×480, same view as CSI's `vial_fill` camera), the recipe queue (CSV) and the tank assignment. **No connection to machine controls:** the program only advises, and people act on the alerts.

## Training: digital twin only
- **Data:** 8,904 pictures of 4,000 simulated vials, rendered in NVIDIA Isaac Sim from the organizers' digital twin, one picture after each pour.
- **Labels:** exact, from the twin. Carryover follows the station's leftover physics (5–30 mL of the previous recipe).
- **Variety:** lighting, camera look, dye shade and vial position vary per vial, so the model can't memorize one look.
- **Split and hardware:** split 80/20 by vial; trained in about 7 minutes on an NVIDIA DGX Spark GPU (YAAD lab).
- **No real video was used for training.** Real video 5 was kept for one final validation.

## Results

| | Twin test (800 vials) | Real video 5 (29 scored vials) |
|---|---|---|
| Carryover vials caught | 336 of 377 (89%) | **8 of 9** |
| Clean vials wrongly flagged | 175 of 423 (41%) | **9 of 20** |
| AUC (1.0 = perfect, 0.5 = guessing) | 0.90 | **0.82** |

- **Mix estimate error on twin vials:** 2.5 / 2.1 percentage points (orange / blue).
- **On video 5, carryover vials stand out:** a typical gap of 28 points against 7 for clean vials. Of the flagged vials, 47% were carryover, against 31% by chance.

**Failure cases:**
- **False alarms:** 8 of the 9 come from four pale or blue-only recipes, where the twin's colors don't match the real camera, so the model over-reads the dye.
- **The one miss:** a recipe change between two equal orange/blue recipes, where the leftover barely changes the color.

## Value for manufacturers
- **Why it matters:** When defects escape the line, the cost grows sharply: a single bottling-defect recall cost Constellation Brands $3.5M in net sales plus a $5.4M charge to cost of product sold in 2014 [10], and even small breweries' recall claims average $25,000–$50,000 [11]. Catching an off-color vial at the filling station costs one vial.
- **Catches carryover per vial,** right after recipe changes, before off-color vials move on to capping and packing.
- **Cheap to add:** one camera and an ordinary computer; no machine changes, and training needs no real defect samples.
- **Leaves a record:** a picture and result for every vial, which quality staff can use to re-test, rework or discard flagged vials.
- **Points to the cause:** alerts cluster after recipe changes, which supports adding a flush step or setting aside the first vial after each change.

## Limits and next steps
- **Limits:**
  - Validation is one real video (30 vials, 10 recipes).
  - Carryover labels follow an assumed rule (the first vial after a recipe change).
  - The twin's dye colors were estimated from video 5 before it was set aside.
  - Only the finished vial is checked on real video.
- **Next steps:**
  - A short color adjustment from a few known-good vials per recipe.
  - A stronger baseline comparison.
  - A live-camera test on CSI's line.

## References
1. CSI vial-fill dataset and digital twin (`csi-vial-fill-dataset`, `csi_digital_twin_v1`): Rockwell Automation, NVIDIA, UWM Connected Systems Institute (CSI), MSOE AI Club, Fall 2026 Innovation Lab.
2. NVIDIA Isaac Sim 5.1 and Omniverse Replicator: docs.isaacsim.omniverse.nvidia.com
3. NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:25.10-py3`, NGC catalog: catalog.ngc.nvidia.com
4. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition" (ResNet), CVPR 2016.
5. J. Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database," CVPR 2009 (pretrained weights via torchvision).
6. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization" (AdamW), ICLR 2019.
7. PyTorch and torchvision (pytorch.org); OpenCV (opencv.org); pandas; NumPy.
8. YAAD DGX Spark GPU lab (compute).
9. Claude Code (Anthropic) was used as an AI coding assistant; see the README's "AI reliance" section.
10. Constellation Brands, Inc., Form 10-Q, product recall note (six months ended Aug 31, 2014), SEC EDGAR: sec.gov/Archives/edgar/data/16918/000001691814000029/R18.htm
11. J. Lassen, The Hanover Insurance Group, quoted in "Product Recall: The Most Overlooked Protection," The Brewer Magazine, Apr 24, 2018: thebrewermagazine.com/product-recall-overlooked-protection/

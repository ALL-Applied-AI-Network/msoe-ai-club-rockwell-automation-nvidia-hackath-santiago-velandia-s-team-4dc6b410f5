# Video Script (about 5 minutes)

**Title:** Vial Color Check for Fill Stations
**Format:** screen recording with voice-over. Times are targets; the spoken text runs about 130 words per minute.

**Before recording:**
1. In `version_1/`, start the dashboard at 30× speed:
   ```bash
   .venv/bin/python model/vial_check.py --live --speed 30
   ```
   At 30×, all of real video 5 (28 minutes) plays in about **1 minute**, and the totals end at the final numbers.
2. Open http://127.0.0.1:8000. On the setup screen:
   - **Video:** upload `csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4`.
   - **Recipe queue:** upload `csi-vial-fill-dataset/vial_fill/labels/video5_vial_recipes.csv`.
   - **Tanks:** leave the defaults: T200 orange, T400 blue, T100 clear.
3. Have these ready to show:
   - the README on GitHub;
   - `model/results/video5_validation/contact_sheet.jpg`;
   - one twin training picture from `samples/`.

---

### 0:00–0:35 · The problem
**Screen:** the real fill-camera video (video 5), paused on a vial right after a recipe change.

> This is Fluid Mixing Station 3 on CSI's vial line. Three tanks fill each vial: orange dye, blue dye, and clear water. The tanks share one mixing path, so when the recipe changes, liquid left over from the previous recipe goes into the next vial. That vial comes out the wrong color: this is called carryover. The line's existing color check only asks whether a color is one it knows, not whether it's the right color for this vial's recipe, and it gives no warning around a recipe change.

### 0:35–1:20 · The approach
**Screen:** a twin training picture next to a real frame, then the "How it works" section of the README.

> The model was trained without any real line pictures. All training data comes from the digital twin of the station in NVIDIA Isaac Sim: 8,904 pictures of 4,000 simulated vials, one picture after every pour. The twin knows the exact mix in every vial, so every picture is labeled for free. Carryover comes from the same physics as the real line: leftover liquid from the previous recipe.
>
> The program works in four steps. It finds each vial in the camera video and takes a picture of the finished vial. Then a neural network looks at the picture and the recipe and estimates how much orange and blue dye is in the vial. If that estimate is more than 8 percentage points away from what a correct vial of the recipe looks like, the vial gets an alert.

### 1:20–2:35 · Live demo on real video 5
**Screen:** the dashboard. Click **Start check**, let it play to the end (about 1 minute), and talk over it.

> This is the operator dashboard, running on a real video from CSI's line that the model never trained on: 30 vials and 10 recipes, played here at 30 times speed. The setup takes the camera video, the recipe queue, and which tank holds which color. No connection to the machine's controls is needed.
>
> For each vial, the dashboard shows the live camera, the recipe being filled, and, when the vial is done, a green OK or a red ALERT. Next to it, the measured mix is shown against the expected mix. Here, vial 4 is the first vial after a change to recipe 6: the recipe says 25 percent orange and no blue, but the vial measures about 35 percent of each. That's carryover, and it's flagged.
>
> *(When it finishes:)* At the end: 8 of the 9 vials right after a recipe change were caught, and 9 of the 20 clean vials were wrongly flagged.

### 2:35–3:30 · Results
**Screen:** the README "Results at a glance" table, then the by-recipe table and the contact sheet.

> On 800 test vials from the twin, the model estimates the dye mix within about 2 to 3 percentage points, and it scores an AUC of 0.90 for separating bad vials from good ones.
>
> On real video 5 it scores 0.82, with no adjustment to the real camera. Carryover vials have a typical gap of 28 points, against 7 for clean vials.
>
> The failures follow a clear pattern. The false alarms come almost entirely from four recipes: recipe 7, which is blue only, and recipes 6, 10 and 14, which are mostly clear water. On those, the model reads more dye than is really there, because the twin's colors don't exactly match the real camera. The one miss is recipe 9 right after recipe 8. Both have equal orange and blue, so the leftover barely changes the color.

### 3:30–4:20 · How it is used on the line
**Screen:** the README section "Using the alerts on the line".

> The program only advises; it never controls the machine. When a vial gets an alert, the operator sees which vial it is, its recipe, and how far off it is, and pulls it at the next stop for a second look.
>
> Every run saves a picture of each vial and a file listing every vial with OK or ALERT. Quality staff can use that file to decide what happens to flagged vials: re-test, rework, or discard.
>
> The alerts also point to the cause. They cluster on the first vial after a recipe change, so the plant could add a short flush between recipes, or set aside the first vial after every change. Later, with the plant's approval, the same signal could drive a stack light or a reject gate.

### 4:20–4:50 · Limits and next steps
**Screen:** the README "Known limits" section.

> The limits, plainly:
> - Validation is one real video of 30 vials.
> - The carryover labels follow an assumed rule: the first vial after a recipe change.
> - The twin's dye colors were first estimated from this same video, before it was set aside for validation.
>
> The next steps are a short color adjustment from a few known-good vials per recipe, a stronger baseline comparison, and a test with a live camera on CSI's line.

### 4:50–5:00 · Close
**Screen:** the GitHub repo page.

> Trained only on a digital twin, this camera check catches carryover on the real line, works from one camera and the recipe queue, and leaves every decision with the operator. The code, the results and every number shown here are in the repository.

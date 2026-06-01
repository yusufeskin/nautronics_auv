> This README was written by Claude (and now by Antigravity as well).

# Camera Calibration — ChArUco Board

Intrinsic calibration scripts for the Nautronics AUV camera system.  
Produces `fx`, `fy`, `cx`, `cy`, and distortion coefficients (`k1`–`k6`, `p1`, `p2`) saved in ROS 2-compatible YAML format.

---

## Directory Structure

```
scripts/calibrator/
├── generate_board.py           # Step 1 — generate the printable board (run once)
├── video_to_calib_images.py    # Option B — extract high-quality frames from a calibration video
├── calibrator.py               # Step 2 — run calibration from collected images
├── calib_images/               # Directory containing calibration photos (git-ignored)
├── charuco_board.png           # Generated calibration board image
└── README.md
```

---

## Workflow

### Step 1 — Print the Board (one-time)

```bash
python generate_board.py
```u

This outputs `charuco_board.png` (A4, 300 DPI).

**Print settings:**
- Paper: A4, Portrait
- Scale: **100% / Actual Size**
- "Fit to Page" / "Scale to fit": **OFF**

After printing, measure the squares with a ruler:

| What to measure | Expected | Variable in code |
|-----------------|----------|------------------|
| White chess square side | 4.0 cm | `0.04` |
| Black ArUco marker side | 2.0 cm | `0.02` |

If the printed dimensions differ (printers may scale slightly), update **both scripts**:

```python
board = aruco.CharucoBoard((5, 7), 0.0392, 0.0196, aruco_dict)  # example
```

> Mount the board on a **rigid flat surface** (foam board, clipboard, etc.).  
> Any flex or curl during shooting corrupts the calibration.

---

### Step 2 — Collect Images

You can collect calibration images in two ways:
- **Option A: Manual Photos** — Take 20-30 separate photos and place them in `calib_images/`.
- **Option B: Video Extraction (Recommended for Underwater)** — Record a short video of the board and automatically extract high-quality, diverse frames using `video_to_calib_images.py`.
u
#### Option A — Manual Photos

Shoot **20–30 photos** and drop them into `calib_images/`.

**Checklist for good coverage:**

- [ ] All four corners of the frame covered across the set
- [ ] Multiple tilt angles (left/right, up/down, diagonal)
- [ ] Multiple distances (close, medium, far)
- [ ] Board fully visible in every shot — no cropping
- [ ] No motion blur
- [ ] **Shoot underwater** — do not calibrate in air if the camera uses a dome/flat port

> For underwater shots: ambient light shifts toward blue-green and contrast drops.  
> Increase the board's lighting or use a torch to improve marker detection.

---

#### Option B — Video Extraction (Recommended)

Instead of taking individual photos, record a short **1–2 minute MP4 video** of the ChArUco board. Slowly move the camera (or the board) to:
- Cover all four corners and the center of the frame.
- Tilt the board at various angles (pitch, yaw, roll).
- Vary distances (close, medium, far).

Then, run `video_to_calib_images.py` to automatically extract high-quality, diverse, and sharp frames:

```bash
python video_to_calib_images.py --video your_video.mp4
```

The script utilizes three intelligent filters to filter out poor frames:
1. **Blur Filter (Laplacian Variance):** Skips blurry frames caused by fast movement.
2. **Similarity Filter:** Compares the frame's pixel difference to the last saved frame. Skips near-identical frames to ensure variety.
3. **ChArUco Pre-check:** Verifies the board is visible by ensuring at least 4 ArUco markers are detected.

##### Advanced Settings & CLI Arguments

To fine-tune extraction parameters:

```bash
python video_to_calib_images.py --video video.mp4 --target 150 --blur 50.0 --diff 10.0 --skip 3
```

| Argument | Default | Description |
|---|---|---|
| `--video` | *(Required)* | Path to the input video file (e.g. `.mp4`). |
| `--output` | `calib_images` | Destination folder for extracted frames. |
| `--target` | `150` | Maximum number of images to extract and save. |
| `--blur` | `60.0` | Minimum sharpness threshold (Laplacian variance). Lower = allow blurrier frames. |
| `--diff` | `10.0` | Minimum frame difference threshold. Lower = allow more similar frames. |
| `--skip` | `3` | Frame step rate (e.g. `--skip 3` processes every 3rd frame, reducing CPU load). |
| `--no-marker-check` | `False` | Disable the ArUco marker pre-check (runs faster, but might save empty frames). |

##### Tuning for Underwater Videos:
- **Sharpness (`--blur`):** Underwater conditions often make images softer. If too many frames are skipped, lower `--blur` to `30.0` – `50.0`.
- **Motion Speed (`--diff`):** If the camera moves very slowly or is mounted on a tripod, lower `--diff` to `5.0` – `8.0` to avoid skipping valid frames.
- **Skip Rate (`--skip`):** Adjust based on the video's frame rate. For a 30 FPS video, `--skip 3` evaluates 10 frames per second, which is ideal.

---

### Step 3 — Run Calibration

```bash
cd scripts/calibrator
python calibrator.py
```

The script auto-detects your OpenCV version (≥ 4.7 new API, < 4.7 legacy API).

**Expected output:**

```
OpenCV 4.x.x — New API in use
Total 25 images found.
  [OK]  calib_images/img_001.png — 24 corners found
  ...
==================================================
  Calibration RMS Error : 0.43 pixels
==================================================
  Evaluation: EXCELLENT ✓

Camera Matrix (fx, fy, cx, cy):
  fx = 821.34  fy = 820.97
  cx = 634.21  cy = 358.74
```

**RMS error guide:**

| RMS (pixels) | Verdict |
|-------------|---------|
| < 0.5 | Excellent |
| 0.5 – 1.0 | Good |
| 1.0 – 2.0 | Acceptable — try more images |
| > 2.0 | Poor — review images, re-shoot |

---

### Step 4 — Output Files

| File | Format | Use |
|------|--------|-----|
| `camera_parameters.yaml` | OpenCV FileStorage | ROS 2 `camera_info` |
| `camera_parameters.npz` | NumPy archive | Direct Python use |

Load in Python:

```python
import numpy as np
data = np.load("camera_parameters.npz")
K    = data["cam_matrix"]    # 3x3 intrinsic matrix
dist = data["dist_coeffs"]   # distortion coefficients
```

Use in ROS 2 — point your launch file or `camera_info_url` to `camera_parameters.yaml`.

---

## Board Specification

| Parameter | Value |
|-----------|-------|
| Dictionary | `DICT_6X6_250` |
| Layout | 5 columns × 7 rows |
| Chess square size | 40 mm |
| ArUco marker size | 20 mm |
| Distortion model | Rational (`CALIB_RATIONAL_MODEL`) — k1–k6 + p1, p2 |

The rational model is used instead of the standard 3-coefficient model because wide-angle and dome-port optics produce higher-order radial distortion that the simpler model cannot fully capture.
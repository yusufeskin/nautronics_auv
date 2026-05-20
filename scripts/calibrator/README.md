> This README was written by Claude.

# Camera Calibration — ChArUco Board

Intrinsic calibration scripts for the Nautronics AUV camera system.  
Produces `fx`, `fy`, `cx`, `cy`, and distortion coefficients (`k1`–`k6`, `p1`, `p2`) saved in ROS 2-compatible YAML format.

---

## Directory Structure

```
scripts/calibrator/
├── generate_board.py       # Step 1 — generate the printable board (run once)
├── calibrator.py           # Step 2 — run calibration from collected images
├── calib_images/           # Place calibration photos here (git-ignored)
└── README.md
```

---

## Workflow

### Step 1 — Print the Board (one-time)

```bash
python generate_board.py
```

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
import cv2
import cv2.aruco as aruco
import glob
import numpy as np

# ─── 1. PARAMETERS ──────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict) # if real dimensions are different change here 

# Check OpenCV version (4.7+ uses new API)
OPENCV_MAJOR = int(cv2.__version__.split('.')[0])
OPENCV_MINOR = int(cv2.__version__.split('.')[1])
USE_NEW_API = (OPENCV_MAJOR > 4) or (OPENCV_MAJOR == 4 and OPENCV_MINOR >= 7)

print(f"OpenCV {cv2.__version__} — {'New' if USE_NEW_API else 'Old'} API in use")

# ─── 2. IMAGE ANALYSIS ──────────────────────────────────────────────────────
images = sorted(glob.glob('calib_images/*.png') + glob.glob('calib_images/*.jpg'))

if len(images) == 0:
    raise FileNotFoundError("No images found in calib_images/ directory!")

print(f"Total {len(images)} images found.")

all_corners = []
all_ids     = []
image_size  = None
basarili    = 0
basarisiz   = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"  [ERROR] Cannot read: {fname}")
        basarisiz += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    # ── Marker Detection ────────────────────────────────────────────────────
    if USE_NEW_API:
        # OpenCV 4.7+ — ArucoDetector class
        detector_params = aruco.DetectorParameters()
        # For underwater use: wider adaptive thresholding
        detector_params.adaptiveThreshWinSizeMin  = 5
        detector_params.adaptiveThreshWinSizeMax  = 25
        detector_params.adaptiveThreshWinSizeStep = 4

        aruco_detector   = aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids, _  = aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) >= 4:
            charuco_detector = aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        else:
            charuco_corners, charuco_ids = None, None
    else:
        # OpenCV < 4.7 — Legacy API
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        if ids is not None and len(ids) >= 4:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
        else:
            charuco_corners, charuco_ids = None, None

    # ── Quality Check ───────────────────────────────────────────────────────
    MIN_CORNERS = 6  # At least 6 ChArUco corners required (for reliability)
    if (charuco_corners is not None and
        charuco_ids    is not None and
        len(charuco_corners) >= MIN_CORNERS):

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        basarili += 1
        print(f"  [OK]  {fname} — {len(charuco_corners)} corners found")
    else:
        basarisiz += 1
        bulunan = len(charuco_corners) if charuco_corners is not None else 0
        print(f"  [--]  {fname} — Insufficient corners ({bulunan}/{MIN_CORNERS}), skipped")

# ─── 3. DATA VALIDATION ─────────────────────────────────────────────────────
print(f"\nUsable images: {basarili} / {basarili + basarisiz}")

if basarili < 10:
    raise RuntimeError(
        f"Not enough valid images! ({basarili} found). "
        "At least 10, preferably 20-30 images from different angles are required."
    )

# ─── 4. CALIBRATION ─────────────────────────────────────────────────────────
print("\nPerforming mathematical optimization...")

# For underwater / wide-angle lenses, RATIONAL_MODEL is recommended (k1–k6 + p1, p2)
flags = (
    cv2.CALIB_RATIONAL_MODEL   # 6 radial coefficients (k1-k6) — for fisheye/wide-angle lenses
    # | cv2.CALIB_FIX_K3      # Uncomment if using standard lens
    # | cv2.CALIB_ZERO_TANGENT_DIST  # Ignore tangential distortion
)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners = all_corners,
    charucoIds     = all_ids,
    board          = board,
    imageSize      = image_size,
    cameraMatrix   = None,
    distCoeffs     = None,
    flags          = flags
)

# ─── 5. RESULTS EVALUATION ───────────────────────────────────────────────────
print("\n" + "="*50)
print(f"  Calibration RMS Error : {ret:.4f} pixels")
print("="*50)

if ret < 0.5:
    print("  Evaluation: EXCELLENT ✓")
elif ret < 1.0:
    print("  Evaluation: GOOD ✓")
elif ret < 2.0:
    print("  Evaluation: ACCEPTABLE (try more images)")
else:
    print("  Evaluation: POOR ✗ — Review images!")

print(f"\nCamera Matrix (fx, fy, cx, cy):")
print(f"  fx = {camera_matrix[0,0]:.2f}  fy = {camera_matrix[1,1]:.2f}")
print(f"  cx = {camera_matrix[0,2]:.2f}  cy = {camera_matrix[1,2]:.2f}")
print(f"\nDistortion Coefficients:\n  {dist_coeffs.ravel()}")

# ─── 6. SAVE RESULTS ─────────────────────────────────────────────────────────
# .npz — Easy NumPy loading format
np.savez("camera_parameters.npz",
         cam_matrix  = camera_matrix,
         dist_coeffs = dist_coeffs,
         rms_error   = ret,
         image_size  = image_size)

# .yaml — ROS2 camera_info format
fs = cv2.FileStorage("camera_parameters.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("image_width",  image_size[0])
fs.write("image_height", image_size[1])
fs.write("rms_error",    ret)
fs.write("camera_matrix",   camera_matrix)
fs.write("distortion_coefficients", dist_coeffs)
fs.release()

print("\nParameters saved:")
print("  → camera_parameters.npz")
print("  → camera_parameters.yaml  (ROS2 compatible)")
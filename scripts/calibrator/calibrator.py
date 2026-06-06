import cv2
import cv2.aruco as aruco
import glob
import numpy as np

# ─── 1. PARAMETERS ──────────────────────────────────────────────────────────
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5, 7), 0.04, 0.02, aruco_dict) # Gerçek fiziksel boyutlar farklıysa burayı değiştir.

OPENCV_MAJOR = int(cv2.__version__.split('.')[0])
OPENCV_MINOR = int(cv2.__version__.split('.')[1])
USE_NEW_API = (OPENCV_MAJOR > 4) or (OPENCV_MAJOR == 4 and OPENCV_MINOR >= 7)

print(f"OpenCV {cv2.__version__} — {'New' if USE_NEW_API else 'Old'} API in use")

# ─── 2. IMAGE ANALYSIS ──────────────────────────────────────────────────────
images = sorted(glob.glob('calib_images/*.png') + glob.glob('calib_images/*.jpg'))

if len(images) == 0:
    raise FileNotFoundError("No images found in calib_images/ directory!")

print(f"Total {len(images)} images found.")

all_object_points = []
all_image_points  = []

image_size  = None
basarili    = 0
basarisiz   = 0

# DÜZELTME 1: Dedektör döngü dışında BİR KEZ tanımlanır (Performans için).
# DÜZELTME 2: Sistemi bozan "detector_params" tamamen kaldırıldı. 
# Artık OpenCV'nin fotoğraflarında daha iyi çalışan varsayılan ayarları devrede.
if USE_NEW_API:
    charuco_detector = aruco.CharucoDetector(board)

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
        # Tek seferde hem marker hem tahta tespiti yapılır (Yeni API)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        if marker_ids is None or len(marker_ids) < 4:
            charuco_corners, charuco_ids = None, None
    else:
        # Eski API mantığı (OpenCV 4.6 ve altı için)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        if ids is not None and len(ids) >= 4:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        else:
            charuco_corners, charuco_ids = None, None

    # ── Quality Check ───────────────────────────────────────────────────────
    MIN_CORNERS = 6  
    if (charuco_corners is not None and charuco_ids is not None and len(charuco_corners) >= MIN_CORNERS):

        # 3D fiziksel tahta noktaları ile 2D piksel noktalarını eşleştiriyoruz
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

        all_object_points.append(obj_points)
        all_image_points.append(img_points)

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

flags = cv2.CALIB_RATIONAL_MODEL

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints = all_object_points,
    imagePoints  = all_image_points,
    imageSize    = image_size,
    cameraMatrix = None,
    distCoeffs   = None,
    flags        = flags
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
# .npz Kaydı (Numpy için kolay yükleme formatı)
np.savez("camera_parameters.npz",
         cam_matrix  = camera_matrix,
         dist_coeffs = dist_coeffs,
         rms_error   = ret,
         image_size  = image_size)

# .yaml Kaydı — ROS2 camera_info formatı 
P = np.zeros((3, 4))
P[:3, :3] = camera_matrix  # Projection matrisi rektifikasyon olmadan hazırlanır

yaml_content = f"""image_width: {image_size[0]}
image_height: {image_size[1]}
camera_name: camera
camera_matrix:
  rows: 3
  cols: 3
  data: {camera_matrix.flatten().tolist()}
distortion_model: rational_polynomial
distortion_coefficients:
  rows: 1
  cols: {dist_coeffs.shape[1]}
  data: {dist_coeffs.flatten().tolist()}
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: {P.flatten().tolist()}
"""

with open("camera_parameters.yaml", "w") as f:
    f.write(yaml_content)

print("\nParameters saved:")
print("  → camera_parameters.npz")
print("  → camera_parameters.yaml  (ROS2 compatible)")
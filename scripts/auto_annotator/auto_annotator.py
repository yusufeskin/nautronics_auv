"""
YOLO Pose Auto-Annotator for CVAT
This script uses a pre-trained YOLO Pose model to automatically annotate unlabeled images.
It creates a perfectly formatted .zip file ready to be uploaded to CVAT without any errors.
"""

from ultralytics import YOLO
import cv2
import os
import shutil
import zipfile

# ==========================================
# 1. USER CONFIGURATIONS (KULLANICI AYARLARI)
# ==========================================
MODEL_PATH = "best.pt"             # Path to your trained YOLO Pose model (.pt)
IMAGE_DIR = "unlabeled_images"     # Folder containing the images you want to annotate
OUTPUT_DIR = "cvat_dataset"        # The name of the output folder that will be created
CLASS_NAME = "object_name"         # Name of your class in CVAT (e.g., torpedo, car, gate)
CLASS_ID = 0                       # Class ID (0 if it's the only class)
NUM_KPTS = 4                       # Number of keypoints your model detects
CONF_THRES = 0.25                  # Confidence threshold for detections

# ==========================================
# 2. SETUP DIRECTORIES
# ==========================================
# Create CVAT standard directory structure: labels/train
dest_label_dir = os.path.join(OUTPUT_DIR, "labels/train")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR) # Clean previous runs
os.makedirs(dest_label_dir, exist_ok=True)

# ==========================================
# 3. INFERENCE AND ANNOTATION
# ==========================================
print(f"Loading YOLO model from '{MODEL_PATH}'...")
model = YOLO(MODEL_PATH)

valid_images = [] # To keep track of images that actually have detections

print(f"Starting auto-annotation for images in '{IMAGE_DIR}'...")
for img_name in sorted(os.listdir(IMAGE_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")): 
        continue

    src_img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(src_img_path)
    
    if img is None: 
        continue

    h, w = img.shape[:2]
    results = model(img, conf=CONF_THRES, verbose=False)
    r = results[0]

    # Skip if no detection (Prevents CVAT React Error #310 caused by empty txt files)
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or len(r.keypoints.data) == 0:
        print(f"[{img_name}] Skipped: No detection.")
        continue

    box = r.boxes.xywhn[0].cpu().tolist()
    kpts = r.keypoints.data[0].clone().cpu()

    # Format: class_id x_center y_center width height kpt1_x kpt1_y kpt1_vis ...
    label = [str(CLASS_ID)]
    label += [f"{box[0]:.6f}", f"{box[1]:.6f}", f"{box[2]:.6f}", f"{box[3]:.6f}"]

    for i in range(NUM_KPTS):
        x = float(kpts[i, 0]) / w
        y = float(kpts[i, 1]) / h
        v = int(kpts[i, 2] > 0)
        label += [f"{x:.6f}", f"{y:.6f}", str(2 if v else 0)]

    label_path = os.path.join(dest_label_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, "w") as f: 
        f.write(" ".join(label))
    
    valid_images.append(img_name)
    print(f"[{img_name}] Annotated successfully.")

# ==========================================
# 4. CVAT CONFIGURATION FILES (train.txt & data.yaml)
# ==========================================
print("\nGenerating CVAT configuration files...")

# Create train.txt
train_txt_path = os.path.join(OUTPUT_DIR, "train.txt")
with open(train_txt_path, "w") as f:
    for img_name in valid_images:
        f.write(f"images/train/{img_name}\n")

# Create data.yaml
yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
yaml_content = f"""names:
  {CLASS_ID}: {CLASS_NAME}
kpt_shape: [{NUM_KPTS}, 3]
path: .
train: train.txt
"""
with open(yaml_path, "w") as f:
    f.write(yaml_content)

# ==========================================
# 5. PACKAGING FOR CVAT (ZIP)
# ==========================================
print("Zipping files for CVAT upload...")
zip_filename = f"{OUTPUT_DIR}_upload.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add root files
    zipf.write(yaml_path, "data.yaml")
    zipf.write(train_txt_path, "train.txt")
    # Add labels
    for root, dirs, files in os.walk(dest_label_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.join("labels/train", file)
            zipf.write(file_path, arcname)

print("\n" + "="*50)
print(f"DONE! Successfully annotated {len(valid_images)} images.")
print(f"Please upload '{zip_filename}' to your CVAT Task via 'Upload Annotations'.")
print("="*50)
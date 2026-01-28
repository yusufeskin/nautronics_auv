from ultralytics import YOLO
import cv2
import os
import shutil

MODEL_NAME = "gate"
IMG_DIR_NAME = "gate_dataset"
NUM_KPTS = 5 # keypoints
CLASS_ID = 0 # label/skeleton ID

ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
image_dir = os.path.join(ws_root, IMG_DIR_NAME)
model_path = os.path.join(ws_root, f"src/nautronics_auv/auv_vision/models/{MODEL_NAME}.pt")

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_base_dir = os.path.join(desktop_path, f"{MODEL_NAME}_auto")
dest_image_dir = os.path.join(output_base_dir, "images/train")
dest_label_dir = os.path.join(output_base_dir, "labels/train")

os.makedirs(dest_image_dir, exist_ok=True)
os.makedirs(dest_label_dir, exist_ok=True)

model = YOLO(model_path)

for img_name in sorted(os.listdir(image_dir)):
    if not img_name.lower().endswith((".jpg", ".png")): continue

    src_img_path = os.path.join(image_dir, img_name)
    dst_img_path = os.path.join(dest_image_dir, img_name)
    
    shutil.copy2(src_img_path, dst_img_path)

    img = cv2.imread(src_img_path)
    if img is None: continue

    h, w = img.shape[:2]
    results = model(img, conf=0.25, verbose=False)
    r = results[0]

    # If no detection, skip image (fills txt after)
    if r.boxes is None or len(r.boxes) == 0:
        print(f"{img_name}: no detections")
        continue

    if r.keypoints is None or len(r.keypoints.data) == 0:
        print(f"{img_name}: no keypoints")
        continue

    box = r.boxes.xywhn[0].cpu().tolist()
    kpts = r.keypoints.data[0].clone().cpu()

    label = [str(CLASS_ID)]
    label += [f"{box[0]:.6f}", f"{box[1]:.6f}", f"{box[2]:.6f}", f"{box[3]:.6f}"]

    for i in range(NUM_KPTS):
        x = float(kpts[i, 0]) / w
        y = float(kpts[i, 1]) / h
        v = int(kpts[i, 2] > 0)
        label += [f"{x:.6f}", f"{y:.6f}", str(2 if v else 0)]

    label_path = os.path.join(dest_label_dir, os.path.splitext(img_name)[0] + ".txt")
    with open(label_path, "w") as f: f.write(" ".join(label))
    print(f"{img_name}: annotated")

# Filling missing detections
print("\nFinalizing CVAT compatibility: Filling missing label files...")
for img_name in os.listdir(dest_image_dir):
    if img_name.lower().endswith((".jpg", ".png")):
        label_filename = os.path.splitext(img_name)[0] + ".txt"
        full_label_path = os.path.join(dest_label_dir, label_filename)
        
        if not os.path.exists(full_label_path):
            with open(full_label_path, "w") as f: pass # Create empty file
            print(f"{img_name}: created empty label")

print("\nProcess complete! All images now have corresponding label files.")
print("Annotations are saved at:", dest_label_dir)
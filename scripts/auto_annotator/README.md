🚀 How to Use

1. Prepare Your Files
Place your trained YOLO Pose weights (e.g., best.pt) and a folder containing your unlabeled images into the same directory as the script.

2. Update Configurations
Open auto_annotator.py and modify the "USER CONFIGURATIONS" section at the top of the file to match your project:
MODEL_PATH = "best.pt"             # Path to your trained model
IMAGE_DIR = "unlabeled_images"     # Folder containing images
OUTPUT_DIR = "cvat_dataset"        # Output folder name
CLASS_NAME = "object_name"         # Skeleton name in CVAT (e.g., gate, torpedo)
CLASS_ID = 0                       # Class ID
NUM_KPTS = 4                       # Number of keypoints

3. Run the Script
python auto_annotator.py
(The script will process the images and create a cvat_dataset_upload.zip file in your directory.)

📥 Importing to CVAT

    Open your CVAT Task. Make sure your images are already uploaded to the task.

    Ensure your Task's Label Constructor has a Skeleton type label with the exact same name as your CLASS_NAME (e.g., object_name) and the exact same number of keypoints.

    Click Menu -> Upload Annotations.

    Select the format: Ultralytics YOLO Pose 1.0.

    Upload the generated cvat_dataset_upload.zip file.

    Done! Review your frames and simply adjust any keypoints the AI misplaced, rather than labeling from scratch.

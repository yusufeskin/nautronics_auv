import os
import shutil

# --- Configuration ---
source_dir = os.getcwd()  # Current folder
train_dir = os.path.join(source_dir, "train")
val_dir = os.path.join(source_dir, "val")

# Create the directories 
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all files
files = os.listdir(source_dir)

print("Starting sort...")
count_moved = 0

for filename in files:
    # Checks for jpg and txt. If you use different file types you can change it.
    if filename.startswith("frame_") and (filename.endswith(".jpg") or filename.endswith(".txt")):
        
        try:
            # Extract number: frame_0004.jpg -> 0004
            number_part = filename.split("_")[1].split(".")[0]
            frame_number = int(number_part)

            # --- Logic ---
            target_folder = train_dir
            # If index is 4, 9, 14, etc., go to 'val'
            if frame_number % 5 == 4:
                target_folder = val_dir

            # Move the file
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_folder, filename)
            
            shutil.move(src_path, dst_path)
            
            if count_moved % 50 == 0:
                print(f"Moved: {filename} -> {os.path.basename(target_folder)}")
            
            count_moved += 1

        except (IndexError, ValueError):
            print(f"Skipping {filename}: unexpected name format.")

print(f"--- Done! Processed {count_moved} files. ---")

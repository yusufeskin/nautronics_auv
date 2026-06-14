> This README was written by Antigravity.
# ROS 2 Bag to MP4 Converter

Python scripts to convert ROS 2 bags (sqlite3/`.db3` format) containing raw or compressed image topics into MP4 videos.

These scripts use the pure-Python `rosbags` library, which means **you do NOT need a ROS 2 installation or active ROS environment** on the host machine to run them.

---

## Directory Structure

```
scripts/rosbag2mp4/
├── bags/                           # Directory to store your ROS 2 bags (git-ignored)
├── rosbag_to_mp4.py                # For raw image topics (sensor_msgs/msg/Image)
├── rosbag_compressed_to_mp4.py     # For compressed image topics (sensor_msgs/msg/CompressedImage)
└── README.md
```

---

## Requirements

Install the dependencies using pip:

```bash
pip install rosbags opencv-python numpy
```

---

## 1. Raw Images (`rosbag_to_mp4.py`)

Use this script if your bag contains raw, uncompressed camera streams (`sensor_msgs/msg/Image`).

### Basic Usage

```bash
# Automatically detects the first available raw image topic and outputs to output.mp4
python rosbag_to_mp4.py --bag bags/my_bag_folder
```

### Advanced Usage

```bash
python rosbag_to_mp4.py --bag bags/my_bag_folder --topic /camera/image_raw --output custom_output.mp4 --fps 15.0
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--bag` | *(Required)* | Path to the ROS 2 bag folder (containing `.db3` and `metadata.yaml`). |
| `--topic` | `None` | Name of the image topic to extract. If omitted, the first raw image topic is automatically chosen. |
| `--output` | `output.mp4` | Filename of the output MP4 video. |
| `--fps` | `10.0` | Output video framerate (frames per second). |

### Supported Encodings
The converter automatically handles and decodes the following ROS 2 raw image encodings:
- **Mono:** `mono8`, `8uc1`, `mono16`, `16uc1` (16-bit images are automatically normalized to 8-bit gray scale).
- **RGB / BGR:** `rgb8`, `bgr8`.
- **RGBA / BGRA:** `rgba8`, `bgra8`.
- **Bayer:** `bayer_*` (read as grayscale).

---

## 2. Compressed Images (`rosbag_compressed_to_mp4.py`)

Use this script if your bag contains compressed camera streams (`sensor_msgs/msg/CompressedImage`).

### Basic Usage

```bash
# Automatically detects the first available compressed topic and outputs to output.mp4
python rosbag_compressed_to_mp4.py --bag bags/my_bag_folder
```

### Advanced Usage

```bash
python rosbag_compressed_to_mp4.py --bag bags/my_bag_folder --topic /camera/image_raw/compressed --output custom_output.mp4 --fps 15.0
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--bag` | *(Required)* | Path to the ROS 2 bag folder (containing `.db3` and `metadata.yaml`). |
| `--topic` | `None` | Name of the compressed image topic to extract. If omitted, the first compressed topic is automatically chosen. |
| `--output` | `output.mp4` | Filename of the output MP4 video. |
| `--fps` | `10.0` | Output video framerate. |

### Supported Compression Formats
Supports common ROS 2 image compression formats, including **JPEG** and **PNG**, decoded directly using OpenCV.

---

## Troubleshooting

- **"No sensor_msgs/msg/Image topic found"**: Verify that the topic is indeed active and recorded in the bag. If the topic is compressed, use `rosbag_compressed_to_mp4.py` instead.
- **"ImportError: 'rosbags' kütüphanesi bulunamadı"**: Ensure you have run `pip install rosbags`.
- **FPS mismatch**: If the resulting video plays too fast or too slow, check your original topic pub rate (e.g., `ros2 topic hz /camera/image_raw` during recording) and set the `--fps` argument to match that frequency.

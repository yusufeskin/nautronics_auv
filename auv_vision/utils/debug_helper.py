import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from auv_interfaces.msg import DetectionArray


def draw_debug(frame, results, detections_msg: DetectionArray, model_type: str) -> np.ndarray:
    debug_frame = frame.copy()
    r = results[0]
    dets = detections_msg.detections

    if model_type in ('bbox', 'keypoint'):
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{dets[i].class_name} {dets[i].confidence:.2f}"
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(debug_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if model_type == 'keypoint':
            for det in dets:
                for idx in range(4):
                    cx, cy = int(det.keypoints[idx].x), int(det.keypoints[idx].y)
                    if cx == 0 and cy == 0:
                        continue  # skip uninitialised / zero keypoints
                    cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(debug_frame, str(idx), (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    elif model_type == 'obb':
        if r.obb is not None:
            for i, det in enumerate(r.obb):
                pts = det.xyxyxyxy[0].cpu().numpy().astype(int)
                cv2.polylines(debug_frame, [pts], isClosed=True,
                              color=(0, 255, 255), thickness=2)
                label = f"{dets[i].class_name} {dets[i].confidence:.2f}"
                cv2.putText(debug_frame, label, (pts[0][0], pts[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return debug_frame


def build_compressed_msg(frame, header, jpeg_quality=50) -> CompressedImage:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    success, encoded = cv2.imencode('.jpg', frame, encode_param)
    if not success:
        return None
    msg = CompressedImage()
    msg.header = header
    msg.format = "jpeg"
    msg.data = np.array(encoded).tobytes()
    return msg
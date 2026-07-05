#!/usr/bin/env python3
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from auv_interfaces.msg import DetectionArray

def draw_debug(frame, results, detections_msg: DetectionArray, model_type: str) -> np.ndarray:
    debug_frame = frame.copy()
    dets = detections_msg.detections

    if model_type in ('bbox', 'keypoint'):
        for det in dets:
            x1 = int(det.bbox_center_x - det.bbox_width / 2)
            y1 = int(det.bbox_center_y - det.bbox_height / 2)
            x2 = int(det.bbox_center_x + det.bbox_width / 2)
            y2 = int(det.bbox_center_y + det.bbox_height / 2)
            
            track_id = det.tracking_id
            is_ghost = (det.confidence == -1.0)
            
            color = (0, 165, 255) if is_ghost else (255, 0, 0)
            
            if is_ghost:
                label = f"GHOST ID:{track_id} {det.class_name}"
            else:
                if track_id != -1:
                    label = f"ID:{track_id} {det.class_name} {det.confidence:.2f}"
                else:
                    label = f"{det.class_name} {det.confidence:.2f}"
            
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if model_type == 'keypoint':
            for det in dets:
                for idx in range(4):
                    cx, cy = int(det.keypoints[idx].x), int(det.keypoints[idx].y)
                    if cx == 0 and cy == 0:
                        continue
                    
                    is_ghost = (det.confidence == -1.0)
                    kp_color = (0, 165, 255) if is_ghost else (0, 255, 0)
                    
                    cv2.circle(debug_frame, (cx, cy), 5, kp_color, -1)
                    cv2.putText(debug_frame, str(idx), (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    elif model_type == 'obb':
        for det in dets:
            cx, cy = det.bbox_center_x, det.bbox_center_y
            w, h = det.bbox_width, det.bbox_height
            angle = det.obb_rotation_angle
            pts = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle)))
            pts = np.int32(pts)
            
            is_ghost = (det.confidence == -1.0)
            color = (0, 165, 255) if is_ghost else (0, 255, 255)
            
            cv2.polylines(debug_frame, [pts], isClosed=True,
                          color=color, thickness=2)
            
            track_id = det.tracking_id
            if is_ghost:
                label = f"GHOST ID:{track_id} {det.class_name}"
            else:
                if track_id != -1:
                    label = f"ID:{track_id} {det.class_name} {det.confidence:.2f}"
                else:
                    label = f"{det.class_name} {det.confidence:.2f}"
                
            cv2.putText(debug_frame, label, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
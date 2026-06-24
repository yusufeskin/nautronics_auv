# yolo_helper.py

import numpy as np
from auv_interfaces.msg import DetectedObject, DetectionArray

MODEL_TYPE_BBOX      = "bbox"
MODEL_TYPE_OBB       = "obb"
MODEL_TYPE_KEYPOINT  = "keypoint"

def get_detections(result, header, class_names, model_type: str) -> DetectionArray:
    det_array = DetectionArray()
    det_array.header = header

    if model_type == MODEL_TYPE_OBB:
        if result.obb is None or len(result.obb) == 0:
            return det_array
        return _parse_obb(result, det_array, class_names)

    elif model_type == MODEL_TYPE_KEYPOINT:
        if result.boxes is None or len(result.boxes) == 0:
            return det_array
        return _parse_keypoint(result, det_array, class_names)

    else:  # default: bbox
        if result.boxes is None or len(result.boxes) == 0:
            return det_array
        return _parse_bbox(result, det_array, class_names)


def _make_base_obj(det, class_names) -> DetectedObject:
    """class_id, class_name, confidence her tipte ortak."""
    obj_msg = DetectedObject()
    cls_id = int(det.cls[0])
    obj_msg.class_id = cls_id
    try:
        obj_msg.class_name = class_names[cls_id]
    except (KeyError, IndexError):
        obj_msg.class_name = "unknown"
    obj_msg.confidence = float(det.conf[0])
    return obj_msg


def _parse_bbox(result, det_array, class_names) -> DetectionArray:
    for det in result.boxes:
        obj_msg = _make_base_obj(det, class_names)
        xywh = det.xywh[0].cpu().numpy()
        obj_msg.bbox_center_x = float(xywh[0])
        obj_msg.bbox_center_y = float(xywh[1])
        obj_msg.bbox_width    = float(xywh[2])
        obj_msg.bbox_height   = float(xywh[3])
        det_array.detections.append(obj_msg)
    return det_array


def _parse_obb(result, det_array, class_names) -> DetectionArray:
    for det in result.obb:
        obj_msg = _make_base_obj(det, class_names)
        xywhr = det.xywhr[0].cpu().numpy()
        obj_msg.bbox_center_x      = float(xywhr[0])
        obj_msg.bbox_center_y      = float(xywhr[1])
        obj_msg.bbox_width         = float(xywhr[2])
        obj_msg.bbox_height        = float(xywhr[3])
        obj_msg.obb_rotation_angle = float(xywhr[4])
        det_array.detections.append(obj_msg)
    return det_array


def _parse_keypoint(result, det_array, class_names) -> DetectionArray:
    kpts_batch = result.keypoints.xy.cpu().numpy()
    for i, det in enumerate(result.boxes):
        obj_msg = _make_base_obj(det, class_names)
        xywh = det.xywh[0].cpu().numpy()
        obj_msg.bbox_center_x = float(xywh[0])
        obj_msg.bbox_center_y = float(xywh[1])
        obj_msg.bbox_width    = float(xywh[2])
        obj_msg.bbox_height   = float(xywh[3])
        kpts = kpts_batch[i]
        for index in range(min(len(kpts), 4)):
            obj_msg.keypoints[index].x = float(kpts[index][0])
            obj_msg.keypoints[index].y = float(kpts[index][1])
            obj_msg.keypoints[index].z = 0.0
        det_array.detections.append(obj_msg)
    return det_array
#!/usr/bin/env python3
import numpy as np
from auv_interfaces.msg import DetectionArray

class EMAFilter:
    def __init__(self, alpha: float, distance_gate: float, miss_limit: int):
        self.alpha = alpha
        self.distance_gate = distance_gate
        self.miss_limit = miss_limit
        self.keypoint_history = {}

    def apply(self, detections_msg: DetectionArray, logger=None) -> DetectionArray:
        seen_ids = set()

        for det in detections_msg.detections:
            cls_id = det.class_id
            seen_ids.add(cls_id)

            raw_pts = np.array(
                [[det.keypoints[i].x, det.keypoints[i].y] for i in range(4)],
                dtype=np.float32
            )

            if cls_id in self.keypoint_history:
                prev_pts = self.keypoint_history[cls_id]['pts']

                max_dist = np.max(np.linalg.norm(raw_pts - prev_pts, axis=1))
                if max_dist > self.distance_gate:
                    self.keypoint_history[cls_id]['miss'] += 1
                    
                    if self.keypoint_history[cls_id]['miss'] >= self.miss_limit:
                        if logger:
                            logger.info(f'[EMA] cls={cls_id} uzun sure uzak kaldi. Yeni konuma kilitleniyor.')
                        smoothed = raw_pts
                        self.keypoint_history[cls_id] = {'pts': smoothed.copy(), 'miss': 0}
                    else:
                        smoothed = prev_pts
                        if logger:
                            logger.debug(
                                f'[EMA] cls={cls_id} gated (jump={max_dist:.1f}px > {self.distance_gate}px)'
                            )
                else:
                    smoothed = self.alpha * raw_pts + (1.0 - self.alpha) * prev_pts
                    self.keypoint_history[cls_id]['pts']  = smoothed
                    self.keypoint_history[cls_id]['miss'] = 0
            else:
                smoothed = raw_pts
                self.keypoint_history[cls_id] = {'pts': smoothed.copy(), 'miss': 0}

            for i in range(4):
                det.keypoints[i].x = float(smoothed[i][0])
                det.keypoints[i].y = float(smoothed[i][1])

        # Increment miss counter for classes absent this frame
        for cls_id in list(self.keypoint_history.keys()):
            if cls_id not in seen_ids:
                self.keypoint_history[cls_id]['miss'] += 1
                if self.keypoint_history[cls_id]['miss'] >= self.miss_limit:
                    if logger:
                        logger.info(
                            f'[EMA] cls={cls_id} evicted after {self.miss_limit} missed frames.'
                        )
                    del self.keypoint_history[cls_id]
                    
        return detections_msg

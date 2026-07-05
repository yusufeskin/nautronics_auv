#!/usr/bin/env python3
import copy
from auv_interfaces.msg import DetectionArray

class TrackManager:
    def __init__(self, min_hits: int, miss_limit: int):
        self.min_hits = min_hits
        self.miss_limit = miss_limit
        self.track_history = {}

    def process_tracks(self, detections_msg: DetectionArray, model) -> DetectionArray:
        current_track_ids = set()
        
        for det in detections_msg.detections:
            tid = det.tracking_id
            if tid == -1:
                continue
            current_track_ids.add(tid)
            if tid not in self.track_history:
                self.track_history[tid] = {'hits': 1, 'misses': 0, 'last_det': copy.deepcopy(det), 'is_confirmed': False}
            else:
                self.track_history[tid]['hits'] += 1
                self.track_history[tid]['misses'] = 0
                self.track_history[tid]['last_det'] = copy.deepcopy(det)
                
            if self.track_history[tid]['hits'] >= self.min_hits:
                self.track_history[tid]['is_confirmed'] = True

        # --- EXTRACT BOTSORT PREDICTIONS FOR LOST TRACKS ---
        lost_stracks_dict = {}
        if hasattr(model, 'predictor') and model.predictor is not None:
            trackers = getattr(model.predictor, 'trackers', [])
            if len(trackers) > 0:
                tracker = trackers[0]
                for t in getattr(tracker, 'lost_stracks', []):
                    # t.tlwh = [top_left_x, top_left_y, width, height]
                    pred_cx = t.tlwh[0] + t.tlwh[2] / 2.0
                    pred_cy = t.tlwh[1] + t.tlwh[3] / 2.0
                    lost_stracks_dict[t.track_id] = (pred_cx, pred_cy, t.tlwh[2], t.tlwh[3])

        final_detections = []
        for tid, state in list(self.track_history.items()):
            if tid not in current_track_ids:
                state['misses'] += 1
                
                if state['misses'] > self.miss_limit:
                    del self.track_history[tid]
                elif state['is_confirmed']:
                    # Coasting: inject ghost point
                    ghost_det = copy.deepcopy(state['last_det'])
                    ghost_det.confidence = -1.0
                    
                    # Apply SORT EKF Prediction if available
                    if tid in lost_stracks_dict:
                        pred_cx, pred_cy, pred_w, pred_h = lost_stracks_dict[tid]
                        dx = pred_cx - ghost_det.bbox_center_x
                        dy = pred_cy - ghost_det.bbox_center_y
                        
                        ghost_det.bbox_center_x = pred_cx
                        ghost_det.bbox_center_y = pred_cy
                        ghost_det.bbox_width = float(pred_w)
                        ghost_det.bbox_height = float(pred_h)
                        
                        for i in range(4):
                            if ghost_det.keypoints[i].x != 0 or ghost_det.keypoints[i].y != 0:
                                ghost_det.keypoints[i].x += dx
                                ghost_det.keypoints[i].y += dy
                    
                    final_detections.append(ghost_det)
            else:
                if state['is_confirmed']:
                    # Add original confirmed detection
                    for d in detections_msg.detections:
                        if d.tracking_id == tid:
                            final_detections.append(d)
                            break

        detections_msg.detections = final_detections
        return detections_msg

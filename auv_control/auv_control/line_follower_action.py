import math
import time
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from auv_interfaces.action import FollowLine


STATE_FOLLOWING     = "FOLLOWING"
STATE_TURNING_LEFT  = "TURNING_LEFT"
STATE_TURNING_RIGHT = "TURNING_RIGHT"
STATE_SEARCHING     = "SEARCHING"
STATE_FINISHED      = "FINISHED"

GRID_ROWS = 3
GRID_COLS = 3
ROW_TOP, ROW_MID, ROW_BOT = 0, 1, 2
COL_L,   COL_C,   COL_R   = 0, 1, 2


def compute_grid_density(mask: np.ndarray, rows: int = 3, cols: int = 3) -> np.ndarray:
    h, w = mask.shape
    cell_h = h // rows
    cell_w = w // cols
    density = np.zeros((rows, cols), dtype=float)

    for r in range(rows):
        for c in range(cols):
            y0 = r * cell_h
            y1 = y0 + cell_h if r < rows - 1 else h
            x0 = c * cell_w
            x1 = x0 + cell_w if c < cols - 1 else w
            cell = mask[y0:y1, x0:x1]
            if cell.size > 0:
                density[r, c] = np.count_nonzero(cell) / cell.size
    return density


def select_line_blob(mask: np.ndarray, min_area: int,
                     prev_centroid=None, track_weight: float = 1.2,
                     debug: bool = False):
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if n_lbl <= 1:
        return np.zeros_like(mask), None, "blob yok"

    h, w = mask.shape
    diag_len = float(max(w, h))
    diag_img = math.hypot(w, h)

    best_idx   = -1
    best_score = -1.0
    dbg_lines  = []

    for i in range(1, n_lbl):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x  = int(stats[i, cv2.CC_STAT_LEFT])
        y  = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        extent  = max(bw, bh) / diag_len
        fill    = area / float(max(bw * bh, 1))
        touches = (x <= 1) or (y <= 1) or (x + bw >= w - 1) or (y + bh >= h - 1)

        score = extent + 0.6 * (1.0 - fill) + (0.4 if touches else 0.0)

        cont = 0.0
        if prev_centroid is not None:
            d = math.hypot(centroids[i][0] - prev_centroid[0],
                           centroids[i][1] - prev_centroid[1])
            cont = track_weight * max(0.0, 1.0 - d / (0.5 * diag_img))
            score += cont

        if debug:
            dbg_lines.append(
                f"#{i} alan={area} ext={extent:.2f} fill={fill:.2f} "
                f"kenar={int(touches)} takip={cont:.2f} skor={score:.2f}"
            )

        if score > best_score:
            best_score = score
            best_idx   = i

    if best_idx < 0:
        return np.zeros_like(mask), None, "min_area'yi gecen blob yok"

    sel = (np.uint8(labels == best_idx) * 255)
    cen = (float(centroids[best_idx][0]), float(centroids[best_idx][1]))
    return sel, cen, (" | ".join(dbg_lines) if debug else "")


def measure_line_pose(mask: np.ndarray, prev_angle: float = 0.0,
                      band_top: float = 0.30, min_points: int = 60):
    h, w = mask.shape

    band  = mask[int(h * band_top):, :]
    pts   = cv2.findNonZero(band)
    y_off = int(h * band_top)

    if pts is None or len(pts) < min_points:
        pts   = cv2.findNonZero(mask)
        y_off = 0

    if pts is None or len(pts) < min_points:
        return None

    pts_f = pts.astype(np.float32).reshape(-1, 2)
    pts_f[:, 1] += y_off

    cx      = float(np.mean(pts_f[:, 0]))
    lateral = float(np.clip((cx - w / 2.0) / (w / 2.0), -1.0, 1.0))

    vx, vy, x0, y0 = cv2.fitLine(pts_f, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    if vy > 0:
        vx, vy = -vx, -vy

    angle = math.atan2(float(vx), -float(vy))

    if angle != 0.0:
        alt = angle - math.copysign(math.pi, angle)
        if abs(alt) <= math.pi / 2 + 1e-6 and abs(alt - prev_angle) < abs(angle - prev_angle):
            angle = alt

    heading = float(np.clip(angle / (math.pi / 2.0), -1.0, 1.0))
    return lateral, heading, (float(x0), float(y0), float(vx), float(vy))


def draw_grid_overlay(frame, density, state, lateral_err, heading_err,
                      fit=None, sway=0.0):
    vis = frame.copy()
    h, w = vis.shape[:2]
    rows, cols = density.shape
    cell_h = h // rows
    cell_w = w // cols

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1 = x0 + cell_w if c < cols - 1 else w
            y1 = y0 + cell_h if r < rows - 1 else h
            d = density[r, c]
            cv2.rectangle(vis, (x0, y0), (x1, y1),
                          (0, int(255 * (1 - d)), int(255 * d)), 1)
            cv2.putText(vis, f"{d:.2f}", (x0 + 5, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if fit is not None:
        x0, y0, vx, vy = fit
        L = max(w, h)
        cv2.line(vis,
                 (int(x0 - vx * L), int(y0 - vy * L)),
                 (int(x0 + vx * L), int(y0 + vy * L)),
                 (255, 0, 255), 2)
        cv2.arrowedLine(vis, (w // 2, h - 10), (w // 2, h - 60),
                        (255, 255, 255), 2, tipLength=0.3)

    if abs(sway) > 0.005:
        cy = h // 2
        ex = int(w // 2 + sway * 200)
        cv2.arrowedLine(vis, (w // 2, cy), (ex, cy), (0, 255, 255), 3, tipLength=0.3)

    color = (0, 165, 255) if state in (STATE_TURNING_LEFT, STATE_TURNING_RIGHT) \
        else (0, 255, 0)
    cv2.putText(vis, f"STATE: {state}", (5, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(vis, f"LAT: {lateral_err:+.3f}  HEAD: {heading_err:+.3f}  SWAY: {sway:+.3f}",
                (5, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return vis


class LineFollowerActionServer(Node):

    DEFAULT_CAMERA_TOPIC = "/camera/bottom"
    DEFAULT_TARGET_SPEED = 0.09

    DEFAULT_BLACK_VALUE_MAX = 130.0
    SAT_MAX                 = 30

    SHADOW_HUE_REJECT = True
    SHADOW_HUE_LO     = 75
    SHADOW_HUE_HI     = 105
    SHADOW_HUE_S_MIN  = 10

    LOOP_RATE_HZ = 20

    ROI_TOP    = 0.00
    ROI_BOTTOM = 1.00
    ROI_LEFT   = 0.00
    ROI_RIGHT  = 1.00

    USE_BLOB_SELECT    = True
    BLOB_TRACK_WEIGHT  = 1.2
    BLOB_TRACK_TIMEOUT = 1.0

    FIT_BAND_TOP   = 0.30
    FIT_MIN_POINTS = 60

    USE_SWAY   = True
    SWAY_SIGN  = -1.0
    Kp_SWAY    = 0.18
    MAX_SWAY   = 0.14
    SWAY_EMA_ALPHA = 0.30

    Kp_YAW = 0.32
    Kd_YAW = 0.03
    D_ERROR_CLAMP = 0.5

    W_LATERAL_YAW = 0.25

    PIVOT_ENTER_HEADING = 0.28
    PIVOT_EXIT_HEADING  = 0.12
    PIVOT_LATERAL_ENTER = 0.65

    PIVOT_MAX_ANGULAR = 0.28
    PIVOT_MIN_ANGULAR = 0.12
    PIVOT_LINEAR      = 0.0

    PIVOT_BRAKE        = -0.14
    PIVOT_BRAKE_CYCLES = 8

    SPEED_HEADING_PENALTY = 1.5
    SPEED_LATERAL_PENALTY = 0.4
    MIN_SPEED_SCALE       = 0.20

    MAX_ANGULAR_FOLLOW = 0.20
    SEARCH_ANGULAR     = 0.12
    ANGULAR_EMA_ALPHA  = 0.25
    PIVOT_EMA_ALPHA    = 0.45

    LINE_PRESENT_THRESHOLD = 0.02
    FINISH_THRESHOLD       = 0.55
    MIN_CONTOUR_AREA       = 500

    DBG_SAVE_EVERY  = 40
    DBG_BLOB_SCORES = False

    def __init__(self):
        super().__init__("line_follower_server")

        cb_group = ReentrantCallbackGroup()

        self.bridge = CvBridge()

        self._frame_lock = threading.Lock()
        self._current_frame: np.ndarray | None = None
        self._dbg_n = 0

        self._cmd_pub      = self.create_publisher(Twist, "/cmd_vel", 10)
        self._mask_pub     = self.create_publisher(Image, "/line_follower/mask", 10)
        self._mask_raw_pub = self.create_publisher(Image, "/line_follower/mask_raw", 10)
        self._vis_pub      = self.create_publisher(Image, "/line_follower/overlay", 10)

        self._action_server = ActionServer(
            self,
            FollowLine,
            "follow_line",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=cb_group,
        )

        self._camera_topic = self.DEFAULT_CAMERA_TOPIC
        self._image_sub = None
        self._subscribe_camera(self.DEFAULT_CAMERA_TOPIC, cb_group)

        self.get_logger().info("Line Follower Action Server hazir.")

    def _subscribe_camera(self, topic: str, cb_group=None) -> None:
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)

        self._image_sub = self.create_subscription(
            Image, topic, self._image_callback, 10, callback_group=cb_group,
        )
        self._camera_topic = topic
        self.get_logger().info(f"Kamera abone: {topic}")

    def _image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._frame_lock:
            self._current_frame = frame

    def goal_callback(self, goal_request):
        self.get_logger().info("Yeni hat takibi hedefi alindi.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Hat takibi iptali istendi.")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info("Hat takibi gorevi baslatiliyor...")
        result = FollowLine.Result()
        try:
            result = self._run_follow(goal_handle)
        except Exception as exc:
            import traceback
            self.get_logger().error(
                f"execute_callback EXCEPTION: {exc}\n{traceback.format_exc()}"
            )
            self._stop_robot()
            result.success      = False
            result.message      = f"EXCEPTION: {exc}"
            result.elapsed_time = 0.0
            goal_handle.abort()
        return result

    def _run_follow(self, goal_handle):
        req = goal_handle.request
        target_speed = req.target_speed    if req.target_speed    > 0 else self.DEFAULT_TARGET_SPEED
        timeout      = req.timeout         if req.timeout         > 0 else 0.0
        camera_topic = req.camera_topic    if req.camera_topic    else self._camera_topic
        black_v_max  = req.black_value_max if req.black_value_max > 0 else self.DEFAULT_BLACK_VALUE_MAX

        if camera_topic != self._camera_topic:
            self._subscribe_camera(camera_topic)

        feedback_msg     = FollowLine.Feedback()
        result           = FollowLine.Result()
        state            = STATE_SEARCHING
        pivoting         = False
        pivot_cycles     = 0
        prev_error       = 0.0
        prev_angle       = 0.0
        smoothed_angular = 0.0
        smoothed_sway    = 0.0
        prev_centroid    = None
        last_seen_time   = 0.0
        start_time       = time.time()
        loop_dt          = 1.0 / self.LOOP_RATE_HZ

        lower_black = np.array([0,   0,                 0],                dtype=np.uint8)
        upper_black = np.array([180, int(self.SAT_MAX), int(black_v_max)], dtype=np.uint8)

        self.get_logger().info(
            f"Parametreler -> hiz={target_speed}  timeout={timeout}s  "
            f"kamera={camera_topic}  V_max={black_v_max}  "
            f"sway={self.USE_SWAY} sign={self.SWAY_SIGN}  "
            f"pivot_gir={self.PIVOT_ENTER_HEADING}  pivot_cik={self.PIVOT_EXIT_HEADING}"
        )

        while rclpy.ok():
            loop_start = time.time()

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self._stop_robot()
                result.success      = False
                result.message      = "CANCELLED"
                result.elapsed_time = float(time.time() - start_time)
                self.get_logger().info("Gorev iptal edildi.")
                return result

            elapsed = time.time() - start_time
            if timeout > 0 and elapsed >= timeout:
                self._stop_robot()
                result.success      = False
                result.message      = "TIMEOUT"
                result.elapsed_time = float(elapsed)
                self.get_logger().warn("Gorev zaman asimina ugradi.")
                goal_handle.abort()
                return result

            with self._frame_lock:
                frame = self._current_frame.copy() if self._current_frame is not None else None

            if frame is None:
                time.sleep(loop_dt)
                continue

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_black, upper_black)

            if self.SHADOW_HUE_REJECT:
                h_ch, s_ch, _ = cv2.split(hsv)
                cyanish = ((h_ch >= self.SHADOW_HUE_LO) &
                           (h_ch <= self.SHADOW_HUE_HI) &
                           (s_ch >= self.SHADOW_HUE_S_MIN))
                mask[cyanish] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            mh, mw = mask.shape
            if self.ROI_TOP > 0.0:
                mask[:int(mh * self.ROI_TOP), :] = 0
            if self.ROI_BOTTOM < 1.0:
                mask[int(mh * self.ROI_BOTTOM):, :] = 0
            if self.ROI_LEFT > 0.0:
                mask[:, :int(mw * self.ROI_LEFT)] = 0
            if self.ROI_RIGHT < 1.0:
                mask[:, int(mw * self.ROI_RIGHT):] = 0

            mask_raw = mask.copy()

            blob_dbg = ""
            if self.USE_BLOB_SELECT:
                if prev_centroid is not None and \
                        (time.time() - last_seen_time) > self.BLOB_TRACK_TIMEOUT:
                    prev_centroid = None

                mask, cen, blob_dbg = select_line_blob(
                    mask, self.MIN_CONTOUR_AREA,
                    prev_centroid=prev_centroid,
                    track_weight=self.BLOB_TRACK_WEIGHT,
                    debug=self.DBG_BLOB_SCORES,
                )
                if cen is not None:
                    prev_centroid  = cen
                    last_seen_time = time.time()

            density       = compute_grid_density(mask, GRID_ROWS, GRID_COLS)
            line_coverage = float(np.mean(density))
            total_black   = float(np.sum(density))

            pose = measure_line_pose(mask, prev_angle,
                                     self.FIT_BAND_TOP, self.FIT_MIN_POINTS)
            if pose is None:
                lateral_error, heading_error, fit = 0.0, 0.0, None
                line_seen = False
            else:
                lateral_error, heading_error, fit = pose
                prev_angle = heading_error * (math.pi / 2.0)
                line_seen  = True

            if (density[ROW_TOP, COL_L] > self.FINISH_THRESHOLD and
                    density[ROW_TOP, COL_C] > self.FINISH_THRESHOLD and
                    density[ROW_TOP, COL_R] > self.FINISH_THRESHOLD and
                    density[ROW_MID, COL_L] > self.FINISH_THRESHOLD and
                    density[ROW_MID, COL_R] > self.FINISH_THRESHOLD):
                state    = STATE_FINISHED
                pivoting = False

            elif (not line_seen or
                  total_black < self.LINE_PRESENT_THRESHOLD * GRID_ROWS * GRID_COLS):
                state    = STATE_SEARCHING
                pivoting = False

            else:
                if pivoting:
                    if abs(heading_error) < self.PIVOT_EXIT_HEADING:
                        pivoting = False
                else:
                    if (abs(heading_error) > self.PIVOT_ENTER_HEADING or
                            abs(lateral_error) > self.PIVOT_LATERAL_ENTER):
                        pivoting     = True
                        pivot_cycles = 0

                if pivoting:
                    state = STATE_TURNING_RIGHT if heading_error >= 0 else STATE_TURNING_LEFT
                else:
                    state = STATE_FOLLOWING

            cmd = Twist()
            sway_cmd = 0.0

            if state == STATE_FINISHED:
                self._stop_robot()
                result.success      = True
                result.message      = STATE_FINISHED
                result.elapsed_time = float(elapsed)
                self.get_logger().info("Hat sonu tespit edildi! Gorev tamamlandi.")
                goal_handle.succeed()
                return result

            elif state == STATE_SEARCHING:
                cmd.linear.x     = 0.0
                raw_angular      = self.SEARCH_ANGULAR * (1.0 if prev_error >= 0 else -1.0)
                smoothed_angular = (self.ANGULAR_EMA_ALPHA * raw_angular +
                                    (1.0 - self.ANGULAR_EMA_ALPHA) * smoothed_angular)
                smoothed_sway    = 0.0
                cmd.angular.z    = smoothed_angular
                prev_error       = 0.0

            elif pivoting:
                yaw_error = heading_error
                d_error   = float(np.clip(yaw_error - prev_error,
                                          -self.D_ERROR_CLAMP, self.D_ERROR_CLAMP))

                raw_angular = self.Kp_YAW * yaw_error + self.Kd_YAW * d_error / loop_dt
                raw_angular = float(np.clip(raw_angular,
                                            -self.PIVOT_MAX_ANGULAR,
                                            self.PIVOT_MAX_ANGULAR))

                if 0.0 < abs(raw_angular) < self.PIVOT_MIN_ANGULAR:
                    raw_angular = math.copysign(self.PIVOT_MIN_ANGULAR, raw_angular)

                smoothed_angular = (self.PIVOT_EMA_ALPHA * raw_angular +
                                    (1.0 - self.PIVOT_EMA_ALPHA) * smoothed_angular)

                if self.USE_SWAY:
                    raw_sway = self.SWAY_SIGN * self.Kp_SWAY * lateral_error
                    raw_sway = float(np.clip(raw_sway, -self.MAX_SWAY, self.MAX_SWAY))
                    smoothed_sway = (self.SWAY_EMA_ALPHA * raw_sway +
                                     (1.0 - self.SWAY_EMA_ALPHA) * smoothed_sway)
                    sway_cmd = smoothed_sway

                pivot_cycles += 1
                if pivot_cycles <= self.PIVOT_BRAKE_CYCLES:
                    cmd.linear.x = float(self.PIVOT_BRAKE)
                else:
                    cmd.linear.x = float(self.PIVOT_LINEAR)

                cmd.linear.y  = float(sway_cmd)
                cmd.angular.z = smoothed_angular
                prev_error    = yaw_error

            else:
                yaw_error = heading_error + self.W_LATERAL_YAW * lateral_error
                d_error   = float(np.clip(yaw_error - prev_error,
                                          -self.D_ERROR_CLAMP, self.D_ERROR_CLAMP))

                raw_angular = self.Kp_YAW * yaw_error + self.Kd_YAW * d_error / loop_dt
                raw_angular = float(np.clip(raw_angular,
                                            -self.MAX_ANGULAR_FOLLOW,
                                            self.MAX_ANGULAR_FOLLOW))

                smoothed_angular = (self.ANGULAR_EMA_ALPHA * raw_angular +
                                    (1.0 - self.ANGULAR_EMA_ALPHA) * smoothed_angular)

                if self.USE_SWAY:
                    raw_sway = self.SWAY_SIGN * self.Kp_SWAY * lateral_error
                    raw_sway = float(np.clip(raw_sway, -self.MAX_SWAY, self.MAX_SWAY))
                    smoothed_sway = (self.SWAY_EMA_ALPHA * raw_sway +
                                     (1.0 - self.SWAY_EMA_ALPHA) * smoothed_sway)
                    sway_cmd = smoothed_sway

                speed_scale = (1.0
                               - self.SPEED_HEADING_PENALTY * abs(heading_error)
                               - self.SPEED_LATERAL_PENALTY * abs(lateral_error))
                speed_scale = max(self.MIN_SPEED_SCALE, speed_scale)

                cmd.linear.x  = float(target_speed * speed_scale)
                cmd.linear.y  = float(sway_cmd)
                cmd.angular.z = smoothed_angular
                prev_error    = yaw_error

            self._cmd_pub.publish(cmd)

            feedback_msg.lateral_error = float(lateral_error)
            feedback_msg.heading_error = float(heading_error)
            feedback_msg.line_coverage = float(line_coverage)
            feedback_msg.current_state = state
            goal_handle.publish_feedback(feedback_msg)

            vis = draw_grid_overlay(frame, density, state,
                                    lateral_error, heading_error, fit, sway_cmd)
            self._mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, encoding="mono8"))
            self._mask_raw_pub.publish(self.bridge.cv2_to_imgmsg(mask_raw, encoding="mono8"))
            self._vis_pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding="bgr8"))

            self._dbg_n += 1
            if self._dbg_n % self.DBG_SAVE_EVERY == 0:
                cv2.imwrite("/tmp/lf_frame.png",   frame)
                cv2.imwrite("/tmp/lf_mask.png",    mask)
                cv2.imwrite("/tmp/lf_overlay.png", vis)
                self.get_logger().info(
                    f"[dbg] {state}  lat={lateral_error:+.3f}  head={heading_error:+.3f}  "
                    f"aci={math.degrees(heading_error * math.pi / 2):+.0f}deg  "
                    f"vx={cmd.linear.x:+.3f}  vy={cmd.linear.y:+.3f}  wz={cmd.angular.z:+.3f}"
                )
                if blob_dbg:
                    self.get_logger().info(f"[blob] {blob_dbg}")

            elapsed_loop = time.time() - loop_start
            sleep_time   = loop_dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._stop_robot()
        result.success      = False
        result.message      = "SHUTDOWN"
        result.elapsed_time = float(time.time() - start_time)
        goal_handle.abort()
        return result

    def _stop_robot(self) -> None:
        self._cmd_pub.publish(Twist())
        self.get_logger().info("Robot durduruldu.")


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerActionServer()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
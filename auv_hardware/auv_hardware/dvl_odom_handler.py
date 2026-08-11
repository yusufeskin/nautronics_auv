#!/usr/bin/env python3
from pymavlink import mavutil
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from marine_acoustic_msgs.msg import Dvl

NAN = float('nan')
LARGE_VAR = 1.0e6  # "guvenme" seviyesinde buyuk varyans (m^2 veya (m/s)^2 mertebesinde)
STALE_THRESHOLD_SEC = 0.5 


UPPER_TRI_DIAG_IDX = [0, 6, 11, 15, 18, 20]


class DvlOdomHandler:
    def __init__(self, node, master, logger):
        self.node = node
        self.master = master
        self.logger = logger

        self._quality = -1
        self._beams_locked = False
        self._reset_counter = 0

        self._pose = None            # (x, y, z, qw, qx, qy, qz)
        self._pose_cov21 = None
        self._pose_recv_time = None  # rclpy Time

        self._twist = None           # (vx, vy, vz)
        self._twist_cov21 = None
        self._twist_recv_time = None

        self.velocity_sub = node.create_subscription(
            Dvl,
            '/waterlinked_dvl_driver/velocity_report',
            self.velocity_callback,
            10
        )
        self.dead_reckoning_sub = node.create_subscription(
            PoseWithCovarianceStamped,
            '/waterlinked_dvl_driver/dead_reckoning_report',
            self.dead_reckoning_callback,
            10
        )
        self.odom_sub = node.create_subscription(
            Odometry,
            '/waterlinked_dvl_driver/odom',
            self.odom_callback,
            10
        )

        self.logger.info(
            "DvlOdomHandler baslatildi: velocity_report (gate) + "
            "dead_reckoning_report (pose) + odom (twist) dinleniyor."
        )


    def velocity_callback(self, msg: Dvl):
        beams_locked_now = bool(msg.beam_ranges_valid and msg.beam_velocities_valid
                                 and msg.num_good_beams >= 3)

        if msg.num_good_beams >= 4 and beams_locked_now:
            self._quality = 100
        elif msg.num_good_beams == 3 and beams_locked_now:
            self._quality = 60
        else:
            self._quality = -1  # ODOMETRY spec: -1 = odometry has failed

        if beams_locked_now and not self._beams_locked:
            self._reset_counter = (self._reset_counter + 1) % 256
            self.logger.info(f"DVL kilidi yeniden kazanildi, reset_counter={self._reset_counter}")
            self._pose = None
            self._twist = None
        elif not beams_locked_now and self._beams_locked:
            self.logger.warn("DVL kilidi kayboldu, odometri gonderimi durduruldu.")

        self._beams_locked = beams_locked_now

    def dead_reckoning_callback(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
        self._pose_cov21 = self._sanitize(self._to_upper_triangular(msg.pose.covariance))
        self._pose_recv_time = self.node.get_clock().now()
        self._try_send()

    def odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear
        self._twist = (v.x, v.y, v.z)
        twist_cov21 = self._sanitize(self._to_upper_triangular(msg.twist.covariance))
        for idx in UPPER_TRI_DIAG_IDX[3:]:
            twist_cov21[idx] = LARGE_VAR
        self._twist_cov21 = twist_cov21
        self._twist_recv_time = self.node.get_clock().now()
        self._try_send()

    def _try_send(self):
        if not self._beams_locked or self._quality <= 0:
            return
        if self._pose is None or self._twist is None:
            return

        now = self.node.get_clock().now()
        pose_age = (now - self._pose_recv_time).nanoseconds / 1e9
        twist_age = (now - self._twist_recv_time).nanoseconds / 1e9

        x, y, z, qw, qx, qy, qz = self._pose
        vx, vy, vz = self._twist

        pose_cov21 = list(self._pose_cov21)
        twist_cov21 = list(self._twist_cov21)

        if pose_age > STALE_THRESHOLD_SEC:
            for idx in UPPER_TRI_DIAG_IDX[:3]:
                pose_cov21[idx] = LARGE_VAR
        if twist_age > STALE_THRESHOLD_SEC:
            for idx in UPPER_TRI_DIAG_IDX[:3]:
                twist_cov21[idx] = LARGE_VAR

        try:
            self.master.mav.odometry_send(
                0,                                          # time_usec=0 -> alis aninda damgala
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,         # frame_id (pozisyon)
                mavutil.mavlink.MAV_FRAME_BODY_FRD,          # child_frame_id (hiz, govde eksenli)
                x, y, z,
                [qw, qx, qy, qz],
                vx, vy, vz,
                NAN, NAN, NAN,                               # acisal hiz: guvenilir degil
                pose_cov21,
                twist_cov21,
                self._reset_counter,
                mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
                self._quality
            )
        except Exception as e:
            self.logger.error(f"ODOMETRY gonderilemedi: {e}")

    @staticmethod
    def _to_upper_triangular(cov36):
        n = 6
        result = []
        for i in range(n):
            for j in range(i, n):
                result.append(cov36[i * n + j])
        return result

    @staticmethod
    def _sanitize(cov21):
        return [LARGE_VAR if v < 0 else v for v in cov21]
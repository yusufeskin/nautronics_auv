#!/usr/bin/env python3
import math
from pymavlink import mavutil
import message_filters
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from marine_acoustic_msgs.msg import Dvl

NAN = float('nan')
LARGE_VAR = 1.0e6

UPPER_TRI_DIAG_IDX = [0, 6, 11, 15, 18, 20]

class DvlOdomHandler:
    def __init__(self, node, master, logger):
        self.node = node
        self.master = master
        self.logger = logger

        self._quality = -1
        self._beams_locked = False
        self._reset_counter = 0
        self._static_pose_cov21 = [0.0] * 21
        for idx in UPPER_TRI_DIAG_IDX:
            self._static_pose_cov21[idx] = LARGE_VAR

        self.velocity_sub = node.create_subscription(
            Dvl, '/waterlinked_dvl_driver/velocity_report', self.velocity_callback, 10
        )

        self.pose_sub = message_filters.Subscriber(
            node, PoseWithCovarianceStamped, '/waterlinked_dvl_driver/dead_reckoning_report'
        )
        self.odom_sub = message_filters.Subscriber(
            node, Odometry, '/waterlinked_dvl_driver/odom'
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.odom_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.logger.info("DvlOdomHandler (TimeSynchronizer) başlatıldı.")

    def velocity_callback(self, msg: Dvl):
        beams_locked_now = bool(msg.beam_ranges_valid and msg.beam_velocities_valid and msg.num_good_beams >= 3)

        if msg.num_good_beams >= 4 and beams_locked_now:
            self._quality = 100
        elif msg.num_good_beams == 3 and beams_locked_now:
            self._quality = 60
        else:
            self._quality = -1 

        if beams_locked_now and not self._beams_locked:
            self._reset_counter = (self._reset_counter + 1) % 256
            self.logger.info(f"DVL kilidi yeniden kazanıldı, reset_counter={self._reset_counter}")
        elif not beams_locked_now and self._beams_locked:
            self.logger.warn("DVL kilidi kayboldu, ODOMETRY akışı (EKF timeout'u için) durduruldu.")

        self._beams_locked = beams_locked_now

    def sync_callback(self, pose_msg: PoseWithCovarianceStamped, odom_msg: Odometry):
        if not self._beams_locked or self._quality <= 0:
            return
            
        self._send_mavlink_odom(pose_msg, odom_msg)

    def _send_mavlink_odom(self, pose_msg, odom_msg):
        p = pose_msg.pose.pose.position
        q = pose_msg.pose.pose.orientation
        
 
        v = odom_msg.twist.twist.linear
        
        twist_cov21 = self._sanitize(self._to_upper_triangular(odom_msg.twist.covariance))
        twist_cov21[15] = LARGE_VAR
        twist_cov21[18] = LARGE_VAR
        twist_cov21[20] = LARGE_VAR


        try:
            self.master.mav.odometry_send(
                0,                                          # time_usec
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,        # frame_id
                mavutil.mavlink.MAV_FRAME_BODY_FRD,         # child_frame_id
                p.x, p.y, p.z,                              
                [q.w, q.x, q.y, q.z],
                v.x, v.y, v.z,                              
                NAN, NAN, NAN,                              
                self._static_pose_cov21,                                 
                twist_cov21,                                
                self._reset_counter,
                mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
                self._quality
            )
        except Exception as e:
            self.logger.error(f"ODOMETRY gönderilemedi: {e}")

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
        return [LARGE_VAR if math.isclose(v, -1.0) or v < 0 else v for v in cov21]
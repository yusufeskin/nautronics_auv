#!/usr/bin/env python3

import math
import time
from collections import deque

from pymavlink import mavutil
from nav_msgs.msg import Odometry
from marine_acoustic_msgs.msg import Dvl

ATTITUDE_TIMEOUT_S = 0.5
LOG_THROTTLE_S = 5.0

LOCK_LOSS_INVALID_STREAK = 4

MAX_SPEED_MPS = 2.0
MAX_ACCEL_MPS2 = 5.0
MAX_DT_S = 1.0

FOM_STD_SATURATION_MPS = 0.02
GOOD_BEAMS_FULL = 4
PARTIAL_BEAM_CONFIDENCE_FACTOR = 0.6

ATTITUDE_BUFFER_LEN = 6


class DvlOdomHandler:
    def __init__(self, node, master, logger):
        self.node = node
        self.master = master
        self.logger = logger

        self._locked = False
        self._invalid_streak = 0
        self._confidence = 0.0

        self._att_buf = deque(maxlen=ATTITUDE_BUFFER_LEN)
        self._att_mono = 0.0

        self._prev_time = None
        self._prev_velocity = None
        self._prev_att = None

        node.create_subscription(
            Dvl, '/waterlinked_dvl_driver/velocity_report',
            self.velocity_callback, 10
        )
        node.create_subscription(
            Odometry, '/waterlinked_dvl_driver/odom',
            self.odom_callback, 10
        )

        self.logger.info("DvlOdomHandler baslatildi (VISION_POSITION_DELTA, attitude kaynagi: Pixhawk).")

    def handle_attitude(self, msg):
        self._att_buf.append((time.time(), msg.roll, msg.pitch, msg.yaw))
        self._att_mono = time.monotonic()

    def velocity_callback(self, msg: Dvl):
        raw_valid = bool(msg.beam_velocities_valid) and msg.num_good_beams >= 3

        if raw_valid:
            self._invalid_streak = 0
            self._confidence = self._compute_confidence(msg)
            if not self._locked:
                self._locked = True
                self._prev_time = None
                self._prev_velocity = None
                self._prev_att = None
                self.logger.info("DVL kilidi kazanildi.")
            return

        self._invalid_streak += 1
        if self._locked and self._invalid_streak >= LOCK_LOSS_INVALID_STREAK:
            self._locked = False
            self.logger.warn(
                "DVL kilidi kayboldu (art arda gecersiz rapor); VISION_POSITION_DELTA durduruldu."
            )

    def odom_callback(self, msg: Odometry):
        if not self._locked:
            return

        now_mono = time.monotonic()
        if self._att_mono == 0.0 or (now_mono - self._att_mono) > ATTITUDE_TIMEOUT_S:
            self.logger.warn(
                "Pixhawk ATTITUDE bayat/yok; VISION_POSITION_DELTA gonderilmiyor.",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return

        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1.0e-9

        v = msg.twist.twist.linear
        vx, vy, vz = v.x, v.y, v.z

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > MAX_SPEED_MPS:
            self.logger.warn(
                f"DVL hizi fiziksel siniri asiyor ({speed:.2f} m/s); ornek atlandi.",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return

        if self._prev_time is None:
            self._prev_time = t
            self._prev_velocity = (vx, vy, vz)
            self._prev_att = self._attitude_at(t)
            return

        dt = t - self._prev_time
        if dt <= 0.0 or dt > MAX_DT_S:
            self._prev_time = t
            self._prev_velocity = (vx, vy, vz)
            self._prev_att = self._attitude_at(t)
            return

        pvx, pvy, pvz = self._prev_velocity
        accel = math.dist((vx, vy, vz), (pvx, pvy, pvz)) / dt
        if accel > MAX_ACCEL_MPS2:
            self.logger.warn(
                f"DVL orneginde imkansiz ivme sicramasi ({accel:.2f} m/s^2); ornek atlandi.",
                throttle_duration_sec=LOG_THROTTLE_S,
            )
            return

        att_now = self._attitude_at(t)
        if self._prev_att is None or att_now is None:
            self._prev_time = t
            self._prev_velocity = (vx, vy, vz)
            self._prev_att = att_now
            return

        position_delta = [
            0.5 * (vx + pvx) * dt,
            0.5 * (vy + pvy) * dt,
            0.5 * (vz + pvz) * dt,
        ]
        angle_delta = [
            self._wrap_angle(att_now[i] - self._prev_att[i]) for i in range(3)
        ]

        time_usec = int(t * 1.0e6)
        time_delta_usec = int(dt * 1.0e6)

        self.logger.info(
            f"VISION_POSITION_DELTA yazilim/ag gecikmesi (DVL ic gecikmesi haric): "
            f"{(time.time() - t) * 1000.0:.1f} ms",
            throttle_duration_sec=LOG_THROTTLE_S,
        )

        self._send_vision_position_delta(
            time_usec, time_delta_usec, angle_delta, position_delta, self._confidence
        )

        self._prev_time = t
        self._prev_velocity = (vx, vy, vz)
        self._prev_att = att_now

    def _send_vision_position_delta(self, time_usec, time_delta_usec, angle_delta, position_delta, confidence):
        try:
            self.master.mav.vision_position_delta_send(
                time_usec,
                time_delta_usec,
                angle_delta,
                position_delta,
                confidence,
            )
        except Exception as e:
            self.logger.error(f"VISION_POSITION_DELTA gonderilemedi: {e}")

    def _attitude_at(self, t):
        buf = self._att_buf
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][1], buf[0][2], buf[0][3]
        if t >= buf[-1][0]:
            return buf[-1][1], buf[-1][2], buf[-1][3]

        for i in range(1, len(buf)):
            t0, r0, p0, y0 = buf[i - 1]
            t1, r1, p1, y1 = buf[i]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return (
                    self._lerp_angle(r0, r1, frac),
                    self._lerp_angle(p0, p1, frac),
                    self._lerp_angle(y0, y1, frac),
                )

        return buf[-1][1], buf[-1][2], buf[-1][3]

    @staticmethod
    def _compute_confidence(msg: Dvl):
        variances = [msg.velocity_covar[0], msg.velocity_covar[4], msg.velocity_covar[8]]
        finite = [v for v in variances if math.isfinite(v) and v >= 0.0]

        if finite:
            std = math.sqrt(max(finite))
            base = 100.0 * (1.0 - std / FOM_STD_SATURATION_MPS)
        else:
            base = 50.0

        beam_factor = 1.0 if msg.num_good_beams >= GOOD_BEAMS_FULL else PARTIAL_BEAM_CONFIDENCE_FACTOR
        return max(0.0, min(100.0, base * beam_factor))

    @staticmethod
    def _wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @classmethod
    def _lerp_angle(cls, a0, a1, frac):
        return cls._wrap_angle(a0 + cls._wrap_angle(a1 - a0) * frac)

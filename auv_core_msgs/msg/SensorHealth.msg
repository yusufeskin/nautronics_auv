# Health of one sensor as judged by its gate node (Layer 1).
#
# Every gate publishes this at a fixed rate whether or not samples are
# arriving, so a silent sensor is distinguishable from a missing node.
# The estimator (Layer 2) reads `fusable` to decide whether the sensor's
# latest output may be used; everything else on this message is for
# operators and post-run log analysis.

std_msgs/Header header

# Short stable name of the gated sensor, e.g. "dvl", "imu", "depth".
string sensor

uint8 STATUS_OK=0         # samples arriving and passing every check
uint8 STATUS_DEGRADED=1   # still usable, but quality is reduced (see reason)
uint8 STATUS_REJECTED=2   # the most recent sample failed a check
uint8 STATUS_STALE=3      # no sample within the configured timeout
uint8 STATUS_NO_DATA=4    # nothing has ever been received
uint8 status

# True when the estimator may fuse this sensor's most recent output.
# DEGRADED is still fusable (with inflated covariance); the rest are not.
bool fusable

# Human-readable explanation for any non-OK status. Empty when OK.
string reason

float32 measured_rate_hz
float32 expected_rate_hz

uint64 accepted_count
uint64 rejected_count

# rejected / (accepted + rejected) over the trailing health window, not
# over the whole run - a run-long ratio hides a sensor that just started
# failing.
float32 reject_ratio

#!/bin/bash
set -e # stop if face an error
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"

# source
source /opt/ros/humble/install/setup.bash
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"

#this file is used as docker entrypoint, once container started its executed 
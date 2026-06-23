#!/bin/bash
set -e # stop if face an error

# source
source /opt/ros/humble/install/setup.bash
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"

#this file is used as docker entrypoint, once container started its executed 
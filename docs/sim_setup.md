# Simulation Environment Setup

This guide sets up a development PC to run the Nautronics AUV simulation: Gazebo Harmonic + ArduSub SITL + ROS 2 Humble.

Do the steps **in order**. Each step ends with a check; don't move on until it passes.

## Versions we use

| Component | Version | How it's installed |
|---|---|---|
| OS | Ubuntu **22.04** (Jammy) | Native install recommended. VM or WSL2 works but Gazebo will be slow. |
| ROS 2 | **Humble** | apt (`ros-humble-desktop`) |
| Gazebo | **Harmonic** (gz-sim 8) | apt, from `packages.osrfoundation.org` |
| ArduPilot | `master`, built for **SITL** / **ArduSub** | from source |
| ArduPilot Gazebo plugin | `main` | from source |
| ros_gz (ROS ↔ Gazebo bridge) | `humble` branch, built against **Harmonic** | **from source** |

> ⚠️ **Why is ros_gz built from source?**
> ROS 2 Humble officially pairs with Gazebo **Fortress**, not Harmonic. The binary packages (`ros-humble-ros-gz*`) are built for Fortress, so installing them pulls Fortress in and breaks our Harmonic setup. Building ros_gz from source with `GZ_VERSION=harmonic` links it against the Harmonic we installed.
>
> **Never install** `ros-humble-ros-gz`, `ros-humble-ros-gz-*`, `ros-humble-ros-ign*`, `ros-humble-gazebo-ros-pkgs` or `ros-humble-desktop-full`.

## 1. ROS 2 Humble

Follow the official guide: <https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>.

Install **`ros-humble-desktop`** (not `desktop-full`) together with the dev tools:

```bash
sudo apt install ros-humble-desktop ros-dev-tools python3-pip
```

Source ROS in every terminal and initialize rosdep:

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc

sudo rosdep init   # skip if it says it's already initialized
rosdep update
```

**Check:**

```bash
ros2 run demo_nodes_cpp talker
```

You should see `Publishing: 'Hello World: 1'` messages. Stop it with Ctrl+C.

## 2. Gazebo Harmonic

Official guide: <https://gazebosim.org/docs/harmonic/install_ubuntu>.

First install some necessary tools:

```bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg
```

Then install Gazebo Harmonic:

```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```

Also install the Gazebo Python bindings. Our simulated DVL bridge (`sim_dvl_bridge`) talks to Gazebo directly through them:

```bash
sudo apt-get install python3-gz-transport13 python3-gz-msgs10
```

Tell every later build (ros_gz, ardupilot_gazebo) which Gazebo version to use:

```bash
echo 'export GZ_VERSION=harmonic' >> ~/.bashrc
source ~/.bashrc
```

**Check:**

```bash
gz sim --version     # must print 8.x.x
gz sim shapes.sdf    # a window with a few shapes should open
```

## 3. ArduPilot and SITL (ArduSub)

Official guide: <https://ardupilot.org/dev/docs/building-setup-linux.html>.

Clone ArduPilot together with its submodules:

```bash
cd ~
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
```

Install the build prerequisites. The script also adds ArduPilot's tools (`sim_vehicle.py`, `mavproxy.py`) to your `PATH` through `~/.profile`.

```bash
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
```

> If later commands say `sim_vehicle.py: command not found`, log out and back in (or reboot) so `~/.profile` is reloaded.

Build ArduSub for SITL (Software In The Loop): the board is `sitl` and the vehicle is `sub`.

```bash
./waf configure --board sitl
./waf sub
```

**Check:** run ArduSub SITL on its own, without Gazebo:

```bash
cd ~/ardupilot/ArduSub
sim_vehicle.py -v ArduSub -w --console --map
```

`-w` wipes stored parameters to defaults, which you only need on the first run. A MAVProxy console and a map window should open, and the console should show `STABILIZE>` or `MANUAL>` within a minute. Close it with Ctrl+C.

## 4. ArduPilot Gazebo plugin

Repository: <https://github.com/ArduPilot/ardupilot_gazebo>.

This plugin (`libArduPilotPlugin.so`) connects Gazebo to ArduPilot SITL. Our vehicle model loads it by name.

Install the dependencies for Harmonic:

```bash
sudo apt update
sudo apt install libgz-sim8-dev rapidjson-dev
sudo apt install libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-gl
```

Clone and build:

```bash
cd ~
git clone https://github.com/ArduPilot/ardupilot_gazebo
cd ardupilot_gazebo
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j4
```

Tell Gazebo where the plugin and its example models are. **Both lines are required.** Without `GZ_SIM_SYSTEM_PLUGIN_PATH`, Gazebo silently skips the plugin and SITL never connects.

```bash
echo 'export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}' >> ~/.bashrc
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
source ~/.bashrc
```

**Check:**

```bash
ls ~/ardupilot_gazebo/build/libArduPilotPlugin.so
```

**Optional end-to-end check** with the plugin's own example drone. The first run builds ArduCopter, which takes a few minutes.

```bash
# Terminal 1
gz sim -v4 -r iris_runway.sdf

# Terminal 2
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console
```

If SITL stops printing `No JSON sensor message received` and the console reaches `STABILIZE>`, Gazebo and SITL are talking.

## 5. ros_gz (from source, against Harmonic)

ros_gz bridges Gazebo topics to ROS 2 and provides the `ros_gz_sim` launch files our `gazebo.launch.py` uses. It lives in its **own workspace** so it is built once and not rebuilt with our code.

First, add OSRF's rosdep rules. They let rosdep resolve the Harmonic library names (`gz-sim8`, `gz-transport13`, ...) to the packages installed in step 2, instead of failing or reaching for Fortress:

```bash
sudo bash -c 'wget https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list -O /etc/ros/rosdep/sources.list.d/00-gazebo.list'
rosdep update
rosdep resolve gz-harmonic   # should resolve without errors
```

Clone the `humble` branch:

```bash
mkdir -p ~/ros_gz_ws/src
cd ~/ros_gz_ws/src
git clone https://github.com/gazebosim/ros_gz.git -b humble
```

Install the dependencies and build. **`GZ_VERSION` must be `harmonic`.** If you followed step 2 it already is; `echo $GZ_VERSION` to be sure.

```bash
cd ~/ros_gz_ws
echo $GZ_VERSION     # must print: harmonic
rosdep install -r --from-paths src -i -y --rosdistro humble
source /opt/ros/humble/setup.bash
colcon build
```

> **Build killed / out of memory?** ros_gz is template-heavy and the compiler can use a lot of RAM. If you see `c++: fatal error: Killed signal terminated program cc1plus`, build one package at a time:
>
> ```bash
> export MAKEFLAGS="-j 1"
> colcon build --parallel-workers=1 --executor sequential
> ```

> **`Could not find a package configuration file provided by "actuator_msgs"`?** Clone it next to ros_gz and build again:
>
> ```bash
> cd ~/ros_gz_ws/src
> git clone https://github.com/rudislabs/actuator_msgs.git
> cd ~/ros_gz_ws && colcon build
> ```

Source the workspace in every terminal, **after** ROS and **before** our own workspace:

```bash
echo 'source ~/ros_gz_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

**Check:**

```bash
ros2 pkg prefix ros_gz_sim        # must point into ~/ros_gz_ws/install, not /opt/ros/humble
apt list --installed 2>/dev/null | grep -E 'ros-humble-ros-(gz|ign)'   # must print nothing
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:=shapes.sdf           # Gazebo Harmonic opens
```

## Final `~/.bashrc`

After all steps, the end of your `~/.bashrc` should look like this. **The order matters**: each `source` line builds on the one above it.

```bash
source /opt/ros/humble/setup.bash
export GZ_VERSION=harmonic
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:${GZ_SIM_RESOURCE_PATH}
source ~/ros_gz_ws/install/setup.bash
```

The ArduPilot `PATH` line lives in `~/.profile`, not `~/.bashrc`; the prerequisites script puts it there.

Your simulation environment is ready. Next: [build the Nautronics workspace](../README.md#installation).

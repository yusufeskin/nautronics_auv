## Requirements

- **ArduPilot**
  https://github.com/ArduPilot/ardupilot

- **ArduPilot Gazebo**  
  https://github.com/ArduPilot/ardupilot_gazebo

- **Gazebo Sim (Harmonic for ROS2)**  
  https://gazebosim.org/docs/latest/getstarted/


## Installation

Aşağıdaki adımları takip ederek projeyi kurabilirsiniz:

```bash
mkdir -p ~/nautronics_auv/src
cd ~/nautronics_auv/src
```

```bash
git clone https://github.com/yusufeskin/nautronics_auv
cd ..

vcs import src < src/nautronics_auv/nautronics.repos
pip3 install -r src/nautronics_auv/requirements.txt

sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y

```



## ArduPilot Gazebo Export Ayarlarının `.bashrc` Dosyasına Eklenmesi

Aşağıdaki komutları terminale yapıştırarak tüm gerekli modelleri ve dünyaları `~/.bashrc` dosyasına ekleyebilirsiniz:

```bash
echo 'export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$HOME/nautronics_auv/src/auv_description/models:$HOME/nautronics_auv/src/auv_description/worlds:${GZ_SIM_RESOURCE_PATH}' >> ~/.bashrc
```
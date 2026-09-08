import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'auv_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sye',
    maintainer_email='yusuf.eskin@metu.edu.tr',
    description='Layer 1: sensor gates for DVL, IMU, depth and heading.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dvl_gate = auv_sensors.dvl_gate:main',
            'imu_gate = auv_sensors.imu_gate:main',
            'depth_gate = auv_sensors.depth_gate:main',
            'heading_gate = auv_sensors.heading_gate:main',
            'sim_pressure_source = auv_sensors.sim_pressure_source:main',
        ],
    },
)

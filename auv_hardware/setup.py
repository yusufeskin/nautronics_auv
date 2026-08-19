from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'auv_hardware'

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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ping_sonar_node = auv_hardware.ping_sonar:main',
            'battery_node = auv_hardware.battery_node:main',
            # 'pixhawk_bridge = auv_hardware.pixhawk_bridge:main',
            'pixhawk_bridge2 = auv_hardware.pixhawk_bridge2:main',
            'sim_dvl_adapter = auv_hardware.sim_dvl_adapter:main',

        ],
    },
)

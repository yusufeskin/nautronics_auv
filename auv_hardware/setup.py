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
            'pwm_router_node = auv_hardware.pwm_router:main',
            'pixhawk_baro_reader = auv_hardware.baro_publisher:main',
            'ping_sonar_node = auv_hardware.ping_sonar:main',
            'battery_node = pixhawk_battery.pixhawk_battery_node:main',
        ],
    },
)

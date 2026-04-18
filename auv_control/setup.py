from glob import glob
from setuptools import find_packages, setup
import os
import time

package_name = 'auv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
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
            'point_follower = auv_control.point_follower:main',
            'thruster_mixer = auv_control.thruster_mixer:main',
            'visual_servoing_action = auv_control.visual_servoing_action:main',
            'yawer=auv_control.yawer:main',
            'blind_push_action = auv_control.blind_push_action:main',
            'solution = auv_control.solution:main',
            'metrekayma = auv_control.metrekayma:main',
            'csvcreater = auv_control.csvcreater:main',
            'pwm_publisher = auv_control.pwm_publisher:main',
            'expandingsearch = auv_control.expandingsearch:main'
        ],
    },
)

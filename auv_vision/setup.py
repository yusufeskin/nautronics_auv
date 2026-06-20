from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'auv_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model'), glob('model/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='berkay',
    maintainer_email='eldemirberkay01@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'image_collector = auv_vision.image_collector:main',
        'model_debugger = auv_vision.model_debugger:main',
        'pnp_debugger = auv_vision.pnp_debugger:main',
        'object_keypoint_detector = auv_vision.object_keypoint_detector:main',
        'torpedo_pnp_solver = auv_vision.torpedo_pnp_solver:main',
    ],
	},
)

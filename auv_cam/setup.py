from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'auv_cam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'model'), glob('model/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='baris',
    maintainer_email='mehmetbgul58@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_collector = auv_cam.image_collector:main',
        ],
    },
)

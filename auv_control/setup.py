from setuptools import find_packages, setup

package_name = 'auv_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'updated_visual_servoing = auv_control.updated_visual_servoing:main',
            'visual_servoing_action = auv_control.visual_servoing_action'
        ],
    },
)

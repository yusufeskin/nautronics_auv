from setuptools import find_packages, setup

package_name = 'auv_bt'

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
            'gate_servo = gate.gate_servo:main',
            'visual_servo = pool.visual_servo:main',
            'teknofest = pool.teknofest:main',
            'manager =  lifecycle_manager.manager:main',
            'gate_gyro = gate.gate_gyro:main',
            'main_behaviour_tree1 = pool.main_behaviour_tree1:main',
            'main_behaviour_tree2 = pool.main_behaviour_tree2:main',
            'slalom_bt = slalom.slalom_bt:main',
            'mixed_mission = pool.mixed_mission:main',
            'main_behaviour_tree = pool.main_behaviour_tree:main',
            'dvl_transit_test = pool.dvl_transit_test:main',
            'multi_point_transit = pool.multi_point_transit:main',
            'relative_zigzag_test = pool.relative_zigzag_test:main',
            'teknofest_dvl_mission = teknofest.teknofest_dvl_mission:main',

        ],
    },
)

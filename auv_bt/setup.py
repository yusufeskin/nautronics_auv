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
            'gate_behaviour_tree = gate.gate_behaviour_tree:main',
            'yaw_turn = pool.yaw_turn:main', 
            'pool2_bt = pool.pool2_bt:main',
            'visual_servo = pool.visual_servo:main',
            'yawer_turn = pool.yawer_turn:main',
            'teknofest = pool.teknofest:main',
            'deneme = pool.deneme:main',
            'main_behaviour_tree = pool.main_behaviour_tree:main',
            'manager =  lifecycle_manager.manager:main'
        ],
    },
)

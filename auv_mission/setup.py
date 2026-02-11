from setuptools import setup

package_name = 'auv_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=['gate', 'gate.behaviours', 'common_behaviours'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='baris',
    maintainer_email='baris@todo.todo',
    description='AUV Mission Package',
    license='TODO: License declaration',
    # tests_require=['pytest'],  <-- BU SATIRI SİLDİK!
    entry_points={
        'console_scripts': [
            'behaviour_tree = gate.behaviour_tree:main',
        ],
    },
)

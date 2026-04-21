from setuptools import find_packages, setup

package_name = 'auv_virtual_dvl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[
    'auv_virtual_dvl',
    'auv_virtual_dvl.models',
    'auv_virtual_dvl.scripts'
],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mete',
    maintainer_email='metteakkan@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dvl_calistir = auv_virtual_dvl.realtime_node:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'fire_vision_ros'
python_module = 'fire_vision_pymodule'

setup(
    name=package_name,
    version='0.0.0',

    packages=[
        python_module,
    ],

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='peanut',
    maintainer_email='peanut@example.com',

    description='Fire vision ROS 2 package',
    license='Apache-2.0',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'image_subscriber = fire_vision_pymodule.image_subscriber:main',
        ],
    },
)

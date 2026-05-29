from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ree_exploration_ppo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sojavola',
    maintainer_email='sojavolar.2002@gmail.com',
    description='PPO REE exploration — global 100x100 obs, split ppo_agent/ppo_trainer nodes',
    license='TODO',
    entry_points={
        'console_scripts': [
            'ppo_agent = ree_exploration_ppo.ppo_agent_node:main',
            'ppo_trainer = ree_exploration_ppo.ppo_trainer_node:main',
        ],
    },
)

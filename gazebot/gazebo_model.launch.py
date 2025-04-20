import os
import xacro
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Define package and model names
    robot_name = 'my_robot'
    package_name = 'my_robot_package'

    # Paths to the Xacro file and the Gazebo world file
    xacro_file = os.path.join(get_package_share_directory(package_name), 'urdf', 'robot.xacro')
    world_file = os.path.join(get_package_share_directory(package_name), 'worlds', 'empty_world.world')

    # Process the Xacro file into URDF XML
    robot_description = xacro.process_file(xacro_file).toxml()

    # Launch Gazebo with the specified world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'world': world_file}.items()
    )

    # Start the robot_state_publisher to publish the robot description (URDF)
    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # Spawn the robot into Gazebo after a delay (to allow Gazebo to start up)
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', robot_name],
        output='screen'
    )

    # Create the launch description and add actions
    ld = LaunchDescription()
    ld.add_action(gazebo_launch)
    ld.add_action(state_publisher)
    ld.add_action(TimerAction(period=5.0, actions=[spawn_robot]))

    return ld


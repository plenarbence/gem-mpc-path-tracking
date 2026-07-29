cmake_minimum_required(VERSION 3.0.2)
project(gem_control)

find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  geometry_msgs
  nav_msgs
  rospy
  std_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS ackermann_msgs geometry_msgs nav_msgs rospy std_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/analyze_full_mpc_run.py
  scripts/analyze_simplified_mpc_run.py
  scripts/analyze_cascaded_p_run.py
  scripts/benchmark_full_mpc.py
  scripts/run_cascaded_p.py
  scripts/run_full_mpc.py
  scripts/run_simplified_mpc.py
  scripts/validate_reference_path.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  models
  profiles
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_full_mpc.py)
  catkin_add_nosetests(test/test_cascaded_p.py)
  catkin_add_nosetests(test/test_learned_dynamics.py)
  catkin_add_nosetests(test/test_reference_path.py)
  catkin_add_nosetests(test/test_simplified_mpc.py)
  catkin_add_nosetests(test/test_timing_compensation.py)
  catkin_add_nosetests(test/test_tracking_errors.py)
endif()

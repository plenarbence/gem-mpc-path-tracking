cmake_minimum_required(VERSION 3.0.2)
project(gem_sysid)

add_compile_options(-std=c++14 -Wall -Wextra -Wpedantic)

find_package(catkin REQUIRED COMPONENTS
  ackermann_msgs
  geometry_msgs
  message_filters
  nav_msgs
  roscpp
  rosgraph_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    ackermann_msgs
    geometry_msgs
    message_filters
    nav_msgs
    roscpp
    rosgraph_msgs
    rospy
    sensor_msgs
    std_msgs
    tf2
    tf2_geometry_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(excitation_node
  src/excitation_node.cpp
)

add_dependencies(excitation_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(excitation_node
  ${catkin_LIBRARIES}
)

install(TARGETS excitation_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS
  scripts/analyze_step0.py
  scripts/compare_integration_methods.py
  scripts/generate_identification_profiles.py
  scripts/generate_sysid_urdf.py
  scripts/generate_step0_profile.py
  scripts/prepare_dataset.py
  scripts/run_identification_suite.py
  scripts/run_neural_identification.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_dataset_preparation.py)
  catkin_add_nosetests(test/test_integration_comparison.py)
  catkin_add_nosetests(test/test_neural_data_model.py)
  catkin_add_nosetests(test/test_neural_training.py)
endif()

install(DIRECTORY
  launch
  profiles
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 3.10)
project(flexbe_kinovagen3_flexbe_behaviors)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  manifest
  DESTINATION local/lib/python3.10/dist-packages/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION local/lib/python3.10/dist-packages/${PROJECT_NAME}
)

install(PROGRAMS
    bin/copy_behavior
    DESTINATION local/lib/python3.10/dist-packages/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

ament_package()

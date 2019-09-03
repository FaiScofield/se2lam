cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# 设置编译选项参数
IF (CMAKE_COMPILER_IS_GNUCXX )
    SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

    ADD_DEFINITIONS("-Wall  -Wno-unused-variable
                    -Wno-unused-but-set-variable -Wno-unknown-pragmas")  #-Werror
ENDIF()

add_subdirectory(Thirdparty/DBoW2)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

# c++ 11 support for g2o, multi-thread and shared_ptr
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-old-style-cast -Wno-deprecated-declarations")

find_package(Eigen REQUIRED)
set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
find_package(OpenCV 3 REQUIRED)
find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
find_package(G2O REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
    rospy
    tf
    roslib
    cv_bridge
    cmake_modules
    image_transport
    message_generation
    rosbag
)

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
list(APPEND LINK_LIBS ${catkin_LIBRARIES})

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/se2lam
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${Cholmod_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

list(APPEND LINK_LIBS
  ${OpenCV_LIBS}
)

file(GLOB G2O_LIBS /home/vance/opt/g2o-20160424/install/lib/*.so)
list(APPEND G2O_LIBS
#    g2o_cli g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core
#    g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba
#    g2o_types_slam3d g2o_csparse_extension g2o_solver_dense g2o_stuff
#    g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3
    cxsparse cholmod
)

## Build a lib
FILE(GLOB_RECURSE IncFiles "include/se2lam/*.h")
aux_source_directory(src/. DIR_SRCS)
add_library(se2lam
    ${DIR_SRCS}
    ${IncFiles}
    include/se2lam/Preprocess.h
    include/se2lam/lineDetection.h
    src/lineDetection.cpp
    src/Preprocess.cpp
)
target_link_libraries(se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so
)

#add_library(lineDetector
#    include/se2lam/Preprocess.h
#    include/se2lam/lineDetection.h
#    src/lineDetection.cpp
#    src/Preprocess.cpp

##    line_descriptor/
#)


add_subdirectory(rk)
add_subdirectory(test)



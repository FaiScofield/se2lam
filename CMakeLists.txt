cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

# Add header file include directories
include_directories(./Thirdparty ./Thirdparty/lsd_161 ./Thirdparty/line_descriptor/include ./Thirdparty/line_descriptor/include/line_descriptor ./Thirdparty/line_descriptor/src)

# Add block directories
ADD_SUBDIRECTORY(Thirdparty/lsd_161)
ADD_SUBDIRECTORY(Thirdparty/line_descriptor)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# 设置编译选项参数
IF (CMAKE_COMPILER_IS_GNUCXX )
#    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fsanitize-recover=address") # -fno-omit-frame-pointer
#    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize-recover=address") # -O1 -g
#    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address -fsanitize-recover=address")

    ADD_DEFINITIONS("-W -Wall -Wno-deprecated-declarations")  #-Werror  -Wno-unused-parameter
ENDIF()

add_subdirectory(Thirdparty/DBoW2)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

# c++ 11 support for g2o, multi-thread and shared_ptr
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -O0") #-Werror -g -O
find_package(Eigen REQUIRED)
#set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
#find_package(OpenCV 3.3 REQUIRED)
find_package(OpenCV 3.3.1 REQUIRED core features2d calib3d highgui imgcodecs imgproc video)
MESSAGE("    ===> OpenCV Information: version: ${OpenCV_VERSION}" )
MESSAGE("    ===> OpenCV Information: inlcude_dir: ${OpenCV_INCLUDE_DIRS}" )
MESSAGE("    ===> OpenCV Information: libs_dir: ${OpenCV_LIBS}" )

find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
#set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
find_package(G2O REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
    rospy
    tf
    roslib
    cv_bridge
    image_transport
    rosbag
)
catkin_package(
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
        geometry_msgs
        nav_msgs
        tf
        roscpp
        image_transport
    )
include_directories(${catkin_INCLUDE_DIRS})

list(APPEND LINK_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    )
MESSAGE("    ===>:   ${catkin_LIBRARIES}" )

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

#file(GLOB G2O_LIBS /home/vance/opt/g2o-20160424/install/lib/*.so)
#list(APPEND G2O_LIBS
#    cxsparse cholmod
#)
list(APPEND G2O_LIBS
        cxsparse
        cholmod
        g2o_cli
        g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
        g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
        g2o_types_sba g2o_types_slam3d g2o_csparse_extension
        g2o_solver_dense g2o_stuff
        g2o_types_sclam2d g2o_parser g2o_solver_pcg
        g2o_types_data g2o_types_sim3
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
        src/readImageImu.cpp include/se2lam/readImageImu.h)
target_link_libraries(se2lam
    linedesc
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so
)


add_subdirectory(rk)
add_subdirectory(test)

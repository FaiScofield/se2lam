################################################################################
# 全局变量, 是否指定G2O和OpenCV的路径
################################################################################
set(CUSTOM_G2O FALSE)
set(CUSTOM_OpenCV FALSE)

################################################################################
# 基本信息
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

################################################################################
# 设置编译选项参数
################################################################################
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("===> CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(CMAKE_COMPILER_IS_GNUCXX )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    add_definitions("-W -Wall -Wno-deprecated-declarations") #-Werror -Wno-unused-parameter
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################################################################################
# 设置相关依赖包
################################################################################
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

## ROS packages setting
find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
    rospy
    tf
    roslib
    cv_bridge
    image_transport
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

## Standard packages
#find_package(CSparse REQUIRED)
#find_package(Cholmod REQUIRED)
find_package(Eigen REQUIRED)

## Boost
find_package(Boost COMPONENTS filesystem REQUIRED)
if(Boost_FOUND)
    message("===> Boost Information: Version: ${Boost_VERSION}" )
    message("===> Boost Information: Include Dir: ${Boost_INCLUDE_DIRS}" )
    message("===> Boost Information: Libs Dir: ${Boost_LIBRARIES}" )
else()
    message("===> Boost Not Found!!!" )
endif()

## OpenCV
if(CUSTOM_OpenCV)
    set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
    find_package(OpenCV 3.4.1 REQUIRED)
else()
    find_package(OpenCV 3 REQUIRED)
endif()
if (OpenCV_FOUND)
    message("===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

## G2O
if(CUSTOM_G2O)
    set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
    find_package(G2O REQUIRED)
    file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
    list(APPEND G2O_LIBS cholmod cxsparse)
else()
    find_package(G2O REQUIRED)
    list(APPEND G2O_LIBS
        cholmod cxsparse
        g2o_cli g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core
        g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba
        g2o_types_slam3d g2o_csparse_extension g2o_solver_dense g2o_stuff
        g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3
    )
endif()
if (G2O_FOUND)
    message("===> G2O Information: Root: ${G2O_ROOT}")
    message("===> G2O Information: Include Dir: ${G2O_INCLUDE_DIR}" )
    message("===> G2O Information: Libs Dir: ${G2O_LIBS}" )
else()
    message("===> G2O Not Found!!!" )
endif()

################################################################################
# 链接库
################################################################################
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/se2lam
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
#    ${CSPARSE_INCLUDE_DIR}
#    ${Cholmod_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    /usr/include/suitesparse # for cholmod
)

list(APPEND LINK_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

################################################################################
# Build a lib
################################################################################
file(GLOB_RECURSE IncFiles "include/se2lam/*.h")
aux_source_directory(src/. DIR_SRCS)
add_library(se2lam
    ${DIR_SRCS}
    ${IncFiles}
)
target_link_libraries(se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so
)

################################################################################
# 子目录要放在最后
################################################################################
add_subdirectory(Thirdparty/DBoW2)
add_subdirectory(rk)
add_subdirectory(test)

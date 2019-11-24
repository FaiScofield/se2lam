################################################################################
# 基本信息
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

add_subdirectory(Thirdparty/DBoW2)

# 设置编译选项参数
################################################################################
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("===> CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# -march=native 参数不可用, 会使g2o出错.
if(CMAKE_COMPILER_IS_GNUCXX )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -ggdb") # -g
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")

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
    cmake_modules
    image_transport
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES se2lam
#   CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy tf
#  DEPENDS Eigen OpenCV Boost

#   CATKIN_DEPENDS message_runtime
)
include_directories(${catkin_INCLUDE_DIRS})

## Eigen
find_package(Eigen REQUIRED)

## Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
if(Boost_FOUND)
    message("===> Boost Information: Version: ${Boost_VERSION}" )
    message("===> Boost Information: Include Dir: ${Boost_INCLUDE_DIRS}" )
    message("===> Boost Information: Libs Dir: ${Boost_LIBRARIES}" )
else()
    message("===> Boost Not Found!!!" )
endif()

## OpenCV
set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

## G2O
find_package(CSparse REQUIRED)
if(CSPARSE_FOUND)
    message("===> CSparse Information: Include Dir: ${CSPARSE_INCLUDE_DIR}" )
    message("===> CSparse Information: Libs Dir: ${CSPARSE_LIBRARY}" )
else()
    message("===> CSparse Not Found!!!")
endif()
find_package(Cholmod REQUIRED)
if(CHOLMOD_FOUND)
    message("===> Cholmod Information: Include Dir: ${CHOLMOD_INCLUDE_DIR}" )
    message("===> Cholmod Information: Libs Dir: ${CHOLMOD_LIBRARIES}" )
else()
    message("===> Cholmod Not Found!!!")
endif()
set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
find_package(G2O REQUIRED)
file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
# list(APPEND G2O_LIBS cxsparse cholmod)
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
list(APPEND LINK_LIBS )

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/se2lam
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
)

list(APPEND LINK_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

################################################################################
# Build a lib
################################################################################
FILE(GLOB_RECURSE IncFiles "include/se2lam/*.h")
aux_source_directory(src/. DIR_SRCS)
add_library(se2lam
    ${DIR_SRCS}
    ${IncFiles}
)

target_link_libraries(se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    DBoW2se2lam
)

## Build an exe
add_executable(test_vn test/test_vn.cpp)
target_link_libraries(test_vn
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

add_executable(datapub test/datapub.cpp)
target_link_libraries(datapub
    se2lam
    ${LINK_LIBS}
)

add_executable(imgview test/imgview.cpp)
target_link_libraries(imgview
    se2lam
    ${LINK_LIBS}
)


# Build a subscriber test for vn dataset
add_executable(test_ros test/test_ros.cpp)
target_link_libraries(test_ros
    se2lam
    ${LINK_LIBS}
)

add_executable(run_pipeline test/run_pipeline.cpp)
target_link_libraries(run_pipeline
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

add_executable(rkOdomToSE2 test/rkOdomToSE2.cpp)
target_link_libraries(rkOdomToSE2
    se2lam
    ${LINK_LIBS}
)

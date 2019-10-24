cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# 设置编译选项参数
IF (CMAKE_COMPILER_IS_GNUCXX )
    ADD_DEFINITIONS("-W -Wall -Wno-deprecated-declarations")  #-Werror  -Wno-unused-parameter
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -g -std=c++11") #-Werror -g -O

add_subdirectory(Thirdparty/DBoW2)
add_subdirectory(Thirdparty/g2o-20160424)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)


find_package(Eigen REQUIRED)
find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

find_package(OpenCV 3 REQUIRED)
if (OpenCV_FOUND)
    MESSAGE("    ===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    MESSAGE("    ===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    MESSAGE("    ===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    MESSAGE("    ===> OpenCV Not Found!!!" )
endif()

set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
#set(G2O_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/g2o-20160424/install)
find_package(G2O REQUIRED)
set(G2O_INCLUDE_DIR ${G2O_ROOT}/include)
file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
list(APPEND G2O_LIBS
    cxsparse cholmod
)
if (G2O_FOUND)
    MESSAGE("    ===> G2O Information: Include Dir: ${G2O_INCLUDE_DIRS}" )
    MESSAGE("    ===> G2O Information: Libs Dir: ${G2O_LIBS}" )
else()
    MESSAGE("    ===> G2O Not Found!!!" )
endif()

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

list(APPEND LINK_LIBS
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

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

## Build a lib
FILE(GLOB_RECURSE IncFiles "include/se2lam/*.h")
aux_source_directory(src/. DIR_SRCS)
add_library(se2lam
    ${DIR_SRCS}
    ${IncFiles}
)
target_link_libraries(se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so
)

add_subdirectory(rk)

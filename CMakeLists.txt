################################################################################
# 全局变量, 是否指定G2O和OpenCV的路径
################################################################################
set(CROSS_COMPILE FALSE)
set(CUSTOM_G2O FALSE)
set(CUSTOM_OpenCV FALSE)
message("===> CROSS_COMPILE: ${CROSS_COMPILE}")
message("===> CUSTOM_G2O: ${CUSTOM_G2O}")
message("===> CUSTOM_OpenCV: ${CUSTOM_OpenCV}")

################################################################################
# 基本信息
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

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

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/Thirdparty
)

## ROS packages setting
find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    roscpp
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
find_package(Eigen REQUIRED)
message("===> Eigen Include DIR: ${EIGEN_INCLUDE_DIRS}")

## Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
if(Boost_FOUND)
    message("===> Boost Version: ${Boost_VERSION}" )
    message("===> Boost Include Dir: ${Boost_INCLUDE_DIRS}" )
    message("===> Boost Libs Dir: ${Boost_LIBRARIES}" )
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
    message("===> OpenCV Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

## G2O
if(CROSS_COMPILE)
    set(CSPARSE_INCLUDE_DIR     ${PROJECT_SOURCE_DIR}/dependencies/csparse/include)
    set(CSPARSE_LIBRARY         ${PROJECT_SOURCE_DIR}/dependencies/csparse/lib/libcxsparse.so)
    set(CHOLMOD_INCLUDE_DIR     ${PROJECT_SOURCE_DIR}/dependencies/cholmod/include)
    file(GLOB CHOLMOD_LIBRARIES ${PROJECT_SOURCE_DIR}/dependencies/cholmod/lib/*.so)
    set(G2O_INCLUDE_DIR         ${PROJECT_SOURCE_DIR}/dependencies/g2o/include)
    file(GLOB G2O_LIBS          ${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/*.so)
else(CROSS_COMPILE)
    find_package(CSparse REQUIRED)
    if(CSPARSE_FOUND)
        message("===> CSparse Include Dir: ${CSPARSE_INCLUDE_DIR}" )
        message("===> CSparse Libs Dir: ${CSPARSE_LIBRARY}" )
    else()
        message("===> CSparse Not Found!!!")
    endif()

    find_package(Cholmod REQUIRED)
    if(CHOLMOD_FOUND)
        message("===> Cholmod Include Dir: ${CHOLMOD_INCLUDE_DIR}" )
        message("===> Cholmod Libs Dir: ${CHOLMOD_LIBRARIES}" )
    else()
        message("===> Cholmod Not Found!!!")
    endif()

    if(CUSTOM_G2O)
        set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
        find_package(G2O REQUIRED)
        file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
        if (G2O_FOUND)
            message("===> G2O Root: ${G2O_ROOT}")
            message("===> G2O Include Dir: ${G2O_INCLUDE_DIR}" )
            message("===> G2O Libs Dir: ${G2O_LIBS}" )
        else()
            message("===> G2O Not Found!!!" )
        endif()
    else()
        file(GLOB G2O_LIBS ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/*.so)
    endif()
endif(CROSS_COMPILE)

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
file(GLOB_RECURSE IncFiles "include/se2lam/*.h")
aux_source_directory(src/. DIR_SRCS)
add_library(se2lam SHARED
    ${DIR_SRCS}
    ${IncFiles}
)

target_link_libraries(se2lam
    ${LINK_LIBS}
    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    ${G2O_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/lib/liblinedesc.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/lsd_161/lib/liblsd_161.so
)
if(CROSS_COMPILE)
    target_link_libraries(se2lam -ltinyxml2)  # 交叉编译中ROS需要
endif()

# 子目录
add_subdirectory(rk)
add_subdirectory(test)

################################################################################
# Install setting for ROS
################################################################################
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



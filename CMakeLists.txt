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
#    ${PROJECT_SOURCE_DIR}/Thirdparty
#    ${PROJECT_SOURCE_DIR}/Thirdparty/lsd_161
#    ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include
#    ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include/line_descriptor
#    ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/src
    ${PROJECT_SOURCE_DIR}/dependencies
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
if(CROSS_COMPILE)
    set(LineDesc_INCLUDE_DIR    ${PROJECT_SOURCE_DIR}/dependencies/line_descriptor/include)
    set(LineDesc_LIBS           ${PROJECT_SOURCE_DIR}/dependencies/line_descriptor/lib/liblinedesc.so)
    set(LSD161_INCLUDE_DIR      ${PROJECT_SOURCE_DIR}/dependencies/lsd_161/include)
    set(LSD161_LIBS             ${PROJECT_SOURCE_DIR}/dependencies/lsd_161/lib/liblsd_161.so)
    set(DBOW2_INCLUDE_DIR       ${PROJECT_SOURCE_DIR}/dependencies/dbow2/include)
    set(DBOW2_LIBS              ${PROJECT_SOURCE_DIR}/dependencies/dbow2/lib/libDBoW2se2lam.so)
    set(CSPARSE_INCLUDE_DIR     ${PROJECT_SOURCE_DIR}/dependencies/csparse/include)
    set(CSPARSE_LIBRARY         ${PROJECT_SOURCE_DIR}/dependencies/csparse/lib/libcxsparse.so)
    set(CHOLMOD_INCLUDE_DIR     ${PROJECT_SOURCE_DIR}/dependencies/cholmod/include)
    file(GLOB CHOLMOD_LIBRARIES ${PROJECT_SOURCE_DIR}/dependencies/cholmod/lib/*.so)
    set(G2O_INCLUDE_DIR         ${PROJECT_SOURCE_DIR}/dependencies/g2o/include)
    file(GLOB G2O_LIBS          ${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/*.so)
else(CROSS_COMPILE)
#    find_package(CSparse REQUIRED)
#    if(CSPARSE_FOUND)
#        message("===> CSparse Information: Include Dir: ${CSPARSE_INCLUDE_DIR}" )
#        message("===> CSparse Information: Libs Dir: ${CSPARSE_LIBRARY}" )
#    else()
#        message("===> CSparse Not Found!!!")
#    endif()

    find_package(Cholmod REQUIRED)
    if(CHOLMOD_FOUND)
        message("===> Cholmod Information: Include Dir: ${CHOLMOD_INCLUDE_DIR}" )
        message("===> Cholmod Information: Libs Dir: ${CHOLMOD_LIBRARIES}" )
    else()
        message("===> Cholmod Not Found!!!")
    endif()

    set(LineDesc_INCLUDE_DIR    ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/include)
    set(LineDesc_LIBS           ${PROJECT_SOURCE_DIR}/Thirdparty/line_descriptor/lib/liblinedesc.so)
    set(LSD161_INCLUDE_DIR      ${PROJECT_SOURCE_DIR}/Thirdparty/lsd_161)
    set(LSD161_LIBS             ${PROJECT_SOURCE_DIR}/Thirdparty/lsd_161/lib/liblsd_161.so)
    set(DBOW2_INCLUDE_DIR       ${PROJECT_SOURCE_DIR}/Thirdparty/dbow2/include)
    set(DBOW2_LIBS              ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2se2lam.so)

    if(CUSTOM_G2O)
        set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
        find_package(G2O REQUIRED)
        file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
    else()
        find_package(G2O REQUIRED)
        list(APPEND G2O_LIBS
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
#    ${CSPARSE_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    ${DBOW2_INCLUDE_DIR}
    ${LSD161_INCLUDE_DIR}
    ${LineDesc_INCLUDE_DIR}
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
add_library(se2lam
    ${DIR_SRCS}
    ${IncFiles}
)

target_link_libraries(se2lam
    ${LINK_LIBS}
    ${DBOW2_LIBS}
    ${LSD161_LIBS}
    ${LineDesc_LIBS}
#    ${CSPARSE_LIBRARY}
    ${CHOLMOD_LIBRARIES}
    ${G2O_LIBS}
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



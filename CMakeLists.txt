cmake_minimum_required(VERSION 2.8.3)
project(se2lam)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE(STATUS "    ===> Build type: " ${CMAKE_BUILD_TYPE})

# 设置编译选项参数
IF (CMAKE_COMPILER_IS_GNUCXX )
    ADD_DEFINITIONS("-W -Wall -Wno-deprecated-declarations")  #-Werror  -Wno-unused-parameter
ENDIF()

# -march=native 参数不可用, 会使g2o出错.
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -std=c++11 -Wall -DLINUX -DCONFIG_C_PLUS_11")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -DNDEBUG")

add_subdirectory(Thirdparty/DBoW2)
#add_subdirectory(Thirdparty/g2o-20160424)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

find_package(Eigen REQUIRED)
#find_package(CSparse REQUIRED)
#find_package(Cholmod REQUIRED)
find_package(SuiteSparse REQUIRED)
if(SUITESPARSE_FOUND)
    message(STATUS "    ===> The SuiteSparse lib is found!")
   include_directories(${SUITESPARSE_INCLUDE_DIRS})
else()
   message(STATUS "    ===> ERROR SuiteSparse not found!")
endif()


find_package(Boost COMPONENTS filesystem REQUIRED)
if(Boost_FOUND)
    message(STATUS "The Boost lib is found!")
    message(STATUS "Boost library status:")
    message(STATUS "    version: ${Boost_VERSION}")
    message(STATUS "    libraries: ${Boost_LIBRARIES}")
    message(STATUS "    include path: ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
    MESSAGE(STATUS "    Boost Not Found!!!" )
endif(Boost_FOUND)

find_package(OpenCV 3 REQUIRED)
if (OpenCV_FOUND)
    MESSAGE("    ===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    MESSAGE("    ===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    MESSAGE("    ===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    MESSAGE("    ===> OpenCV Not Found!!!" )
endif()

#set(G2O_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/g2o-20160424/install)
find_package(G2O REQUIRED)
#set(G2O_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/g2o-20160424)
#file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
list(APPEND G2O_LIBS
    cxsparse cholmod
    g2o_cli g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d
    g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only
    g2o_types_sba g2o_types_slam3d g2o_csparse_extension
    g2o_solver_dense g2o_stuff
    g2o_types_sclam2d g2o_parser g2o_solver_pcg
    g2o_types_data g2o_types_sim3
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

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES launch/run_pipeline.launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})



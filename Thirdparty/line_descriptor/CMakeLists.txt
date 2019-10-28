project( line-descriptor )

cmake_minimum_required(VERSION 2.7)
set(OpenCV_DIR)

find_package(OpenCV 3.3.1 REQUIRED core features2d calib3d highgui imgcodecs imgproc)
if(OpenCV_FOUND)
    message(STATUS "The OpenCV lib is found!")
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
endif(OpenCV_FOUND)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
link_directories(${OpenCV_LIBS_DIR})

#SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -mtune=native -march=native")

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
#set(LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/lib)


include_directories( include ${OpenCV_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}2 ${OpenCV_INCLUDE_DIRS}2\\core)
list(APPEND LINK_LIBS ${OpenCV_LIBS} )
file(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

# Find Eigen3
FIND_PACKAGE(Eigen3)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
#link_directories(${CMAKE_SOURCE_DIR}/src/)
#file(GLOB_RECURSE all_source_files RELATIVE "${CMAKE_SOURCE_DIR}src/" *.cpp )
AUX_SOURCE_DIRECTORY(./src  all_source_files)
#add_custom_target( linedesc_includes DEPENDS ${all_include_files} SOURCES ${all_source_files} )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library( linedesc SHARED ${all_source_files} )
target_link_libraries( linedesc ${LINK_LIBS} )


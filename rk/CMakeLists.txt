

add_executable(test_rk test_rk.cpp)
target_link_libraries(test_rk
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

add_executable(rkOdomToSE2 rkOdomToSE2.cpp)
target_link_libraries(rkOdomToSE2
    se2lam
    ${LINK_LIBS}
)

add_executable(rkImageUndistort rkImageUndistort.cpp)
target_link_libraries(rkImageUndistort
    se2lam
    ${LINK_LIBS}
    ${Boost_LIBRARIES}
)

find_package(PythonLibs 2.7)
find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
add_executable(ICPtrajectory ICPtrajectory.cpp)
target_link_libraries(ICPtrajectory
    se2lam
    ${LINK_LIBS}
    ${Pangolin_LIBRARIES}
    ${PYTHON_LIBRARIES}
)


add_executable(lineDetection lineDetection.cpp)
target_link_libraries(lineDetection
    se2lam
    ${LINK_LIBS}
)

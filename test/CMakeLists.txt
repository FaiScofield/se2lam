## Build an exe
add_executable(test_vn test_vn.cpp)
target_link_libraries(test_vn
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)


add_executable(test_dataPub test_dataPub.cpp)
target_link_libraries(test_dataPub
    se2lam
    ${LINK_LIBS}
)


# Build a subscriber test for vn dataset
add_executable(test_ros test_ros.cpp)
target_link_libraries(test_ros
    se2lam
    ${LINK_LIBS}
)


add_executable(test_imageUndistort test_imageUndistort.cpp)
target_link_libraries(test_imageUndistort
    se2lam
    ${LINK_LIBS}
    ${Boost_LIBRARIES}
)

add_executable(test_pointDetection test_pointDetection.cpp)
target_link_libraries(test_pointDetection
    se2lam
    ${LINK_LIBS}
)

add_executable(test_lineDetection test_lineDetection.cpp)
target_link_libraries(test_lineDetection
    se2lam
    ${LINK_LIBS}
)

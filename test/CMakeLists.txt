## Build an exe
add_executable(test_vn test_vn.cpp)
target_link_libraries(test_vn
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

#add_executable(test_dataPub test_dataPub.cpp)
#target_link_libraries(test_dataPub
#    se2lam
#    ${LINK_LIBS}
#)


#add_executable(test_imageUndistort test_imageUndistort.cpp)
#target_link_libraries(test_imageUndistort
#    se2lam
#    ${LINK_LIBS}
#    ${Boost_LIBRARIES}
#)

add_executable(test_KeyPointMatch
    test_KeyPointMatch.cpp
    test_functions.hpp
)
target_link_libraries(test_KeyPointMatch
    se2lam
    ${LINK_LIBS}
)

#add_executable(test_lineDetection test_lineDetection.cpp)
#target_link_libraries(test_lineDetection
#    se2lam
#    ${LINK_LIBS}
#)

add_executable(test_MapPointManager
    test_MapPointManager.cpp
    test_functions.hpp
    TestTrack.hpp
    TestViewer.hpp
)
target_link_libraries(test_MapPointManager
    se2lam
    ${LINK_LIBS}
)

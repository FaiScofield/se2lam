## Build an exe
add_executable(test_vn test_vn.cpp)
target_link_libraries(test_vn
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

add_executable(test_Affine
   test_Affine.cpp
   test_functions.hpp
)
target_link_libraries(test_Affine
   se2lam
   ${LINK_LIBS}
)

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

add_executable(test_SingleThreadTracking
    test_SingleThreadTracking.cpp
    test_functions.hpp
    TestTrack.h
    TestTrack.cpp
)
target_link_libraries(test_SingleThreadTracking
    se2lam
    ${LINK_LIBS}
)


#add_executable(test_KltTracking
#    test_KltTracking.cpp
#    test_functions.hpp
#)
#target_link_libraries(test_KltTracking
#    se2lam
#    ${LINK_LIBS}
#)

add_executable(test_BA_SE2
    test_BA_SE2.cpp
    test_functions.hpp
    test_optimizer.hpp
)
target_link_libraries(test_BA_SE2
    se2lam
    ${G2O_LIBS}
)

#add_executable(test_BA_SE3
#    test_BA_SE3.cpp
#    test_functions.hpp
#    test_optimizer.hpp
#)
#target_link_libraries(test_BA_SE3
#    se2lam
#    ${G2O_LIBS}
#)

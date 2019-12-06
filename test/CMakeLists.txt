## Build an exe
add_executable(test_vn test_vn.cpp)
target_link_libraries(test_vn
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

#add_executable(datapub datapub.cpp)
#target_link_libraries(datapub
#    se2lam
#    ${LINK_LIBS}
#)

#add_executable(imgview imgview.cpp)
#target_link_libraries(imgview
#    se2lam
#    ${LINK_LIBS}
#)


## Build a subscriber test for vn dataset
#add_executable(test_ros test_ros.cpp)
#target_link_libraries(test_ros
#    se2lam
#    ${LINK_LIBS}
#)

add_executable(run_pipeline run_pipeline.cpp)
target_link_libraries(run_pipeline
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

#add_executable(rkOdomToSE2 rkOdomToSE2.cpp)
#target_link_libraries(rkOdomToSE2
#    se2lam
#    ${LINK_LIBS}
#)

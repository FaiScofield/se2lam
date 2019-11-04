
add_executable(rkOdomToSE2 rkOdomToSE2.cpp)
target_link_libraries(rkOdomToSE2
    se2lam
    ${LINK_LIBS}
)

#add_executable(rkDataToRosbag rkDataToRosbag.cpp)
#target_link_libraries(rkDataToRosbag
#    ${catkin_LIBRARIES}
#    ${OpenCV_LIBS}
#)

add_executable(se2lamPipeline se2lamPipeline.cpp)
target_link_libraries(se2lamPipeline
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)

add_executable(se2lamKltTrack se2lamKltTrack.cpp)
target_link_libraries(se2lamKltTrack
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
    ${DBOW2_LIBS}
    ${LSD161_LIBS}
    ${LineDescriptor_LIBS}
)






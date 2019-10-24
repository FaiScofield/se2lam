add_executable(se2lamPipeline se2lamPipeline.cpp)
target_link_libraries(se2lamPipeline
    se2lam
    ${LINK_LIBS}
    ${G2O_LIBS}
)





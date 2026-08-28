#=============================================================================
# Driver executables.
#
# ltlt_debug  - correctness sweep over every variant, size, and step
# ltlt_perf   - timing harness, writes results to CSV
#=============================================================================

add_executable(ltlt_debug ltlt_debug.cpp)
target_link_libraries(ltlt_debug PRIVATE ltlt_core)

add_executable(ltlt_perf ltlt_perf.cpp)
target_link_libraries(ltlt_perf PRIVATE ltlt_core docopt)

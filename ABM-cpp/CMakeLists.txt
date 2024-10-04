cmake_minimum_required(VERSION 3.8.2)
project(hw2_1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
# set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP REQUIRED)

# # Group number
# set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# # Packaging system
# if (NOT ${GROUP_NAME} STREQUAL None)
#     set(CPACK_GENERATOR TGZ)
#     set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw2_1")
#     install(FILES serial.cpp openmp.cpp DESTINATION .)
#     install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
#     include(CPack)
# endif ()

# # We require the GNU compiler for this assignment.
# if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
#     set(Prg-Clang "PrgEnv-cray")
#     message(WARNING
#             "Must use GNU Compiler for submission. Make sure you ran:\n"
#             "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
# endif ()

add_executable(serial main.cpp serial.cpp common.h)
add_executable(serial_multisim main_multisim.cpp serial_multisim.cpp common_multisim.h)
# enable_language(CUDA)
# add_executable(gpu main.cu gpu.cu common_gpu.h)
# target_compile_features(gpu PRIVATE cxx_std_14)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
# set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "80")

# add_executable(openmp main.cpp openmp.cpp common.h)
# target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
# configure_file(job-openmp job-openmp COPYONLY)

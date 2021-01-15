cmake_minimum_required(VERSION 3.6)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_LIST_DIR}/cmake")

project(test_xgc)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED)
find_package(MGARD REQUIRED)

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)

add_executable(test_xgc test_xgc.cpp)
add_executable(test_xgc_by_surface test_xgc_by_surface.cpp)

target_link_libraries(test_xgc adios2::adios2 mgard::mgard MPI::MPI_C)
target_link_libraries(test_xgc_by_surface adios2::adios2 mgard::mgard MPI::MPI_C)

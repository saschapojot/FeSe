cmake_minimum_required(VERSION 3.28)
project(kt_square_cpp)

set(CMAKE_CXX_STANDARD 17)
message(STATUS "Using C++ compiler: ${CMAKE_CXX_COMPILER}")
# Find Python
find_package(Python3 REQUIRED COMPONENTS Development)
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



# Find Boost
set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_ROOT "$ENV{HOME}/cppLocal/boostBuild")
find_package(Boost REQUIRED COMPONENTS python numpy filesystem)

add_executable(run_mc
        run_mc.cpp
        mc_subroutine/mc_read_load_compute.cpp
)

target_compile_options(run_mc PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-ftree-vectorize>
)

# Set specific optimization for critical files


# Include directories
target_include_directories(run_mc PRIVATE

        ${Python3_INCLUDE_DIRS}
)


# Link libraries
target_link_libraries(run_mc PRIVATE
        pthread
        m
        Boost::filesystem
        Boost::python
        Boost::numpy
        Python3::Python
)
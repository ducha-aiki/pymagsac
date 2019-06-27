cmake_minimum_required(VERSION 2.8.12)
project(pymagsac)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate python module
add_subdirectory(lib/pybind11)
add_subdirectory(lib/magsac)

find_package(OpenCV 3.4)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
endif (OPENCV_FOUND)

SET(SOURCE_DIR "src/pymagsac")
# Tell cmake that headers are in alse in source_dir
include_directories(lib/magsac/include)
include_directories(lib/magsac/src)
include_directories(${SOURCE_DIR})

# Set header files for the library
file(GLOB_RECURSE HDRS_MAGSAC
    "lib/magsac/include/*.h"
    "lib/magsac/include/*.cpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MAGSAC 
    "lib/magsac/src/*.cpp"
)

SET(SOURCES "${SOURCE_DIR}/magsac_python.cpp")


pybind11_add_module(pymagsac ${SOURCES} "${SOURCE_DIR}/bindings.cpp"  ${HDRS_MAGSAC} ${SRCS_MAGSAC} )
target_link_libraries(pymagsac PRIVATE  ${OpenCV_LIBS})
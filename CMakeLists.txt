cmake_minimum_required(VERSION 2.8.12)
project(pymagsac)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate python module
add_subdirectory(lib/pybind11)

find_package(OpenCV 3.4)
if (OPENCV_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS}")
endif (OPENCV_FOUND)

FIND_PACKAGE(OpenMP)
if (OPENMP_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (OPENMP_FOUND)

SET (CMAKE_BUILD_TYPE "RELEASE")

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-strict-aliasing")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wno-write-strings")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -ansi")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ftree-vectorize -funroll-loops")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

SET(SOURCE_DIR "src/pymagsac")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR}/include)

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MAGSAC 
    "${SOURCE_DIR}/src/*.cpp"
)



pybind11_add_module(pymagsac ${SRCS_MAGSAC} )
target_link_libraries(pymagsac PRIVATE  ${OpenCV_LIBS})

cmake_minimum_required(VERSION 2.8)
project(testCanny)

#find openCV
find_package(OpenCV REQUIRED)
if(OPENCV_FOUND)
    message("OPENCV FOUND")
endif()

#find OpenMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(testCanny ${SOURCE_FILES})

target_link_libraries(testCanny ${OpenCV_LIBS} )
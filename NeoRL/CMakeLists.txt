cmake_minimum_required(VERSION 2.8)

project(NeoRL)

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(SFML 2 COMPONENTS system window graphics audio network REQUIRED)
 
include_directories(${SFML_INCLUDE_DIR})

file(GLOB_RECURSE LINK_SRC
    "source/*.h"
    "source/*.cpp"
)
 
add_executable(NeoRL ${LINK_SRC})

target_link_libraries(NeoRL ${OpenCL_LIBRARIES})
target_link_libraries(NeoRL ${SFML_LIBRARIES})
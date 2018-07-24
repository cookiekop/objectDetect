cmake_minimum_required(VERSION 3.9)
project(objectDetect)

set(CMAKE_CXX_STANDARD 11)

add_executable(objectDetect main.cpp)

find_package( OpenCV REQUIRED )

target_link_libraries(objectDetect ${OpenCV_LIBS})
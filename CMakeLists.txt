cmake_minimum_required(VERSION 3.10)
project (TestLibrary)

set(CMAKE_CXX_STANDARD 11)
 
#For the shared library:
set ( PROJECT_LINK_LIBS libpipert.so )
link_directories(./pipert/build) 
 
include_directories(./pipert/include; ./yolo)

find_package(OpenCV REQUIRED)

add_executable(main_yolo main_yolo.cpp)
target_link_libraries(main_yolo ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(main_yolo PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main_yolo PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")
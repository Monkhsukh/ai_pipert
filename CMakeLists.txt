cmake_minimum_required(VERSION 3.10)
project (TestLibrary)

set(CMAKE_CXX_STANDARD 11)
 
#For the shared library:
set ( PROJECT_LINK_LIBS libpipert.so )
link_directories(./pipert/build) 
 
include_directories(./pipert/include; ./yolo/cfg ./yolo/src ./scheduler/src)

find_package(OpenCV REQUIRED)

add_executable(main_fast main_fast.cpp)
add_executable(main2 main2.cpp yolo/src/yolo.cpp scheduler/src/YoloDetector.cpp scheduler/src/YoloPrinter.cpp)

target_link_libraries(main_fast ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})
target_link_libraries(main2 ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(main_fast PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main_fast PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")

target_compile_options(main2 PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main2 PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")

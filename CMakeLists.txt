cmake_minimum_required(VERSION 3.10)
project (TestLibrary)

set(CMAKE_CXX_STANDARD 11)
 
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#For the shared library:
set ( PROJECT_LINK_LIBS libpipert.so )
link_directories(./pipert/build) 
 
include_directories(./pipert/include; ./yolo/cfg ./yolo/src ./scheduler/src ./utils)

find_package(OpenCV REQUIRED)

#Main
add_executable(main main.cpp yolo/src/yolo.cpp scheduler/src/YoloDetector.cpp yolo/src/MyKalmanFilter.cpp yolo/src/KalmanTracker.cpp scheduler/src/KFTracker.cpp utils/HungarianAlgorithm.cpp)

target_link_libraries(main ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(main PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")

#Main fast
add_executable(main_fast main_fast.cpp)

target_link_libraries(main_fast ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(main_fast PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main_fast PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")

#yolo_run
add_executable(yolo_run yolo_run.cpp yolo/src/yolo.cpp)

target_link_libraries(yolo_run ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(yolo_run PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(yolo_run PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")

#main_server
add_executable(main_server main_server.cpp yolo/src/yolo.cpp scheduler/src/YoloDetector.cpp yolo/src/MyKalmanFilter.cpp yolo/src/KalmanTracker.cpp scheduler/src/KFTracker.cpp utils/HungarianAlgorithm.cpp)

target_link_libraries(main_server ${PROJECT_LINK_LIBS} ${OpenCV_LIBS})

target_compile_options(main_server PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
set_target_properties(main_server PROPERTIES LINK_FLAGS_DEBUG "-fsanitize=address")
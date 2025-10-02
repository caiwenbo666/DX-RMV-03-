cmake_minimum_required(VERSION 3.10.0)
project(work3 VERSION 0.1.0 LANGUAGES C CXX)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED) 
find_package(Eigen3 REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}  
    ${CERES_INCLUDE_DIRS}   
    ${EIGEN3_INCLUDE_DIRS})
add_executable(work3 main.cpp)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}      
    ${CERES_LIBRARIES}  )

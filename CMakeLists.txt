cmake_minimum_required(VERSION 3.20)
set(GLM_PATH "C:/dev_libs/glm-0.9.9.8/glm")
set(GLEW_PATH "C:/dev_libs/glew")
set(FREEGLUT_PATH "C:/dev_libs/freeglut")
set(src/headers "src/headers")
project(Main LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB SRC_FILES "src/*.cpp")
add_executable(Main ${SRC_FILES} src/MainCuda.cu)
# set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
# set_source_files_properties(MainOpenGL.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(Main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(Main PRIVATE "/headers")
target_include_directories(Main PUBLIC
        "$<BUILD_INTERFACE:${GLM_PATH};${GLEW_PATH}/include;${FREEGLUT_PATH}/include>")

target_link_libraries(Main PUBLIC "${GLEW_PATH}/lib/glew32.lib")
target_link_libraries(Main PUBLIC "${FREEGLUT_PATH}/lib/freeglut.lib")

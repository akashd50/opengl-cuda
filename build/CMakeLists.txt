cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GLM_PATH "C:/dev_libs/glm-0.9.9.8/glm")
set(GLEW_PATH "C:/dev_libs/glew")
set(FREEGLUT_PATH "C:/dev_libs/freeglut")
set(LIBDIRS "${GLEW_PATH}/lib;${FREEGLUT_PATH}/lib")

# set the project name
project(MainNew)
#find_library(glew32.lib ${GLEW_PATH}/lib)
#add_library(GLEW STATIC "${GLEW_PATH}/lib")
#set_target_properties(glew32.lib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(MainNew test_extra.cpp)
target_link_libraries(MainNew PUBLIC "${GLEW_PATH}/lib/glew32.lib")
target_link_libraries(MainNew PUBLIC "${FREEGLUT_PATH}/lib/freeglut.lib")
target_include_directories(MainNew PUBLIC
        "$<BUILD_INTERFACE:${GLM_PATH};${GLEW_PATH}/include;${FREEGLUT_PATH}/include>")

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# ==========================
# 交叉编译配置
# ==========================
set(TOO_CHAIN_PATH "/home/yang/workdir/toolchain/tool/arm-gcc12.2.0-linux-uclibceabi/bin/")
set(TOO_CHAIN "arm-gcc12.2.0_uClibcng1.0.48-linux-uclibceabi-")
set(CMAKE_C_COMPILER "${TOO_CHAIN_PATH}${TOO_CHAIN}gcc")
set(CMAKE_CXX_COMPILER "${TOO_CHAIN_PATH}${TOO_CHAIN}g++")

project(dt VERSION 1.0.0)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# ==========================
# 搜索头文件、库文件的查找规则
# ==========================
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# ==========================
# 编译选项
# ==========================
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
set(CMAKE_C_FLAGS "-O3 -ldl -pthread -lstdc++")
set(CMAKE_CXX_FLAGS "-O3 -ldl -pthread -lstdc++")
else()
set(CMAKE_C_FLAGS "-O3 -ldl -lstdc++")
set(CMAKE_CXX_FLAGS "-O3 -ldl -lstdc++")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ==========================
# 头文件路径
# ==========================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/goke/include
)

# ==========================
# 库路径
# ==========================
link_directories(
    ${PROJECT_SOURCE_DIR}/lib
)

# ==========================
# 源码收集
# ==========================
file(GLOB SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# ==========================
# 构建静态库 libxcalgo_dt.a
# ==========================
add_library(xcalgo_dt STATIC ${SRC_FILES})

# ==========================
# 头文件路径：PUBLIC / PRIVATE
# ==========================
target_include_directories(xcalgo_dt
    PUBLIC 
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/goke/include

    # PRIVATE 
    #     ${PROJECT_SOURCE_DIR}/internal
)
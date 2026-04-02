cmake_minimum_required(VERSION 3.16)

project(goke7605 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

option(XC_BUILD_DEMO "Build examples/test_image_demo.c" OFF)
option(XC_BUILD_SAMPLE "Build sample/test_image.c" OFF)

set(XC_SDK_INCLUDE_DIRS "" CACHE PATH "Additional SDK include directories (for xmedia_mmz.h/xmedia_sys.h)")
set(XC_SDK_LIBRARY_DIRS "" CACHE PATH "Additional SDK library directories")
set(XC_SDK_LIBRARIES "xmedia_cl;xmedia_mmz;xmedia_sys" CACHE STRING "SDK libraries required by xc_image_infer")
set(XC_SAMPLE_INCLUDE_DIRS "" CACHE PATH "Additional include directories for sample/test_image.c (for FFmpeg headers, etc.)")
set(XC_SAMPLE_LIBRARIES "avformat;avcodec;avutil;swscale;swresample" CACHE STRING "Extra libraries required by sample/test_image.c")

find_package(Threads REQUIRED)

add_library(xc_image_infer STATIC
    src/xc_dt.cpp
)

target_include_directories(xc_image_infer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${XC_SDK_INCLUDE_DIRS}
)

target_link_directories(xc_image_infer
    PRIVATE
        ${XC_SDK_LIBRARY_DIRS}
)

target_link_libraries(xc_image_infer
    PUBLIC
        Threads::Threads
        m
    PRIVATE
        ${XC_SDK_LIBRARIES}
)

if(XC_BUILD_DEMO)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/xm_image_infer.h")
        message(FATAL_ERROR
            "examples/test_image_demo.c includes ../include/xm_image_infer.h, but that file does not exist in this repo. "
            "Update the example include or provide a compatibility header before enabling XC_BUILD_DEMO.")
    endif()

    add_executable(test_image_demo
        examples/test_image_demo.c
    )

    target_link_libraries(test_image_demo
        PRIVATE
            xc_image_infer
    )
endif()

if(XC_BUILD_SAMPLE)
    add_executable(test_image
        sample/test_image.c
    )

    target_include_directories(test_image
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${XC_SDK_INCLUDE_DIRS}
            ${XC_SAMPLE_INCLUDE_DIRS}
    )

    target_link_directories(test_image
        PRIVATE
            ${XC_SDK_LIBRARY_DIRS}
    )

    target_link_libraries(test_image
        PRIVATE
            xc_image_infer
            ${XC_SAMPLE_LIBRARIES}
    )
endif()

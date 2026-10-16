cmake_minimum_required(VERSION 3.14)
project(rdna3_ai_engine C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Configure output directories for clean artifact discovery
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Shared library target
add_library(rdna3_ai_engine SHARED src/rdna3_ai_engine.c)
target_include_directories(rdna3_ai_engine PUBLIC include)
target_compile_definitions(rdna3_ai_engine PRIVATE RDNA3_EXPORTS)

if (MSVC)
    target_compile_options(rdna3_ai_engine PRIVATE /W4 /wd4100 /wd4204 /wd4152)
else()
    target_compile_options(rdna3_ai_engine PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_link_libraries(rdna3_ai_engine PRIVATE ${CMAKE_DL_LIBS} m)
endif()

# C Example executable
add_executable(use_in_c examples/use_in_c.c)
target_include_directories(use_in_c PRIVATE include)
target_link_libraries(use_in_c PRIVATE rdna3_ai_engine)

if (NOT WIN32)
    set_target_properties(use_in_c PROPERTIES BUILD_RPATH "\$ORIGIN/../lib")
endif()

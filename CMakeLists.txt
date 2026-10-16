cmake_minimum_required(VERSION 3.14)
project(rdna3_ai_engine C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

# Shared library target
add_library(rdna3_ai_engine SHARED src/rdna3_ai_engine.c)
target_compile_definitions(rdna3_ai_engine PRIVATE RDNA3_EXPORTS)

if (WIN32)
    # Windows links against standard runtime
else()
    target_link_libraries(rdna3_ai_engine PRIVATE ${CMAKE_DL_LIBS} m)
endif()

# C Example target
add_executable(use_in_c examples/use_in_c.c)
target_link_libraries(use_in_c PRIVATE rdna3_ai_engine)

if (NOT WIN32)
    set_target_properties(use_in_c PROPERTIES BUILD_RPATH "\$ORIGIN")
endif()

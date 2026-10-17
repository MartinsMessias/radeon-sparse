cmake_minimum_required(VERSION 3.15)
project(radeon_sparse VERSION 2.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 1. Shared Library (radeon_sparse.dll on Windows, libradeon_sparse.so on Linux)
add_library(radeon_sparse SHARED src/radeon_sparse.c)
target_include_directories(radeon_sparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX AND NOT APPLE)
    target_link_libraries(radeon_sparse PRIVATE dl m)
    target_compile_options(radeon_sparse PRIVATE -O3 -fPIC -fvisibility=hidden -Wall -Wextra)
elseif(MSVC)
    target_compile_options(radeon_sparse PRIVATE /O2 /W3)
endif()

# 2. Official Benchmark Suite (radeon-sparse-bench)
add_executable(radeon-sparse-bench benchmarks/radeon_sparse_bench.c)
target_link_libraries(radeon-sparse-bench PRIVATE radeon_sparse)
if(UNIX AND NOT APPLE)
    target_link_libraries(radeon-sparse-bench PRIVATE m)
endif()

# 3. C Quickstart Example (use_in_c)
add_executable(use_in_c examples/use_in_c.c)
target_link_libraries(use_in_c PRIVATE radeon_sparse)
if(UNIX AND NOT APPLE)
    target_link_libraries(use_in_c PRIVATE m)
endif()

# 4. System-wide Global Installation Rules (cmake --install build)
include(GNUInstallDirs)
install(TARGETS radeon_sparse radeon-sparse-bench
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/radeon_sparse.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

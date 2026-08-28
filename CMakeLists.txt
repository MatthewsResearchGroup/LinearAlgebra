cmake_minimum_required(VERSION 3.23.0)
project(ltlt LANGUAGES C CXX)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# CMake's stock Release flags are "-O3 -DNDEBUG". NDEBUG would disable the
# assert()-based bounds checking throughout the MArray headers, which is the
# safety net ltlt_debug relies on. Keep the assertions live. For an
# absolute-fastest timing run, configure with -DCMAKE_CXX_FLAGS=-DNDEBUG.
set(CMAKE_CXX_FLAGS_RELEASE "-O3" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE   "-O3" CACHE STRING "" FORCE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#=============================================================================
# OpenMP
#
# OpenMP is a compiler feature plus a separate runtime library, and BLIS wants
# both reachable through a bare -fopenmp: its makefiles hand the C compiler
# that one flag for any compiler identifying as clang (common.mk, under
# CC_VENDOR=clang), relying on the driver to enable the pragmas and link the
# runtime in one go. Probe for it before anything else, since a compiler
# failing here fails deep inside the BLIS sub-build otherwise.
#=============================================================================
check_c_compiler_flag("-fopenmp" HAVE_C_FOPENMP)
if(NOT HAVE_C_FOPENMP)
    message(FATAL_ERROR
        "${CMAKE_C_COMPILER} rejects -fopenmp, which BLIS requires. Set "
        "CMAKE_C_COMPILER/CMAKE_CXX_COMPILER to a compiler that accepts it "
        "-- see CMakeUserPresets.json.example.")
endif()

find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR
        "OpenMP not found")
endif()

#=============================================================================
# Architecture tuning: -march=native, falling back to -mcpu=native on AArch64
#=============================================================================
check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
if(HAVE_MARCH_NATIVE)
    set(LTLT_ARCH_FLAGS -march=native)
else()
    check_cxx_compiler_flag("-mcpu=native" HAVE_MCPU_NATIVE)
    if(HAVE_MCPU_NATIVE)
        set(LTLT_ARCH_FLAGS -mcpu=native)
    else()
        set(LTLT_ARCH_FLAGS)
        message(WARNING "Neither -march=native nor -mcpu=native accepted; "
                        "building without architecture tuning.")
    endif()
endif()

add_subdirectory(external)

#=============================================================================
# The LTLT algorithm library.
#
# Every .cpp under src/ is library code, so the glob needs no exclusions: the
# two translation units carrying main() live in apps/. CONFIGURE_DEPENDS
# re-globs when the directory changes, so a newly added ltlt_*.cpp variant is
# picked up without editing this file.
#=============================================================================
file(GLOB LTLT_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(ltlt_core OBJECT ${LTLT_SOURCES})

target_include_directories(ltlt_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(ltlt_core PUBLIC ${LTLT_ARCH_FLAGS} -Wno-unused-result)
target_compile_definitions(ltlt_core PUBLIC PROFILE)
target_link_libraries(ltlt_core PUBLIC
    marray
    blis
    OpenMP::OpenMP_CXX
    ${CMAKE_DL_LIBS}
    m
)

# Imported-target dependencies are honoured, but be explicit so BLIS is always
# built before anything that compiles against its headers.
add_dependencies(ltlt_core blis_build)

add_subdirectory(apps)

message(STATUS "LTLT: BLIS config  = ${BLIS_CONFIG}")
message(STATUS "LTLT: arch flags   = ${LTLT_ARCH_FLAGS}")
message(STATUS "LTLT: C++ compiler = ${CMAKE_CXX_COMPILER}")

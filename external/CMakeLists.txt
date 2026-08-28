# Vendored third-party dependencies; nothing is fetched at configure time.

include(ExternalProject)
include(ProcessorCount)

# BLIS has no CMake build of its own: drive its configure+make as an
# ExternalProject and expose the result as an imported static library.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_blis_config_default firestorm)
elseif(APPLE)
    set(_blis_config_default haswell)
else()
    set(_blis_config_default x86_64)
endif()
set(BLIS_CONFIG ${_blis_config_default} CACHE STRING
    "BLIS sub-configuration to build (see external/blis/config/)")

set(BLIS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/blis)

# Built out-of-tree, so deleting a build directory is a real reset. BLIS keys
# its own output paths on the sub-configuration alone, so the compiler is added
# here to keep a shared root (see CMakePresets.json) correct; sharing is safe
# because nothing here varies with CMAKE_BUILD_TYPE. Concurrent builds sharing
# a root will race.
set(BLIS_BUILD_ROOT ${CMAKE_BINARY_DIR}/blis CACHE PATH
    "Parent directory for out-of-tree BLIS builds")

set(BLIS_BUILD_DIR
    ${BLIS_BUILD_ROOT}/${BLIS_CONFIG}-${CMAKE_C_COMPILER_ID}-${CMAKE_C_COMPILER_VERSION})

set(BLIS_INCLUDE_DIR ${BLIS_BUILD_DIR}/include/${BLIS_CONFIG})
set(BLIS_LIBRARY     ${BLIS_BUILD_DIR}/lib/${BLIS_CONFIG}/libblis.a)

# INTERFACE_INCLUDE_DIRECTORIES must exist at generate time; BLIS creates it later.
file(MAKE_DIRECTORY ${BLIS_INCLUDE_DIR})

ProcessorCount(NPROC)
if(NPROC EQUAL 0)
    set(NPROC 1)
endif()

# $(MAKE) passes the jobserver, but only a Makefile generator can expand it.
if(CMAKE_GENERATOR MATCHES "Makefiles")
    set(_blis_build_cmd $(MAKE))
else()
    set(_blis_build_cmd make -j${NPROC})
endif()

# Stamps must sit with the shared build; ExternalProject would otherwise keep
# them per build tree and re-run configure+make in each one.
ExternalProject_Add(blis_build
    SOURCE_DIR        ${BLIS_ROOT}
    BINARY_DIR        ${BLIS_BUILD_DIR}
    PREFIX            ${BLIS_BUILD_DIR}/.ep
    STAMP_DIR         ${BLIS_BUILD_DIR}/.ep/stamp
    TMP_DIR           ${BLIS_BUILD_DIR}/.ep/tmp
    DOWNLOAD_COMMAND  ""
    UPDATE_COMMAND    ""
    CONFIGURE_COMMAND ${BLIS_ROOT}/configure
                      CC=${CMAKE_C_COMPILER}
                      CXX=${CMAKE_CXX_COMPILER}
                      -t openmp
                      --disable-shared
                      ${BLIS_CONFIG}
    BUILD_COMMAND     ${CMAKE_COMMAND} -E env ${_blis_build_cmd}
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  ${BLIS_LIBRARY}
    USES_TERMINAL_CONFIGURE TRUE
    USES_TERMINAL_BUILD     TRUE
)

add_library(blis STATIC IMPORTED GLOBAL)
set_target_properties(blis PROPERTIES
    IMPORTED_LOCATION             ${BLIS_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${BLIS_INCLUDE_DIR}
)
add_dependencies(blis blis_build)

# docopt: built as one TU rather than add_subdirectory, because its vendored
# CMakeLists requests cmake_minimum_required(3.1), which CMake >= 4.0 rejects.
add_library(docopt STATIC ${CMAKE_CURRENT_SOURCE_DIR}/docopt.cpp/docopt.cpp)
set_target_properties(docopt PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
target_include_directories(docopt SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/docopt.cpp)

# docopt reaches <algorithm> only transitively, which recent libc++ dropped.
target_compile_options(docopt PRIVATE -include algorithm)

# MArray (header-only).
add_library(marray INTERFACE)
target_include_directories(marray SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/marray/marray)

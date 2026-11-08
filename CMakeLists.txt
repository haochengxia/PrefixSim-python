cmake_minimum_required(VERSION 3.15...3.31)
project(prefixsim_python LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # static libs linked into a shared module

set(PREFIXSIM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/extern/PrefixSim)
if(NOT EXISTS ${PREFIXSIM_DIR}/libCacheSim/CMakeLists.txt)
    message(FATAL_ERROR "run `git submodule update --init --recursive`")
endif()

# tcmalloc must own the heap from process start, so it cannot come in with a module.
set(CMAKE_DISABLE_FIND_PACKAGE_Tcmalloc ON)

# For wheels: manylinux counts glib as a system library, so auditwheel does not
# bundle it, but slim images lack it. libCacheSim links glib by the name
# glib-2.0, and a target of that name takes precedence: point it at the static
# archive and its static dependencies (pkg-config --static).
option(PREFIXSIM_STATIC_GLIB "link glib statically" OFF)
if(PREFIXSIM_STATIC_GLIB)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLIB_STATIC REQUIRED glib-2.0)
    set(glib_deps "")
    foreach(lib IN LISTS GLIB_STATIC_STATIC_LIBRARIES)
        if(lib MATCHES "^(glib-2.0|m|dl|pthread|rt)$")
            continue()
        endif()
        find_library(${lib}_archive lib${lib}.a REQUIRED)
        list(APPEND glib_deps ${${lib}_archive})
    endforeach()
    find_library(glib_archive libglib-2.0.a REQUIRED)
    find_package(Threads REQUIRED)
    add_library(glib-2.0 STATIC IMPORTED GLOBAL)
    set_target_properties(glib-2.0 PROPERTIES IMPORTED_LOCATION ${glib_archive}
        INTERFACE_LINK_LIBRARIES "${glib_deps};Threads::Threads;m")
    message(STATUS "glib linked statically: ${glib_archive} ${glib_deps}")
endif()
# libCacheSim's xxhash (0.7.4) reads keys through a packed union when GCC targets
# ARM, which breaks strict aliasing: on Linux aarch64 at -O3 GCC hashes every
# inserted object as id 0, so lookups miss it. Use the memcpy reads that x86 and
# clang already get; they compile to the same loads. Fixed in libCacheSim by
# 1a1a11a/libCacheSim#383; drop this once PrefixSim's libCacheSim includes it.
add_compile_definitions(XXH_FORCE_MEMORY_ACCESS=0)
set(PREFIXSIM_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${PREFIXSIM_DIR} ${CMAKE_BINARY_DIR}/PrefixSim EXCLUDE_FROM_ALL)
# upstream's -Werror should not fail the build on a newer compiler
foreach(target libCacheSim prefixsim_lib)
    target_compile_options(${target} PRIVATE -Wno-error)
endforeach()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core MODULE csrc/bindings.cpp)
target_link_libraries(_core PRIVATE prefixsim_lib)
# libCacheSim links glib by name and adds its directory only in its own scope
# (see PrefixSim's CMakeLists); glib is off the default path on macOS
get_directory_property(glib_dirs DIRECTORY ${PREFIXSIM_DIR}/libCacheSim
    DEFINITION GLib_LIBRARY_DIRS)
target_link_directories(_core PRIVATE ${glib_dirs})
if(PREFIXSIM_STATIC_GLIB)
    target_link_options(_core PRIVATE -Wl,--exclude-libs,ALL)  # keep the bundled glib private
endif()
install(TARGETS _core LIBRARY DESTINATION prefixsim)

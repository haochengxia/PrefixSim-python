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
if(PREFIXSIM_STATIC_GLIB)
    target_link_options(_core PRIVATE -Wl,--exclude-libs,ALL)  # keep the bundled glib private
endif()
install(TARGETS _core LIBRARY DESTINATION prefixsim)

# brewpy: pybind11 bindings for BREW, auto-generated from the @mex annotations in the brew
# submodule (the same annotations that drive the MATLAB MEX build in brew-lab).
#
#   cmake -S . -B build -DBREW_TARGET=desktop && cmake --build build
#
# Builds _brewpy<ext> and copies it next to brewpy/__init__.py so `import brewpy` works.
cmake_minimum_required(VERSION 3.18)
project(brewpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# brew's static subsystem libs are linked into the shared _brewpy module, so they must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BREW_TARGET "desktop" CACHE STRING "brew build target (desktop needed for sampling/clustering)")
# Python verification uses BREW's estimators and samplers, while figures are
# rendered by matplotlib.  Excluding BREW's C++ plotting layer keeps this
# binding smaller and avoids linking an unrelated desktop plotting target.
set(BREW_ENABLE_PLOTTING OFF CACHE BOOL "" FORCE)
set(BREW_BUILD_TESTS OFF CACHE BOOL "Build BREW tests with the Python bindings")
set(BREW_BUILD_EXAMPLES OFF CACHE BOOL "Build BREW examples with the Python bindings")

# the brew library (INTERFACE target `brew` + its subsystem libs)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brew/CMakeLists.txt")
  message(FATAL_ERROR "BREW source is missing. Initialize the brew submodule before building: git submodule update --init --recursive")
endif()
add_subdirectory(brew)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# regenerate the bindings from the @mex annotations on every configure/build
set(GEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator_files)
set(GENERATED ${CMAKE_CURRENT_BINARY_DIR}/bindings_generated.cpp)
file(GLOB_RECURSE BREW_HEADERS CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/brew/include/brew/*.hpp)
add_custom_command(
  OUTPUT ${GENERATED}
  COMMAND ${Python3_EXECUTABLE} ${GEN_DIR}/generate_pybind.py
          --brew ${CMAKE_CURRENT_SOURCE_DIR}/brew --out ${GENERATED}
  DEPENDS ${GEN_DIR}/generate_pybind.py ${GEN_DIR}/pybind_static.cpp.in ${BREW_HEADERS}
  COMMENT "Generating brewpy bindings from @mex annotations"
  VERBATIM)

pybind11_add_module(_brewpy ${GENERATED})
target_include_directories(_brewpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/brew/include)
target_link_libraries(_brewpy PRIVATE brew)

# drop the built module into the python package so `import brewpy` resolves with no install
add_custom_command(TARGET _brewpy POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:_brewpy> ${CMAKE_CURRENT_SOURCE_DIR}/brewpy/$<TARGET_FILE_NAME:_brewpy>
  COMMENT "Copying _brewpy into brewpy/")

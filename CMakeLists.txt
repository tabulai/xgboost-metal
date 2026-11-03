cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS version")
endif()
project(xgboost_metal VERSION 3.5.0 LANGUAGES CXX OBJCXX)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  message(FATAL_ERROR "xgboost-metal requires Apple silicon and Metal. There is no CPU backend.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "14.0")
  message(FATAL_ERROR "Metal 3.1 kernels require macOS 14 or later.")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_program(XCRUN xcrun REQUIRED)

set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/xgboost.air)
set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/xgboost.metallib)
set(METAL_HEADER ${CMAKE_CURRENT_BINARY_DIR}/metal_library.hpp)
add_custom_command(OUTPUT ${METAL_HEADER}
  COMMAND ${XCRUN} -sdk macosx metal -std=metal3.1 -O3 -fno-fast-math
          -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
          -I${CMAKE_CURRENT_SOURCE_DIR}/include
          -c ${CMAKE_CURRENT_SOURCE_DIR}/kernels/boost.metal -o ${METAL_AIR}
  COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed.py ${METAL_LIB} ${METAL_HEADER}
  DEPENDS kernels/boost.metal kernels/quantile.metal.h kernels/sketch_cuts.metal.h kernels/sketch_summary.metal.h kernels/sketch_sort.metal.h kernels/sketch_merge.metal.h kernels/sketch_arithmetic.metal.h kernels/float_order.metal.h kernels/objectives.metal.h kernels/distributions.metal.h kernels/survival.metal.h kernels/survival_wide.metal.h kernels/survival_constants.metal.h kernels/cox.metal.h kernels/cox_scan.metal.h kernels/cox_math.metal.h kernels/cox_arithmetic.metal.h kernels/cox_constants.metal.h kernels/ranking.metal.h kernels/linear.metal.h kernels/refresh.metal.h kernels/sampling.metal.h kernels/vector_tree.metal.h include/metalgb/shared.h tools/embed.py
  DEPENDS kernels/split_statistics.metal.h kernels/scalar_root.metal.h kernels/scalar_gain.metal.h kernels/scalar_weight_fast.metal.h kernels/scalar_gain_fast.metal.h kernels/scalar_statistics_fast.metal.h kernels/scalar_score_fused.metal.h kernels/sketch_weighted.metal.h kernels/gradient_quantization.metal.h kernels/tree_driver.metal.h kernels/histogram_missing.metal.h kernels/histogram_present.metal.h kernels/histogram_compact.metal.h kernels/histogram_tiled.metal.h kernels/scalar_sparse.metal.h kernels/scalar_fused.metal.h kernels/scalar_parallel.metal.h kernels/scalar_compact.metal.h kernels/scalar_root_parallel.metal.h kernels/scalar_screen.metal.h kernels/tree_driver_fused.metal.h kernels/tree_driver_parallel.metal.h kernels/tree_emit.metal.h
  VERBATIM)

add_library(xgboost SHARED src/runtime.mm src/data.cpp src/matrix_binary.cpp src/quantile.cpp src/categories.cpp src/learner.cpp src/linear.cpp src/tree.cpp src/refresh.cpp src/sampling.cpp src/dropout.cpp src/objectives.cpp src/survival.cpp src/ranking.cpp src/model.cpp src/interpret.cpp src/metrics.cpp src/c_api.cpp ${METAL_HEADER})
target_include_directories(xgboost PUBLIC include PRIVATE third_party ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(xgboost PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_compile_options(xgboost PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
set_target_properties(xgboost PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
install(TARGETS xgboost LIBRARY DESTINATION xgboost/lib)
install(DIRECTORY include/ DESTINATION include)

option(METALGB_BUILD_TESTS "Build native GPU validation tests" OFF)
if(METALGB_BUILD_TESTS)
  # Certified split screening: a test-only metallib adds a probe kernel to the
  # production kernels; the driver compares enclosures and kernel records.
  set(SCREEN_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/split-screening-test)
  set(SCREEN_TEST_HEADER ${SCREEN_TEST_DIR}/metal_library.hpp)
  add_custom_command(OUTPUT ${SCREEN_TEST_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SCREEN_TEST_DIR}
    COMMAND ${XCRUN} -sdk macosx metal -std=metal3.1 -O3 -fno-fast-math
            -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/kernels
            -c ${CMAKE_CURRENT_SOURCE_DIR}/tests/split_screening_probe.metal -o ${SCREEN_TEST_DIR}/test.air
    COMMAND ${XCRUN} -sdk macosx metallib ${SCREEN_TEST_DIR}/test.air -o ${SCREEN_TEST_DIR}/test.metallib
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/embed.py
            ${SCREEN_TEST_DIR}/test.metallib ${SCREEN_TEST_HEADER}
    DEPENDS tests/split_screening_probe.metal ${METAL_HEADER}
    VERBATIM)
  add_executable(split_screening_bounds_test tests/split_screening_bounds.cpp src/runtime.mm ${SCREEN_TEST_HEADER})
  target_include_directories(split_screening_bounds_test PRIVATE include third_party ${SCREEN_TEST_DIR})
  target_link_libraries(split_screening_bounds_test PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

option(METALGB_BUILD_BENCHMARKS "Build native GPU benchmarks" OFF)
if(METALGB_BUILD_BENCHMARKS)
  add_executable(metal_bench benchmarks/bandwidth.cpp src/runtime.mm ${METAL_HEADER})
  target_include_directories(metal_bench PRIVATE include third_party ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(metal_bench PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

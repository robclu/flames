#==--- flame/CMakeLists.txt -------------------------------------------------==#
#
#                      Copyright (c) 2020 Rob Clucas
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(flame)

# Allow clang-format to configure from the build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#===== [packages] =============================================================#

# Change to path to pytorch install
set(CMAKE_PREFIX_PATH "~/opt/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#===== [subdirs] ==============================================================#

add_subdirectory(examples)

#==============================================================================#
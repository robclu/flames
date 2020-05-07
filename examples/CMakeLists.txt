#==--- flame/examples/CMakeLists.txt ----------------------------------------==#
#
#                      Copyright (c) 2020 Ripple
#
#  This file is distributed under the MIT License. See LICENSE for details.
#
#==--------------------------------------------------------------------------==#

#===== [sources] ==============================================================#

set(FLAME_INCLUDE_DIR ${flame_SOURCE_DIR}/include)
set(FLAME_SOURCE_DIR  ${flame_SOURCE_DIR}/src)
include_directories(${FLAME_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})

#===== [resnet_v2] ============================================================#

set(RESNETV2_SOURCES resnet_v2.cpp)
list(
  APPEND RESNETV2_SOURCES
    ${FLAME_SOURCE_DIR}/models/basic_block.cpp
    ${FLAME_SOURCE_DIR}/models/bottleneck.cpp
    ${FLAME_SOURCE_DIR}/util/conv.cpp
)

add_executable(resnetv2 ${RESNETV2_SOURCES})
target_link_libraries(resnetv2 "${TORCH_LIBRARIES}")
set_property(TARGET resnetv2 PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(
    TARGET resnetv2
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${TORCH_DLLS}
    $<TARGET_FILE_DIR:resnetv2>)
endif(MSVC)
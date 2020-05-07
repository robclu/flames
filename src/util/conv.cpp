//==--- flame/src/util/conv.cpp ---------------------------- -*- C++ -*- ---==//
//
//                            Ripple - Flame
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  conv.cpp
/// \brief Implementation file for convolution utilities.
//
//==------------------------------------------------------------------------==//

#include <ripple/flame/util/conv.hpp>

namespace ripple::flame {

auto conv_3x3(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t padding) -> torch::nn::Conv2d {
  constexpr int64_t kernel_size_xy = 3;
  constexpr bool    bias           = false;
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
      .stride(stride)
      .padding(padding)
      .bias(bias));
}

auto conv_1x1(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t padding) -> torch::nn::Conv2d {
  constexpr int64_t kernel_size_xy = 1;
  constexpr bool    bias           = false;
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
      .stride(stride)
      .padding(padding)
      .bias(bias));
}

} // namespace ripple::flame
//==--- flame/src/util/conv.cpp ---------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  conv.cpp
/// \brief Implementation file for convolution utilities.
//
//==------------------------------------------------------------------------==//

#include <flame/util/conv.hpp>

namespace flame {

auto conv_3x3_bn(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t padding) -> torch::nn::Sequential {
  constexpr int64_t kernel_size_xy = 1;
  constexpr bool    bias           = false;
  return torch::nn::Sequential{
    torch::nn::Conv2d(
      torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
        .stride(stride)
        .padding(padding)
        .bias(bias)),
    torch::nn::BatchNorm2d(output_channels),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))};
}

auto conv_3x3_bn(
  int64_t input_channels, int64_t output_channels, int64_t stride)
  -> torch::nn::Sequential {
  constexpr int64_t kernel_size_xy = 3;
  constexpr int64_t padding        = 1;
  constexpr bool    bias           = false;
  return torch::nn::Sequential{
    torch::nn::Conv2d(
      torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
        .stride(stride)
        .padding(padding)
        .bias(bias)),
    torch::nn::BatchNorm2d(output_channels),
    torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))};
}

auto conv_7x7(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t padding) -> torch::nn::Conv2d {
  constexpr int64_t kernel_size_xy = 7;
  constexpr bool    bias           = false;
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
      .stride(stride)
      .padding(padding)
      .bias(bias));
}

auto conv_5x5(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t padding) -> torch::nn::Conv2d {
  constexpr int64_t kernel_size_xy = 5;
  constexpr bool    bias           = false;
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
      .stride(stride)
      .padding(padding)
      .bias(bias));
}

auto conv_3x3(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride,
  int64_t groups,
  int64_t dilation) -> torch::nn::Conv2d {
  constexpr int64_t kernel_size_xy = 3;
  constexpr bool    bias           = false;
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size_xy)
      .stride(stride)
      .padding(dilation)
      .groups(groups)
      .dilation(dilation)
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

} // namespace flame
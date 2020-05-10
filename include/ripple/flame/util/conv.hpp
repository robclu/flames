//==--- flame/util/conv.hpp -------------------------------- -*- C++ -*- ---==//
//
//                            Ripple - Flame
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flame.hpp
/// \brief Header file for different convolution types.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FLAME_UTIL_CONV_HPP
#define RIPPLE_FLAME_UTIL_CONV_HPP

#include <torch/torch.h>

namespace ripple::flame {

//==--- [constants] --------------------------------------------------------==//

constexpr int64_t pad_0    = 0; //!< Constant for a padding of zero.
constexpr int64_t pad_1    = 1; //!< Constant for a padding of one.
constexpr int64_t stride_1 = 1; //!< Constants for a stride of one.
constexpr int64_t stride_2 = 2; //!< Constants for a stride of two.

//==--- [functions] --------------------------------------------------------==//

/// Returns a torch::nn::Conv2d operation with a 7x7 kernel, wiht no bias.
/// \param input_channels  The number of input channels.
/// \param output_channels The number of output channels.
/// \param stride          The stride for the convolution.
/// \param padding         The padding for the kernel.
auto conv_7x7(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride  = 1,
  int64_t padding = 1) -> torch::nn::Conv2d;

/// Returns a torch::nn::Conv2d operation with a 5x5 kernel, wiht no bias.
/// \param input_channels  The number of input channels.
/// \param output_channels The number of output channels.
/// \param stride          The stride for the convolution.
/// \param padding         The padding for the kernel.
auto conv_5x5(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride  = 1,
  int64_t padding = 1) -> torch::nn::Conv2d;

/// Returns a torch::nn::Conv2d operation with a 3x3 kernel, wiht no bias.
/// \param input_channels  The number of input channels.
/// \param output_channels The number of output channels.
/// \param stride          The stride for the convolution.
/// \param padding         The padding for the kernel.
auto conv_3x3(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride  = 1,
  int64_t padding = 1) -> torch::nn::Conv2d;

/// Returns a torch::nn::Conv2d operation with a 1x1 kernel, with no bias.
/// \param input_channels  The number of input channels.
/// \param output_channels The number of output channels.
/// \param stride          The stride for the convolution.
/// \param padding         The padding for the kernel.
auto conv_1x1(
  int64_t input_channels,
  int64_t output_channels,
  int64_t stride  = 1,
  int64_t padding = 1) -> torch::nn::Conv2d;

} // namespace ripple::flame

#endif // RIPPLE_FLAME_UTIL_CONV_HPP
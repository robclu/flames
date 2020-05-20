//==--- flame/models/bottleneck.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  bottleneck.hpp
/// \brief Header file for a bottleneck block for resnet V2.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_BOTTLENECK_HPP
#define FLAME_MODELS_BOTTLENECK_HPP

#include <torch/torch.h>

namespace flame::models {

/// This type defines a bottleneck block for resnet V2.
class BottleneckImpl : public torch::nn::Module {
  using Conv = torch::nn::Conv2d;      //!< Convolution type.
  using Norm = torch::nn::BatchNorm2d; //!< Norm type.
  using Relu = torch::nn::ReLU;        //!< Relu type.

 public:
  using DownSampler = torch::nn::Sequential; //!< Sampler type.

  /// The expansion factor for the block.
  static constexpr int64_t expansion = 4;

  // Default constructor.
  BottleneckImpl() = default;

  /// Constructor to configure the block.
  /// \param input_channels  The number of input channels.
  /// \param output_channels The number of output channels.
  /// \param stride          The stride for the convolution.
  /// \param downsamper      The downsampler for the block.
  BottleneckImpl(
    int64_t     input_channels,
    int64_t     output_channels,
    int64_t     stride      = 1,
    DownSampler downsampler = nullptr);

  /// Feeds the tensor \p x through the block in the forward direction,
  /// returning the result.
  /// \param x The input tensor to pass through the block.
  auto forward(torch::Tensor x) -> torch::Tensor;

 private:
  Conv        _conv_1      = nullptr; //!< 1st conv layer.
  Norm        _batchnorm_1 = nullptr; //!< 1st normalization layer.
  Conv        _conv_2      = nullptr; //!< 2nd conv layer.
  Norm        _batchnorm_2 = nullptr; //!< 2nd normalization layer.
  Conv        _conv_3      = nullptr; //!< 3rd conv layer.
  Norm        _batchnorm_3 = nullptr; //!< 3rd normlaization layer.
  Relu        _relu        = nullptr; //!< ReLu layer.
  DownSampler _downsampler = nullptr; //!< Downsampler.
};

/// Wrapper to make the Bottleneck block into a torch module. We don;t use the
/// macro here because we need to expose the expansion factor of the block.
class Bottleneck : public torch::nn::ModuleHolder<BottleneckImpl> {
 public:
  using torch::nn::ModuleHolder<BottleneckImpl>::ModuleHolder;

  /// Make the expansion available to networks using the block.
  static constexpr int64_t expansion = BottleneckImpl::expansion;
};

} // namespace flame::models

#endif // FLAME_MODELS_BOTTLENECK_HPP
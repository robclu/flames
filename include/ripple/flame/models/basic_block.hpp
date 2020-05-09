//==--- flame/models/basic_block.hpp ----------------------- -*- C++ -*- ---==//
//
//                           Ripple - Flame
//
//                      Copyright (c) 2020 Ripple
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_block.hpp
/// \brief Header file for a basic block for a residual network.
//
//==------------------------------------------------------------------------==//

#ifndef RIPPLE_FLAME_MODELS_BASIC_BLOCK_HPP
#define RIPPLE_FLAME_MODELS_BASIC_BLOCK_HPP

#include <torch/torch.h>

namespace ripple::flame::models {

/// This type defines a block in a residual neural network.
class BasicBlockImpl : public torch::nn::Module {
  using Conv = torch::nn::Conv2d;      //!< Convolution type.
  using Norm = torch::nn::BatchNorm2d; //!< Norm type.
  using Relu = torch::nn::ReLU;        //!< Relu type.

 public:
  using DownSampler = torch::nn::Sequential; //!< Sampler type.

  /// Defines the expansion factor for the block.
  static constexpr int64_t expansion = 1;

  /// Default constructor.
  BasicBlockImpl() = default;

  /// Constructor to configure the block.
  /// \param input_channels  The number of input channels.
  /// \param output_channels The number of output channels.
  /// \param stride          The stride for the convolution.
  /// \param downsamper      The downsampler for the block.
  BasicBlockImpl(
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
  Norm        _batchnorm_2 = nullptr; //!< 2nd normlaization layer.
  Relu        _relu        = nullptr; //!< ReLu layer.
  DownSampler _downsampler = nullptr; //!< Downsampler.
};

// Make the implementation into a torch module.
TORCH_MODULE(BasicBlock);

} // namespace ripple::flame::models

#endif // RIPPLE_FLAME_MODELS_BASIC_BLOCK_HPP
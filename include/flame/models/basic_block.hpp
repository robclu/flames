//==--- flame/models/basic_block.hpp ----------------------- -*- C++ -*- ---==//
//
//                              Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_block.hpp
/// \brief Header file for a basic block for a residual network.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_BASIC_BLOCK_HPP
#define FLAME_MODELS_BASIC_BLOCK_HPP

#include <torch/torch.h>

namespace flame::models {

/// This type defines a block in a residual neural network.
class BasicBlockImpl : public torch::nn::Module {
  using Conv = torch::nn::Conv2d;      //!< Convolution type.
  using Norm = torch::nn::BatchNorm2d; //!< Norm type.
  using Relu = torch::nn::ReLU;        //!< Relu type.

 public:
  using DownSampler = torch::nn::Sequential; //!< Sampler type.

  /// The expansion factor for the block.
  static constexpr int64_t expansion = 1;

  /// Default constructor.
  BasicBlockImpl() = default;

  /// \param inplanes   The number of input planes
  /// \param planes     The number of output planes.
  /// \param stride     The stride for the convolution.
  /// \param downsamper The downsampler for the block.
  /// \param groups     The numebr of groups in the convolution.
  /// \param base_width The base width (number of planes) of the convolution.
  /// \param dilation   The dilation for the convolution.
  BasicBlockImpl(
    int64_t     inplanes,
    int64_t     planes,
    int64_t     stride      = 1,
    DownSampler downsampler = nullptr,
    int64_t     groups      = 1,
    int64_t     base_width  = 64,
    int64_t     dilation    = 1);

  /// Feeds the tensor \p x through the block in the forward direction,
  /// returning the result.
  /// \param x The input tensor to pass through the block.
  auto forward(const torch::Tensor& x) -> torch::Tensor;

  /// Zero initializes the last residual layer.
  auto zero_init_residual() -> void;

 private:
  Conv        conv_1_      = nullptr; //!< 1st conv layer.
  Norm        batchnorm_1_ = nullptr; //!< 1st normalization layer.
  Conv        conv_2_      = nullptr; //!< 2nd conv layer.
  Norm        batchnorm_2_ = nullptr; //!< 2nd normlaization layer.
  Relu        relu_        = nullptr; //!< ReLu layer.
  DownSampler downsampler_ = nullptr; //!< Downsampler.
};

/// Wrapper to make the Bottleneck block into a torch module. We don;t use the
/// macro here because we need to expose the expansion factor of the block.
class BasicBlock : public torch::nn::ModuleHolder<BasicBlockImpl> {
 public:
  using torch::nn::ModuleHolder<BasicBlockImpl>::ModuleHolder;

  /// Make the expansion available to networks using the block.
  static constexpr int64_t expansion = BasicBlockImpl::expansion;
};

} // namespace flame::models

#endif // FLAME_MODELS_BASIC_BLOCK_HPP
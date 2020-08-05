//==--- flame/models/sls_block.hpp ------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SlsBlock.hpp
/// \brief Header file for Select SLS Block.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_SLS_BLOCK_HPP
#define FLAME_MODELS_SLS_BLOCK_HPP

#include <torch/torch.h>
#include <vector>

namespace flame::models {

/// Options for an SLS block.
struct SlsBlockOptions {
  int64_t inplanes;         //!< Number of input planes.
  int64_t skip;             //!< Number of skip inputs.
  int64_t planes;           //!< Number of planes for block middle layers
  int64_t outplanes;        //!< Number of output planes.
  int64_t stride   = 1;     //!< Stride for the block.
  bool    is_first = false; //!< If the block is the first block.
};

/// This class defines and SLS Block from the paper
/// XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB
/// Camera, Mehta et al. 2019 : https://arxiv.org/abs/1907.00837.
class SlsBlockImpl : public torch::nn::Module {
  using Layer = torch::nn::Sequential;  //!< Layer type.
  using Conv  = torch::nn::Conv2d;      //!< Convolution type.
  using Norm  = torch::nn::BatchNorm2d; //!< Norm type.
  using Relu  = torch::nn::ReLU;        //!< Relu type.

 public:
  using TensorList = std::vector<torch::Tensor>; //!< List of tensors.

  /// Default constructor.
  SlsBlockImpl() = default;

  /// Constructor to create the block.
  /// \param inplanes  The number of input planes.
  /// \param skip      Number of skips to add to the last layer.
  /// \param planes    The number of planes in the middle layers.
  /// \param outplanes The number of output planes.
  /// \param is_first  If this is the first block.
  /// \param stride    The stride for the block convolutions.
  SlsBlockImpl(
    int64_t inplanes,
    int64_t skip,
    int64_t planes,
    int64_t outplanes,
    bool    is_first = false,
    int64_t stride   = 1);

  /// Constructor to create the block from the block \p options.
  /// \param options The options for the bl
  SlsBlockImpl(const SlsBlockOptions& options);

  /// Feeds the first tensor from \p x through the block in the forward
  /// direction, and uses the second as the input to the skip if this block
  /// is not the first block.
  ///
  /// This will fail if `x.size() == 1 && !is_first` or `x.size() == 2 &&
  /// is_first()`.
  ///
  /// It returns a new tensor list.
  ///
  /// \param x The input tensors to pass through the block.
  auto forward(const TensorList& x) -> TensorList;

 private:
  Layer conv_1_   = nullptr; //!< First conv layer.
  Layer conv_2_   = nullptr; //!< Second conv layer.
  Layer conv_3_   = nullptr; //!< Third conv layer.
  Layer conv_4_   = nullptr; //!< Fourth conv layer.
  Layer conv_5_   = nullptr; //!< Fifth conv layer.
  Layer conv_6_   = nullptr; //!< Sixth conv layer.
  bool  is_first_ = false;   //!< If the first block in the chain.

  /// Makes a layer in the block, returning the new layer.
  /// \param inplanes     The number of input planes.
  /// \param outplanes    The number of output planes.
  /// \param kernel_width The width of the convolutional kernel.
  /// \param stride       The stride for the convolution.
  /// \param passing      The padding for the convolution.
  static auto make_layer(
    int64_t inplanes,
    int64_t outplanes,
    int64_t kernel_width,
    int64_t stride,
    int64_t padding) -> Layer;
};

/// Wrapper to make the sls block into a torch module.
TORCH_MODULE(SlsBlock);

} // namespace flame::models

#endif // FLAME_MODELS_SLS_BLOCK_HPP

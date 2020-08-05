//==--- flame/models/resnet.hpp ---------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  resnet.hpp
/// \brief Header file for a resnet model.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_RESNET_HPP
#define FLAME_MODELS_RESNET_HPP

#include "basic_block.hpp"
#include "bottleneck.hpp"
#include <flame/util/conv.hpp>
#include <flame/util/load.hpp>
#include <array>

namespace flame::models {

/// Thhis class defines an implementation of a residual neural network,
/// which has the same form as the Resnet implementation as in the pytoch python
/// library, and can be loaded as pretrained.
/// \tparam Block The type of the block for the network.
template <typename Block>
class ResnetImpl : public torch::nn::Module {
  using Conv          = torch::nn::Conv2d;            //!< Type of convolution.
  using Norm          = torch::nn::BatchNorm2d;       //!< Normalization type.
  using Relu          = torch::nn::ReLU;              //!< Relu op type.
  using AvgPool       = torch::nn::AdaptiveAvgPool2d; //!< Average pool type.
  using MaxPool       = torch::nn::MaxPool2d;         //!< Max pool type.
  using FeatConnector = torch::nn::Linear;            //!< Feature connector.
  using Layer         = torch::nn::Sequential;        //!< layer types.

  //==--- [default sizes] --------------------------------------------------==//

  // clang-format off
  /// Number of outputs for the first convolution.
  static constexpr int64_t first_conv_outputs = 64;
  /// Default number of input channels.
  static constexpr int64_t in_channels_def    = 3;
  /// Size of the maxpool kernel.
  static constexpr int64_t maxpool_size       = 3;
  /// Size of the adaptive avg pool kernel.
  static constexpr int64_t avgpool_size       = 1;

  //==--- [layer output channels] ------------------------------------------==//

  /// Number of output channels for the first layer.
  static constexpr int64_t layer_1_out_channels = 64;
  /// Number of output channels for the second layer.
  static constexpr int64_t layer_2_out_channels = 128;
  /// Number of output channels for the third layer.
  static constexpr int64_t layer_3_out_channels = 256;
  /// Number of output channels for the fourth layer.
  static constexpr int64_t layer_4_out_channels = 512;
  // clang-format on

 public:
  /// Type of container for specifying the layer sizes.
  using LayerSizes = std::array<int64_t, 4>;

  /// Constructor to set the number of blocks for each of the layers in the
  /// network, and the number of classes.
  /// \param layer_sizes         Number of blocks in each of the four layers.
  /// \param classes             The number of output classes. If this is zero,
  ///                            then the features from the last layer are
  ///                            returned.
  /// \param zero_init_residual  If the last layer residuals in the blocks must
  ///                            be set to zero.
  /// \param groups              The number of groups for convolutions.
  /// \param width_per_group     The convolution width for the groups.
  /// \param dilation_for_stride If the stride for a layer must be replaced
  ///                            with dilation.
  ResnetImpl(
    const LayerSizes&    layer_sizes,
    int64_t              classes             = 1000,
    int64_t              zero_init_residual  = false,
    int64_t              groups              = 1,
    int64_t              width_per_group     = 64,
    std::vector<int64_t> dilation_for_stride = {false, false, false})
  : classes_{classes},
    groups_{groups},
    base_width_{width_per_group},
    conv_1_{default_first_conv()},
    batchnorm_1_{first_conv_outputs},
    relu_{relu_options()},
    max_pool_{maxpool_options()},
    avg_pool_{avgpool_options()},
    fc_{layer_4_out_channels * Block::expansion, classes} {
    layer_1_ = make_layer(layer_1_out_channels, layer_sizes[0]);
    layer_2_ = make_layer(
      layer_2_out_channels, layer_sizes[1], stride_2, dilation_for_stride[0]);
    layer_3_ = make_layer(
      layer_3_out_channels, layer_sizes[2], stride_2, dilation_for_stride[1]);
    layer_4_ = make_layer(
      layer_4_out_channels, layer_sizes[3], stride_2, dilation_for_stride[2]);

    register_module("conv1", conv_1_);
    register_module("bn1", batchnorm_1_);
    register_module("relu", relu_);
    register_module("maxpool", max_pool_);
    register_module("layer1", layer_1_);
    register_module("layer2", layer_2_);
    register_module("layer3", layer_3_);
    register_module("layer4", layer_4_);
    register_module("avgpool", avg_pool_);
    register_module("fc", fc_);

    namespace nn = torch::nn;
    for (auto& module : modules(false)) {
      if (auto mod = dynamic_cast<nn::Conv2dImpl*>(module.get())) {
        nn::init::kaiming_normal_(mod->weight, 0, torch::kFanOut, torch::kReLU);
      } else if (auto mod = dynamic_cast<nn::BatchNorm2dImpl*>(module.get())) {
        nn::init::constant_(mod->weight, 1);
        nn::init::constant_(mod->bias, 0);
      }
    }

    // Zero - initialize the last BN in each residual branch,
    // so that the residual branch starts with  zeros,
    // and each residual block behaves like an identity.
    //  This improves the model by 0.2 ~0.3 %
    // according to https: // arxiv.org/abs/1706.02677
    if (zero_init_residual) {
      for (auto& module : modules(false)) {
        if (auto* mod = dynamic_cast<BottleneckImpl*>(module.get())) {
          mod->zero_init_residual();
        }
      }
    }
  }

  /// Forward pass for the network, returning the result.
  /// \param x The input tensor to pass through the network.
  auto forward(const torch::Tensor& x) -> torch::Tensor {
    auto out = conv_1_->forward(x);
    out      = batchnorm_1_->forward(out);
    out      = relu_->forward(out);
    out      = max_pool_->forward(out);

    out = layer_1_->forward(out);
    out = layer_2_->forward(out);
    out = layer_3_->forward(out);
    out = layer_4_->forward(out);
    out = avg_pool_->forward(out);

    // Flatten all dimensions past the batch dimension into a single dimension:
    out = out.view({out.size(0), -1});

    if (classes_ != 0) {
      out = fc_->forward(out);
    }
    return out;
  }

 private:
  int64_t       inplanes_    = first_conv_outputs; //!< Number of inplanes.
  int64_t       classes_     = 0;                  //!< Number of classes.
  int64_t       dilation_    = 1;                  //!< Dilation amount.
  int64_t       groups_      = 1;                  //!< Convolution groups.
  int64_t       base_width_  = 64;                 //!< Convolution width.
  Conv          conv_1_      = nullptr;            //!< Convolution layer.
  Norm          batchnorm_1_ = nullptr;            //!< Batchnorm layer.
  Relu          relu_        = nullptr;            //!< Relu layer.
  MaxPool       max_pool_    = nullptr;            //!< Max pooling layer.
  AvgPool       avg_pool_    = nullptr;            //!< Average pool layer.
  FeatConnector fc_          = nullptr;            //!< Feature connector.
  Layer         layer_1_;                          //!< First layer of blocks.
  Layer         layer_2_;                          //!< Second layer of block.
  Layer         layer_3_;                          //!< Third layer of blocks.
  Layer         layer_4_;                          //!< Fourth layer of blocks

  /// Returns the defualy convolution for the first layer.
  static auto default_first_conv() -> Conv {
    constexpr int64_t in_channels = 3;
    constexpr int64_t padding     = 3;
    return conv_7x7(in_channels, first_conv_outputs, stride_2, padding);
  }

  /// Returns the default relu options.
  static auto relu_options() -> torch::nn::ReLUOptions {
    return torch::nn::ReLUOptions().inplace(true);
  }

  /// Returns the max pool options.
  static auto maxpool_options() -> torch::nn::MaxPool2dOptions {
    return torch::nn::MaxPool2dOptions(maxpool_size)
      .stride(stride_2)
      .padding(pad_1);
  }

  /// Returns the averag pool options.
  static auto avgpool_options() -> torch::nn::AdaptiveAvgPool2dOptions {
    return torch::nn::AdaptiveAvgPool2dOptions(avgpool_size);
  }

  /// Makes a layer with \p planes output pleanes, \p blocks blocks in
  /// the layer, and \p stride for the convolutions in the layer. It returns the
  /// newly created layer.
  ///
  /// \post The inplanes_ class member is modified to represent the number of
  ///       inputs to a layer which follows this.
  ///
  /// \param planes The number of output planes
  /// \param blocks The number of blocks in the layer.
  /// \param stride The stride in the layer convolutions.
  /// \param dilate If the stride must be replaced with dilation.
  auto make_layer(
    int64_t planes, int64_t blocks, int64_t stride = 1, bool dilate = false)
    -> Layer {
    Layer      layer;
    Layer      downsample    = nullptr;
    const auto out_planes    = planes * Block::expansion;
    const auto prev_dilation = dilation_;

    if (dilate) {
      dilation_ *= stride;
      stride = 1;
    }

    if (stride != stride_1 || inplanes_ != out_planes) {
      downsample =
        Layer{conv_1x1(inplanes_, out_planes, stride), Norm(out_planes)};
    }

    layer->push_back(Block(
      inplanes_,
      planes,
      stride,
      downsample,
      groups_,
      base_width_,
      prev_dilation));
    inplanes_ = out_planes;

    for (int64_t i = 1; i != blocks; ++i) {
      layer->push_back(Block(
        inplanes_, planes, stride_1, nullptr, groups_, base_width_, dilation_));
    }
    return layer;
  }
};

/// Make the resnet implementation into a torch module.
/// See `pimpl.h` in pytorch documentation.
/// The torch module macro can't be used here here because of the template.
/// \tparam Block The type of the block for the network.
template <typename Block>
class Resnet : public torch::nn::ModuleHolder<ResnetImpl<Block>> {
 public:
  /// Inherit all the base functionality.
  using torch::nn::ModuleHolder<ResnetImpl<Block>>::ModuleHolder;
};

//==--- [resnet implementations] -------------------------------------------==//

/// Creates a resnet 18 model.
/// \param classes             The number of output classes. If this is zero,
///                            then the features from the last layer are
///                            returned.
/// \param pretrained          If a pretrained model should be returned.
/// \param zero_init_residual  If the last layer residuals in the blocks must
///                            be set to zero.
/// \param groups              The number of groups for convolutions.
/// \param width_per_group     The convolution width for the groups.
/// \param dilation_for_stride If the stride for a layer must be replaced
///                            with dilation.
auto resnet18(
  int64_t              classes             = 1000,
  bool                 pretrained          = false,
  int64_t              zero_init_residual  = false,
  int64_t              groups              = 1,
  int64_t              width_per_group     = 64,
  std::vector<int64_t> dilation_for_stride = {false, false, false})
  -> Resnet<BasicBlock> {
  std::array<int64_t, 4> layers = {2, 2, 2, 2};
  auto                   model  = Resnet<BasicBlock>{layers,
                                  classes,
                                  zero_init_residual,
                                  groups,
                                  width_per_group,
                                  std::move(dilation_for_stride)};

  if (pretrained) {
    load_pretrained(model, "resnet_18_pretrained.pt");
  }
  return model;
}

/// Creates a resnet 34 model.
/// \param classes             The number of output classes. If this is zero,
///                            then the features from the last layer are
///                            returned.
/// \param pretrained          If a pretrained model should be returned.
/// \param zero_init_residual  If the last layer residuals in the blocks must
///                            be set to zero.
/// \param groups              The number of groups for convolutions.
/// \param width_per_group     The convolution width for the groups.
/// \param dilation_for_stride If the stride for a layer must be replaced
///                            with dilation.
auto resnet34(
  int64_t              classes             = 1000,
  bool                 pretrained          = false,
  int64_t              zero_init_residual  = false,
  int64_t              groups              = 1,
  int64_t              width_per_group     = 64,
  std::vector<int64_t> dilation_for_stride = {false, false, false})
  -> Resnet<BasicBlock> {
  std::array<int64_t, 4> layers = {3, 4, 6, 3};
  auto                   model  = Resnet<BasicBlock>{layers,
                                  classes,
                                  zero_init_residual,
                                  groups,
                                  width_per_group,
                                  std::move(dilation_for_stride)};

  if (pretrained) {
    load_pretrained(model, "resnet_34_pretrained.pt");
  }
  return model;
}

/// Creates a resnet 50 model.
/// \param classes             The number of output classes. If this is zero,
///                            then the features from the last layer are
///                            returned.
/// \param pretrained          If a pretrained model should be returned.
/// \param zero_init_residual  If the last layer residuals in the blocks must
///                            be set to zero.
/// \param groups              The number of groups for convolutions.
/// \param width_per_group     The convolution width for the groups.
/// \param dilation_for_stride If the stride for a layer must be replaced
///                            with dilation.
auto resnet50(
  int64_t              classes             = 1000,
  bool                 pretrained          = false,
  int64_t              zero_init_residual  = false,
  int64_t              groups              = 1,
  int64_t              width_per_group     = 64,
  std::vector<int64_t> dilation_for_stride = {false, false, false})
  -> Resnet<Bottleneck> {
  std::array<int64_t, 4> layers = {3, 4, 6, 3};
  auto                   model  = Resnet<Bottleneck>{layers,
                                  classes,
                                  zero_init_residual,
                                  groups,
                                  width_per_group,
                                  std::move(dilation_for_stride)};
  if (pretrained) {
    load_pretrained(model, "resnet_50_pretrained.pt");
  }
  return model;
}

} // namespace flame::models

#endif // FLAME_MODELS_RESNET_HPP

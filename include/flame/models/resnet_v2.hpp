//==--- flame/models/resnet_v2.hpp ------------------------- -*- C++ -*- ---==//
//
//                                Ripple
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  resnet_v2.hpp
/// \brief Header file for a resnet V2 model.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_RESNET_V2_HPP
#define FLAME_MODELS_RESNET_V2_HPP

#include <flame/util/conv.hpp>
#include <array>

namespace flame::models {

/// Thhis class defines an implementation of a residual neural network,
/// version 2. It can be configured by changing the type of the block.
/// \tparam Block The type of the block for the network.
template <typename Block>
class ResnetV2Impl : public torch::nn::Module {
  using Conv          = torch::nn::Conv2d;      //!< Type of convolution.
  using Norm          = torch::nn::BatchNorm2d; //!< Normalization type.
  using Relu          = torch::nn::ReLU;        //!< Relu op type.
  using AvgPool       = torch::nn::AvgPool2d;   //!< Average pool type.
  using MaxPool       = torch::nn::MaxPool2d;   //!< Max pool type.
  using FeatConnector = torch::nn::Linear;      //!< Feature connector type.
  using Layer         = torch::nn::Sequential;  //!< The type of the layers.

  //==--- [default sizes] --------------------------------------------------==//

  static constexpr int64_t first_conv_outputs = 64;
  static constexpr int64_t classes_def        = 1000;
  static constexpr int64_t in_channels_def    = 3;
  static constexpr int64_t maxpool_size       = 7;
  static constexpr int64_t avgpool_size       = 7;

  //==--- [layer output channels] ------------------------------------------==//

  static constexpr int64_t layer_1_out_channels = 64;
  static constexpr int64_t layer_2_out_channels = 128;
  static constexpr int64_t layer_3_out_channels = 256;
  static constexpr int64_t layer_4_out_channels = 512;

 public:
  /// Type of container for specifying the layer sizes.
  using LayerSizes = std::array<int64_t, 4>;

  /// Constructor to set the number of blocks for each of the layers in the
  /// network, and the number of classes.
  /// \param layer_sizes The number of blocks in each of the four layers.
  /// \param classes     The number of output classes. If this is zero, then
  ///                    the features from the last layer are returned.
  ResnetV2Impl(const LayerSizes& layer_sizes, int64_t classes = classes_def)
  : _classes{classes},
    _conv{default_first_conv()},
    _batchnorm{first_conv_outputs},
    _relu{relu_options()},
    _max_pool{maxpool_options()},
    _layer_1{make_layer(layer_1_out_channels, layer_sizes[0], stride_2)},
    _layer_2{make_layer(layer_2_out_channels, layer_sizes[0], stride_2)},
    _layer_3{make_layer(layer_3_out_channels, layer_sizes[0], stride_2)},
    _layer_4{make_layer(layer_4_out_channels, layer_sizes[0])},
    _avg_pool{avgpool_options()},
    _fc{_in_channels, classes} {
    register_module("conv", _conv);
    register_module("batchnorm", _batchnorm);
    register_module("relu", _relu);
    register_module("max_pool", _max_pool);
    register_module("layer_1", _layer_1);
    register_module("layer_2", _layer_2);
    register_module("layer_3", _layer_3);
    register_module("layer_4", _layer_4);
    register_module("average_pool", _avg_pool);
    register_module("feature_connector", _fc);
  }

  /// Forward pass for the network, returning the result.
  /// \param x The input tensor to pass through the network.
  auto forward(torch::Tensor x) -> torch::Tensor {
    auto out = _conv->forward(x);
    out      = _batchnorm->forward(out);
    out      = _relu->forward(out);
    out      = _max_pool->forward(out);

    out = _layer_1->forward(out);
    out = _layer_2->forward(out);
    out = _layer_3->forward(out);
    out = _layer_4->forward(out);
    out = _avg_pool->forward(out);

    // Flatten all dimensions past the batch dimension into a single dimension:
    out = out.view({out.size(0), -1});

    if (_classes != 0) {
      out = _fc->forward(out);
    }
    return out;
  }

 private:
  int64_t _in_channels = first_conv_outputs; //!< Number of input channels.
  int64_t _classes     = 0;                  //!< The number of classes.

  Conv          _conv      = nullptr; //!< First convoltional layer.
  Norm          _batchnorm = nullptr; //!< Batchnorm layer.
  Relu          _relu      = nullptr; //!< Relu layer.
  MaxPool       _max_pool  = nullptr; //!< Max pooling layer.
  Layer         _layer_1;             //!< First layer of blocks.
  Layer         _layer_2;             //!< Second layer of block.
  Layer         _layer_3;             //!< Third layer of blocks.
  Layer         _layer_4;             //!< Fourth layer of blocks
  AvgPool       _avg_pool = nullptr;  //!< Average pool layer.
  FeatConnector _fc       = nullptr;  //!< Feature connector.

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
  static auto avgpool_options() -> torch::nn::AvgPool2dOptions {
    return torch::nn::AvgPool2dOptions(avgpool_size).stride(stride_1);
  }

  /// Makes a layer with \p out_channels output channels, \p blocks blocks in
  /// the layer, and \p stride for the convolutions in the layer. It returns the
  /// newly created layer.
  ///
  /// \post The _in_channels class member is modified to represent the number of
  ///       inputs to a layer which follows this.
  ///
  /// \param out_channels The number of output channels.
  /// \param blocks       The number of blocks in the layer.
  /// \param stride       The stride in the layer convolutions.
  auto make_layer(int64_t out_channels, int64_t blocks, int64_t stride = 1)
    -> Layer {
    Layer      layer;
    Layer      downsample = nullptr;
    const auto out_planes = out_channels * Block::expansion;
    if (stride != stride_1 || _in_channels != out_planes) {
      downsample =
        Layer{conv_1x1(_in_channels, out_planes, stride), Norm(out_planes)};
    }

    layer->push_back(Block(_in_channels, out_channels, stride, downsample));
    _in_channels = out_planes;

    for (int64_t i = 1; i != blocks; ++i) {
      layer->push_back(Block(_in_channels, out_channels));
    }
    return layer;
  }
};

/// Make the resnet implementation into a torch module.
/// See `pimpl.h` in pytorch documentation.
/// The torch module macro can't be used here here because of the template.
/// \tparam Block The type of the block for the network.
template <typename Block>
class ResnetV2 : public torch::nn::ModuleHolder<ResnetV2Impl<Block>> {
 public:
  /// Inherit all the base functionality.
  using torch::nn::ModuleHolder<ResnetV2Impl<Block>>::ModuleHolder;
};

/// Creates a RednetV2-50 model, with the Block type of block.
/// \param classes    The number of classes for the model. If 0, then the output
///                   of the forward pass returns the features instead of the
///                   classification.
/// \param pretrained If a pretrained model should be returned.
/// \tparam Block     The type of the block for the model.
template <typename Block>
auto resnet_v2_50(int64_t classes = 1000, bool pretrained = false)
  -> ResnetV2<Block> {
  auto layer_sizes = std::array<int64_t, 4>{3, 4, 6, 3};
  return ResnetV2<Block>{layer_sizes, classes};
}

} // namespace flame::models

#endif // FLAME_MODELS_RESNET_V2_HPP

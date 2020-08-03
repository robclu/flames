//==--- src/models/sls_block.cpp --------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  sls_block.cpp
/// \brief Implementation file for a SLS block.
//
//==------------------------------------------------------------------------==//

#include <flame/models/sls_block.hpp>
#include <flame/util/conv.hpp>

namespace flame::models {

SlsBlockImpl::SlsBlockImpl(
  int64_t inplanes,
  int64_t skip,
  int64_t planes,
  int64_t outplanes,
  bool    is_first = false,
  int64_t stride   = 1)
: conv_1_{make_layer(inplanes, planes, 3, stride, 1)},
  conv_2_{make_layer(planes, planes, 1, 1, 0)},
  conv_3_{make_layer(planes, planes / 2, 3, 1, 1)},
  conv_4_{make_layer(planes / 2, planes, 1, 1 0)},
  conv_5_{make_layer(planes, planes / 2, 3, 1, 1)},
  conv_6_{make_layer(2 * planes + (is_first ? 0 : skip), outplanes, 1, 1, 0)},
  is_first_{is_first} {
  register_module("conv1", conv_1_);
  register_module("conv2", conv_2_);
  register_module("conv3", conv_3_);
  register_module("conv4", conv_4_);
  register_module("conv5", conv_5_);
  register_module("conv6", conv_6_);
}

auto SlsBlockImpl::forward(const std::vector<torch::Tensor>& x)
  -> std::vector<torch::Tensor> {
  if (is_first_ && x.size() != 1 || !is_first_ && x.size() != 2) {
    assert(false && "Invalid tensor input size!");
  }
  const auto& input = x[0];
  auto        d1    = conv_1_->forward(input);
  auto        d2    = conv_3_->forward(conv_2_->forward(d1));
  auto        d3    = conv_5_->forward(conv_4_->forward(d2));
  if (is_first_) {
    auto out = conv_6_->forward(torch::cat({d1, d2, d3}, 1));
    return {out, std::move(out)};
  }
  return {conv_6_->forward(torch::cat({d1, d2, d3, x[1]}, 1)), x[1]};
}

auto SlsBlockImpl::make_layer(
  int64_t inplanes,
  int64_t outplanes,
  int64_t kernel_width,
  int64_t stride,
  int64_t padding) -> Layer {
  torch::nn::Conv2dOptions ops(inplanes, outplanes)
    .kernel_size(kernel_size)
    .stride(stride)
    .padding(padding)
    .bias(false)
    .dilation(1)
    .groups(1);

  return Layer{
    {Conv(ops), Norm(outplanes), Relu(torch::nn::ReLUOptions().inplace(true))}};
}

} // namespace flame::models
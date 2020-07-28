//==--- src/models/bottleneck.cpp -------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  bottlenck.cpp
/// \brief Implementation file for a bottleneck block.
//
//==------------------------------------------------------------------------==//

#include <flame/models/bottleneck.hpp>
#include <flame/util/conv.hpp>

namespace flame::models {

BottleneckImpl::BottleneckImpl(
  int64_t     inplanes,
  int64_t     planes,
  int64_t     stride,
  DownSampler downsampler,
  int64_t     groups,
  int64_t     base_width,
  int64_t     dilation)
: relu_(torch::nn::ReLUOptions().inplace(true)), downsampler_(downsampler) {
  int64_t width = planes * (base_width / 64.0) * groups;

  conv_1_      = conv_1x1(inplanes, width);
  batchnorm_1_ = Norm(width);
  conv_2_      = conv_3x3(width, width, stride, groups, dilation);
  batchnorm_2_ = Norm(width);
  conv_3_      = conv_1x1(width, planes * expansion);
  batchnorm_3_ = Norm(planes * expansion);

  register_module("conv1", conv_1_);
  register_module("bn1", batchnorm_1_);
  register_module("conv2", conv_2_);
  register_module("bn2", batchnorm_2_);
  register_module("conv3", conv_3_);
  register_module("bn3", batchnorm_3_);
  register_module("relu", relu_);

  if (downsampler) {
    register_module("downsample", downsampler_);
  }
}

auto BottleneckImpl::forward(const torch::Tensor& x) -> torch::Tensor {
  auto out = conv_1_->forward(x);
  out      = batchnorm_1_->forward(out);
  out      = relu_->forward(out);
  out      = conv_2_->forward(out);
  out      = batchnorm_2_->forward(out);
  out      = relu_->forward(out);
  out      = conv_3_->forward(out);
  out      = batchnorm_3_->forward(out);

  auto residual = downsampler_ ? downsampler_->forward(x) : x;

  out += residual;
  out = relu_->forward(out);

  return out;
}

auto BottleneckImpl::zero_init_residual() -> void {
  torch::nn::init::constant_(batchnorm_3_->weight, 0);
}

} // namespace flame::models
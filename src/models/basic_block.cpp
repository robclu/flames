//==--- src/models/basic_block.cpp ------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Flame
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_block.cpp
/// \brief Implementation file for a basic block for a resenet.
//
//==------------------------------------------------------------------------==//

#include <flame/models/basic_block.hpp>
#include <flame/util/conv.hpp>

namespace flame::models {

BasicBlockImpl::BasicBlockImpl(
  int64_t     inplanes,
  int64_t     planes,
  int64_t     stride,
  DownSampler downsampler,
  int64_t     groups,
  int64_t     base_width,
  int64_t     dilation)
: conv_1_{conv_3x3(inplanes, planes, stride)},
  batchnorm_1_{planes},
  conv_2_{conv_3x3(planes, planes)},
  batchnorm_2_{planes},
  relu_{torch::nn::ReLUOptions().inplace(true)},
  downsampler_{downsampler} {
  if (groups != 1 || base_width != 64) {
    assert(false && "BasicBlock only supports 1 groups and base width of 64!");
  }

  if (dilation > 1) {
    assert(false && "BasicBlock does not yet supports dilation > 1!");
  }

  register_module("conv1", conv_1_);
  register_module("bn1", batchnorm_1_);
  register_module("conv2", conv_2_);
  register_module("bn2", batchnorm_2_);
  register_module("relu", relu_);

  if (downsampler_) {
    register_module("downsample", downsampler_);
  }
}

auto BasicBlockImpl::forward(const torch::Tensor& x) -> torch::Tensor {
  auto out = conv_1_->forward(x);
  out      = batchnorm_1_->forward(out);
  out      = relu_->forward(out);
  out      = conv_2_->forward(out);
  out      = batchnorm_2_->forward(out);

  auto residual = downsampler_ ? downsampler_->forward(x) : x;
  out += residual;
  out = relu_->forward(out);

  return out;
}

auto BasicBlockImpl::zero_init_residual() -> void {
  torch::nn::init::constant_(batchnorm_2_->weight, 0);
}

} // namespace flame::models
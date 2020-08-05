//==--- src/models/select_sls_net.cpp ---------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  select_sls_net.cpp
/// \brief Implementation file for the Select SLS Net.
//
//==------------------------------------------------------------------------==//

#include <flame/models/sls_block.hpp>
#include <flame/models/select_sls_net.hpp>
#include <flame/util/conv.hpp>
#include <flame/util/load.hpp>

namespace flame::models {

SelectSlsNetImpl::SelectSlsNetImpl(
  HeadInput     head_inputs,
  int64_t       head_outputs,
  int64_t       classes,
  FeatureConfig config)
: num_features_{head_outputs},
  classes_{classes},
  head_{make_net_head(head_inputs, head_outputs)},
  stem_{conv_3x3_bn(stem_inputs, stem_outputs, stride_2)},
  features_{make_core_features(config)},
  classifier_{Layer{torch::nn::Linear(num_features_, classes)}} {
  register_module("head", head_);
  register_module("stem", stem_);
  register_module("features", features_);
  register_module("classifier", classifier_);
}

auto SelectSlsNetImpl::make_net_head(
  const HeadInput& head_inputs, int64_t head_outputs) -> Layer {
  return torch::nn::Sequential{
    conv_3x3_bn(head_inputs[0], head_inputs[1], stride_2),
    conv_3x3_bn(head_inputs[1], head_inputs[2], stride_1),
    conv_3x3_bn(head_inputs[2], head_inputs[3], stride_2),
    conv_1x1_bn(head_inputs[3], head_outputs, stride_1)};
}

auto SelectSlsNetImpl::forward(torch::Tensor x) -> torch::Tensor {
  using TensorList = typename SlsBlockImpl::TensorList;

  x = stem_->forward(x);
  x = features_->forward<TensorList>(TensorList{x}).front();
  x = head_->forward(x);
  x = x.mean(3).mean(2);

  if (classes_ != 0) {
    x = classifier_->forward(x);
  }
  return x;
}

auto SelectSlsNetImpl::make_core_features(const FeatureConfig& config)
  -> Layer {
  Layer core;
  for (const auto& block_option : config) {
    core->push_back(SlsBlock(block_option));
  }
  return core;
}

//==--- [models] -----------------------------------------------------------==//

auto select_sls_42(int64_t classes, bool pretrained) -> SelectSlsNet {
  constexpr std::array<int64_t, 4> head_inputs  = {480, 960, 1024, 1024};
  constexpr int64_t                head_outputs = 1280;
  const auto                       config       = SelectSlsNetImpl::config_42();

  auto model = SelectSlsNet{head_inputs, head_outputs, classes, config};
  if (pretrained) {
    load_pretrained(model, "select_sls_42_pretrained.pt");
  }
  return model;
}

auto select_sls_42b(int64_t classes, bool pretrained) -> SelectSlsNet {
  constexpr std::array<int64_t, 4> head_inputs  = {480, 960, 1024, 1280};
  constexpr int64_t                head_outputs = 1024;
  const auto                       config       = SelectSlsNetImpl::config_42();

  auto model = SelectSlsNet{head_inputs, head_outputs, classes, config};
  if (pretrained) {
    load_pretrained(model, "select_sls_42b_pretrained.pt");
  }
  return model;
}

} // namespace flame::models

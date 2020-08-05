//==--- flame/models/select_sls_net.hpp -------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  SelectSlsNet.hpp
/// \brief Header file for Select SLS Network
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_MODELS_SELECT_SLS_NET_HPP
#define FLAME_MODELS_SELECT_SLS_NET_HPP

#include "sls_block.hpp"

namespace flame::models {

/// This class defines the Select SLS Network from:
/// XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB
/// Camera, Mehta et al. 2019 : https://arxiv.org/abs/1907.00837.
class SelectSlsNetImpl : public torch::nn::Module {
 public:
  using Layer         = torch::nn::Sequential;  //!< Type for the net layers.
  using HeadInput     = std::array<int64_t, 4>; //!< Head input type.
  using FeatureConfig = std::vector<SlsBlockOptions>; //!< Feature config type.

  static constexpr int64_t stem_inputs  = 3;  //!< Num inputs for the stem.
  static constexpr int64_t stem_outputs = 32; //!< Num outputs for the stem.

  /// Constructor to create the network.
  /// \param head_inputs  Input sizes for each layer in the head, must be len 4.
  /// \param head_outputs The number of outputs for the head.
  /// \param classes      The number of output classes.
  /// \param config       The configuration for each layer.
  SelectSlsNetImpl(
    HeadInput     head_inputs,
    int64_t       head_outputs,
    int64_t       classes,
    FeatureConfig config);

  /// Feeds the input tensor \p x through the network.
  /// \param x The input tensor to pass through the network.
  auto forward(torch::Tensor x) -> torch::Tensor;

  /// Gets the configuration for Sls 42 versions.
  static auto config_42() -> FeatureConfig {
    // clang-format off
    return FeatureConfig{
      {32 , 0  , 64 , 64 , 2, true },
      {64 , 64 , 64 , 128, 1, false},
      {128, 0  , 144, 144, 2, true },
      {144, 144, 144, 288, 1, false},
      {288, 0  , 304, 304, 2, true },
      {304, 304, 304, 480, 1, false}
    };
    // clang-format on
  }

 private:
  int64_t num_features_ = 0;       //!< Number of features from the head.
  int64_t classes_      = 1000;    //!< Number of classes to output.
  Layer   head_         = nullptr; //!< Head of the network.
  Layer   stem_         = nullptr; //!< Stem of the network.
  Layer   features_     = nullptr; //!< Core of the network.
  Layer   classifier_   = nullptr; //!< Classifier for the network.

  /// Makes the network head.
  /// \param head_inputs  Input sizes for each layer in the head, must be len 4.
  /// \param head_outputs The number of outputs for the head.
  static auto
  make_net_head(const HeadInput& head_inputs, int64_t head_outptus) -> Layer;

  /// Makes the network core.
  /// \param head_inputs  Input sizes for each layer in the head, must be len 4.
  /// \param head_outputs The number of outputs for the head.
  static auto make_core_features(const FeatureConfig& config) -> Layer;
};

/// Wrapper to make the sls block into a torch module.
TORCH_MODULE(SelectSlsNet);

//==--- [select sls net models] --------------------------------------------==//

/// Creates a new SelectSlSNet42 model.
/// \param classes    The number of output classes.
/// \param pretrained If a pretrained model should be returned.
auto select_sls_42(int64_t classes = 1000, bool pretrained = false)
  -> SelectSlsNet;

/// Creates a new SelectSlSNet42 B model.
/// \param classes    The number of output classes.
/// \param pretrained If a pretrained model should be returned.
auto select_sls_42b(int64_t classes = 1000, bool pretrained = false)
  -> SelectSlsNet;

} // namespace flame::models

#endif // FLAME_MODELS_SELECT_SLS_NET_HPP

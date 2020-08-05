//==--- flame/examples/example_utils.hpp ------------------- -*- C++ -*- ---==//
//
//                                  Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  example_utils.hpp
/// \brief This file contains utilities for examples.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_EXAMPLES_EXAMPLE_UTILS_HPP
#define FLAME_EXAMPLES_EXAMPLE_UTILS_HPP

#include <flame/transforms/transforms.hpp>
#include <opencv2/highgui.hpp>

/// The path of this file
constexpr const char* this_file = __FILE__;

static inline auto input_path() -> std::filesystem::path {
  return std::filesystem::path(this_file)
    .parent_path() // strip filename
    .parent_path() // strip examples
    .append("models")
    .append("grace_hopper_517x606.jpg");
}

static inline auto
make_tensor(int64_t resize = 256, int64_t cropsize = 224) -> torch::Tensor {
  using namespace flame::transforms;
  auto transform =
    Transformer()
      .add(Resize(resize))
      .add(CenterCrop(cropsize))
      .add(ConvertImageDType(torch::kFloat32))
      .add(Normalize(Normalize::resnet_mean, Normalize::resnet_stddev))
      .add(ToTensor());
  return transform.make_tensor(cv::imread(input_path())).unsqueeze(0);
}

#endif // FLAME_EXAMPLES_EXAMPLE_UTILS_HPP

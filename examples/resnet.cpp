//==--- flame/examples/resnet.cpp -------------------------- -*- C++ -*- ---==//
//
//                                  Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  resnet.cpp
/// \brief This is an example using the resnet models.
//
//==------------------------------------------------------------------------==//

#include <flame/models/basic_block.hpp>
#include <flame/models/bottleneck.hpp>
#include <flame/models/resnet.hpp>
#include <flame/transforms/transforms.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <array>

/// The path of this file
constexpr const char* this_file = __FILE__;

auto input_path() -> std::filesystem::path {
  return std::filesystem::path(this_file)
    .parent_path() // strip filename
    .parent_path() // strip examples
    .append("models")
    .append("grace_hopper_517x606.jpg");
}

auto make_tensor() -> torch::Tensor {
  using namespace flame::transforms;
  auto transform =
    Transformer()
      .add(Resize(256))
      .add(CenterCrop(224))
      .add(ConvertImageDType(torch::kFloat32))
      .add(Normalize(Normalize::resnet_mean, Normalize::resnet_stddev))
      .add(ToTensor());
  return transform.make_tensor(cv::imread(input_path())).unsqueeze(0);
}

int main() {
  auto resnet = flame::models::resnet50(1000, true);
  auto tensor = make_tensor();
  resnet->eval();

  auto out1 = resnet->forward(tensor);
  auto topk = out1.softmax(1).topk(5);
  std::cout << std::get<0>(topk) << std::endl;
  std::cout << std::get<1>(topk) << std::endl;
}
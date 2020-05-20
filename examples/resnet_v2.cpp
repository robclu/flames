//==--- flame/examples/resnet_v2.cpp ----------------------- -*- C++ -*- ---==//
//
//                                  Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  resnet_v2.cpp
/// \brief This is an example for training and using a resnet v2 model.
//
//==------------------------------------------------------------------------==//

#include <flame/models/basic_block.hpp>
#include <flame/models/bottleneck.hpp>
#include <flame/models/resnet_v2.hpp>
#include <iostream>

int main() {
  auto r1 = flame::models::resnet_v2_50<flame::models::Bottleneck>();

  torch::Tensor tensor = torch::rand({2, 3, 224, 224});
  auto          out1   = r1->forward(tensor);
  std::cout << out1 << std::endl;
}
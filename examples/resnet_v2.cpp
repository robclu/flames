//==--- flame/examples/resnet_v2.cpp ----------------------- -*- C++ -*- ---==//
//
//                              Ripple - Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  resnet_v2.cpp
/// \brief This is an example for training and usign a resnet v2 model.
//
//==------------------------------------------------------------------------==//

#include <ripple/flame/models/basic_block.hpp>
#include <ripple/flame/models/bottleneck.hpp>
#include <iostream>

int main() {
  using namespace ripple::flame;

  models::BasicBlock bb;
  models::Bottleneck bottleneck;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
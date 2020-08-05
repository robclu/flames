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

#include "example_utils.hpp"
#include <flame/models/basic_block.hpp>
#include <flame/models/bottleneck.hpp>
#include <flame/models/resnet.hpp>
#include <flame/transforms/transforms.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <array>

int main() {
  auto resnet = flame::models::resnet50(1000, true);
  auto tensor = make_tensor();
  resnet->eval();

  auto out1 = resnet->forward(tensor);
  auto topk = out1.softmax(1).topk(5);
  std::cout << std::get<0>(topk) << std::endl;
  std::cout << std::get<1>(topk) << std::endl;
}
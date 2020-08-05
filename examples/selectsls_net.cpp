//==--- flame/examples/select_sls_net.cpp ------------------ -*- C++ -*- ---==//
//
//                                  Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  select_sls_net.cpp
/// \brief This is an example using the select_sls_net models.
//
//==------------------------------------------------------------------------==//

#include "example_utils.hpp"
#include <flame/models/select_sls_net.hpp>

int main() {
  auto net    = flame::models::select_sls_42(1000, false);
  auto tensor = make_tensor();
  net->eval();

  auto out1 = net->forward(tensor);
  auto topk = out1.softmax(1).topk(5);
  std::cout << std::get<0>(topk) << std::endl;
  std::cout << std::get<1>(topk) << std::endl;
}
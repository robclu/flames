//==--- flame/util/stack_sequential.hpp -------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  stack_sequntial.hpp
/// \brief Implementation file to enable stacking of torch::nn::Sequential.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_UTIL_STACK_SEQUENTIAL_HPP
#define FLAME_UTIL_STACK_SEQUENTIAL_HPP

#include <torch/torch.h>

namespace flame {

/// Struct when allows stacking of torch::nn::Sequential. Without this, they
/// can't be stacked because the forward implementation in Sequential is
/// templated.
struct StackSequentialImpl final : torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  /// Forward function to pass the tensor \p x through the sequential layer.
  /// \param x The tensor to pass through the network.
  auto forward(torch::Tensor x) -> torch::Tensor {
    return SequentialImpl::forward(x);
  }
};

TORCH_MODULE(StackSequential);

} // namespace flame

#endif // FLAME_UTIL_STACK_SEQUENTIAL_HPP

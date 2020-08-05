//==--- flame/util/load.hpp -------------------------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load.hpp
/// \brief Header file for load utilites.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_UTIL_LOAD_HPP
#define FLAME_UTIL_LOAD_HPP

#include <torch/script.h>
#include <filesystem>

namespace flame {

/// Loads the pretrained parameters from the archive at \p archive_name into
/// the \p model. This uses the environment variable FLAME_MODEL_PATH to
/// get load the archive.
/// \param  model        The model to load the pretrained data from.
/// \param  archive_name The name of the archive to load.
/// \tparam Model        The type of the model.
template <typename Model>
auto load_pretrained(Model& model, std::string archive_name) -> void {
  namespace fs   = std::filesystem;
  auto root_path = std::getenv("FLAME_MODEL_PATH");
  if (!root_path) {
    assert(false && "Model path not loaded: Set $FLAME_MODEL_PATH!");
  }
  auto model_path = fs::path(root_path).append(archive_name);
  torch::load(model, model_path);
}

} // namespace flame

#endif // FLAME_UTIL_LOAD_HPP
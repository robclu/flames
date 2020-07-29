//==--- flame/src/transforms/transforms.cpp ---------------- -*- C++ -*- ---==//
//
//                                Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  transforms.cpp
/// \brief Implementation file for transforms.
//
//==------------------------------------------------------------------------==//

#include <flame/transforms/transforms.hpp>

namespace flame::transforms {

//==--- [resize] -----------------------------------------------------------==//

auto Resize::transform(const cv::Mat& img) const noexcept -> cv::Mat {
  cv::Mat result(width_, height_, img.type());
  cv::resize(img, result, cv::Size(width_, height_), 0, 0, interpolation_);
  return result;
}

auto Resize::transform(cv::Mat& img) const noexcept -> void {
  cv::resize(img, img, cv::Size(width_, height_), 0, 0, interpolation_);
}

//==--- [center crop] ------------------------------------------------------==//

auto get_roi(const cv::Mat& img, int w, int h) -> cv::Rect {
  const int offset_w = (img.cols - w) / 2;
  const int offset_h = (img.rows - h) / 2;
  return cv::Rect(offset_w, offset_h, w, h);
}
auto CenterCrop::transform(const cv::Mat& img) const noexcept -> cv::Mat {
  return img(get_roi(img, width_, height_)).clone();
}

auto CenterCrop::transform(cv::Mat& img) const noexcept -> void {
  img = img(get_roi(img, width_, height_));
}

//==--- [convert d type] ---------------------------------------------------==//

auto get_conversion_props(
  const cv::Mat&  img,
  c10::ScalarType comp_type,
  int&            type,
  double&         alpha) noexcept -> void {
  constexpr double scale = 1.0 / 255.0;
  // clang-format off
    switch (comp_type) {
      case torch::kUInt8  : type = CV_8U ; break;
      case torch::kInt8   : type = CV_8S ; break;
      case torch::kInt16  : type = CV_16S; break;
      case torch::kInt32  :
      case torch::kInt64  : type = CV_32S; break;
      case torch::kFloat16: type = CV_16F; alpha = scale; break;
      case torch::kFloat32: type = CV_32F; alpha = scale; break;
      case torch::kFloat64: type = CV_64F; alpha = scale; break;
      default: break;
    }
  // clang-format on
}

auto ConvertImageDType::transform(const cv::Mat& img) const noexcept
  -> cv::Mat {
  int    type  = img.depth();
  double alpha = 1.0;
  get_conversion_props(img, type_, type, alpha);

  if (type == img.depth()) {
    return img;
  }
  cv::Mat result;
  img.convertTo(result, type, alpha);
  cv::cvtColor(result, result, cv::COLOR_BGR2RGB);
  return result;
}

auto ConvertImageDType::transform(cv::Mat& img) const noexcept -> void {
  int    type  = img.depth();
  double alpha = 1.0;
  get_conversion_props(img, type_, type, alpha);

  if (type == img.depth()) {
    return;
  }
  img.convertTo(img, type, alpha);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
}

//==--- [normalize] --------------------------------------------------------==//

auto Normalize::transform(const cv::Mat& img) const noexcept -> cv::Mat {
  return (img - mean_) / stddev_;
}

auto Normalize::transform(cv::Mat& img) const noexcept -> void {
  img -= mean_;
  img /= stddev_;
}

//==--- [to tensor] --------------------------------------------------------==//

auto ToTensor::create(cv::Mat& img) const noexcept -> torch::Tensor {
  switch (img.type()) {
    case CV_8U:
    case CV_8S:
    case CV_16U:
    case CV_16S:
    case CV_32S: ConvertImageDType(torch::kFloat32).transform(img); break;
    default: break;
  }

  torch::Tensor t = torch::from_blob(
    img.data, {img.rows, img.cols, img.channels()}, torch::kFloat32);
  return t.permute({2, 0, 1});
}

//==--- [transformer] ------------------------------------------------------==//

auto Transformer::make_image(const cv::Mat& img) const noexcept -> cv::Mat {
  auto image = transforms_.front()->transform(img);

  for (size_t i = 1; i < transforms_.size(); ++i) {
    transforms_[i]->transform(image);
  }
  return image;
}

auto Transformer::update_image(cv::Mat& img) const noexcept -> void {
  for (const auto& t : transforms_) {
    t->transform(img);
  }
}

auto Transformer::make_tensor(const cv::Mat& img) const noexcept
  -> torch::Tensor {
  if (auto* t = dynamic_cast<ToTensor*>(transforms_.front().get())) {
    auto image = img;
    return t->create(image);
  }

  ToTensor* tensor_transform = nullptr;
  cv::Mat   image            = transforms_.front()->transform(img);
  for (size_t i = 1; i < transforms_.size(); ++i) {
    auto& t_ptr = transforms_[i];
    if (auto* t = dynamic_cast<ToTensor*>(t_ptr.get())) {
      tensor_transform = t;
      continue;
    }
    t_ptr->transform(image);
  }
  return tensor_transform ? tensor_transform->create(image)
                          : ToTensor().create(image);
}

} // namespace flame::transforms
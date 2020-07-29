//==--- flame/transforms/transforms.hpp -------------------- -*- C++ -*- ---==//
//
//                                  Flame
//
//                      Copyright (c) 2020 Rob Clucas
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  transforms.hpp
/// \brief Header file for image transforms.
//
//==------------------------------------------------------------------------==//

#ifndef FLAME_TRANSFORMS_TRANSFORMS_HPP
#define FLAME_TRANSFORMS_TRANSFORMS_HPP

#include <torch/torch.h>
#include <opencv2/imgproc.hpp>

namespace flame::transforms {

/// Defines an interface for transforming images.
struct Transform {
  /// Default constructor.
  virtual ~Transform() = default;

  /// Applies the transform to the \p img, returning a new image.
  virtual auto transform(const cv::Mat& img) const noexcept -> cv::Mat {
    return img;
  }

  /// Applies the transform to \p img.
  /// \param img The image to apply the transform to.
  virtual auto transform(cv::Mat& img) const noexcept -> void {
    img = transform(static_cast<const cv::Mat&>(img));
  }

  /// Applies a transform to \p img to create a tensor.
  /// \param img The image to transfor.
  virtual auto create(cv::Mat& img) const noexcept -> torch::Tensor {
    return torch::Tensor();
  }
};

/// Class which can be used to define a sequence of transforms on an image.
/// Transforms are applied in the order in which they are added.
class Transformer {
 public:
  /// Adds a transform to the transformer.
  /// \param transform_impl The implementation of the transform.
  /// \tparam TranformImpl  The type of the transform implementation.
  template <typename TransformImpl>
  auto add(TransformImpl&& transform) -> Transformer& {
    transforms_.emplace_back(
      std::make_shared<TransformImpl>(std::forward<TransformImpl>(transform)));
    return *this;
  }

  /// Applies all the transformations, returning a transformed image.
  /// \param img The image to transform.
  auto make_image(const cv::Mat& img) const noexcept -> cv::Mat;

  /// Applies all the transformations, modifying the input image.
  /// \param img The image to transform.
  auto update_image(cv::Mat& img) const noexcept -> void;

  /// Applies all the transformations to the image, returning a torch tensor.
  /// \param img The image to transform.
  auto make_tensor(const cv::Mat& img) const noexcept -> torch::Tensor;

 private:
  // Transforms for the transformer.
  std::vector<std::shared_ptr<Transform>> transforms_ = {};
};

//==--- [transforms] -------------------------------------------------------==//

/// Transform implementation to resize an image.
struct Resize final : public Transform {
  // Default interpolation for resizing.
  static constexpr int default_interp = cv::INTER_LINEAR;

  /// Constructor to set the width and height to resize to.
  /// \param w      The width tor resize to.
  /// \param interp The interpolation for the resizing.
  Resize(int w, int interp = default_interp)
  : width_{w}, height_{w}, interpolation_{interp} {}

  /// Constructor to set the width and height to resize to.
  /// \param w      The width tor resize to.
  /// \param h      The height to resize to.
  /// \param interp The interpolation for the resizing.
  Resize(int w, int h, int interp)
  : width_{w}, height_{h}, interpolation_{interp} {}

  /// Resizes the \p img, returning a new image.
  /// \param img The image to resize.
  auto transform(const cv::Mat& img) const noexcept -> cv::Mat override;

  /// Resizes the image inplace.
  /// \param img The image to apply the transform to.
  auto transform(cv::Mat& img) const noexcept -> void override;

 private:
  int width_         = 0;              //!< Width to resize to.
  int height_        = 0;              //!< Height to resize to;
  int interpolation_ = default_interp; //!< Interpolation for resizing.
};

/// Transform implementation to center crop an image.
struct CenterCrop final : public Transform {
  /// Constructor to set the width and height to crop to to width.
  /// \param w The width tor crop to.
  CenterCrop(int w) : width_{w}, height_{w} {}

  /// Constructor to set the width and height to crop to.
  /// \param w The width tor crop to.
  /// \param h The height to crop to.
  CenterCrop(int w, int h) : width_{w}, height_{h} {}

  /// Center crops the \p img, returning a new image.
  /// \param img The image to center crop.
  auto transform(const cv::Mat& img) const noexcept -> cv::Mat override;

  /// Center crops the \img, setting it to the cropped image.
  /// \param img The image to apply the transform to.
  auto transform(cv::Mat& img) const noexcept -> void override;

 private:
  int width_  = 0; //!< Width to resize to.
  int height_ = 0; //!< Height to resize to;
};

/// Converts the image data type, and ensures that the format of the image is
/// RGB.
struct ConvertImageDType final : public Transform {
  /// Sets the type to convert to.
  /// \param type The type to convert to.
  ConvertImageDType(c10::ScalarType type) : type_{type} {}

  /// Converts the type of the \p img data.
  /// \param img The image to covert the typeof.
  auto transform(const cv::Mat& img) const noexcept -> cv::Mat override;

  /// Converts the type of the \p img data.
  /// \param img The image to covert the typeof.
  auto transform(cv::Mat& img) const noexcept -> void override;

 private:
  c10::ScalarType type_ = torch::kFloat; //!< Type to convert to.
};

/// Normalizes the image with a given mean and standard deviation. This requires
/// that the input is RGB. If not, use ConvertImageDType to convert it.
struct Normalize final : public Transform {
  //==--- [means] ----------------------------------------------------------==//

  /// Mean for resnet.
  static const inline auto resnet_mean = cv::Scalar(0.485, 0.456, 0.406);

  //==--- [standard deviations] --------------------------------------------==//

  /// Std deviation for resnet.
  static const inline auto resnet_stddev = cv::Scalar(0.229, 0.224, 0.225);

  /// Sets the mean and variance for normalization.
  /// \param mean   The mean for normalization.
  /// \param stddev The standard deviation for the normalization.
  Normalize(cv::Scalar mean, cv::Scalar stddev)
  : mean_(std::move(mean)), stddev_(std::move(stddev)) {}

  /// Normalizes the \p img.
  /// \param img The image to normalize.
  auto transform(const cv::Mat& img) const noexcept -> cv::Mat override;

  /// Normalizes the \p img.
  /// \param img The image to normalize.
  auto transform(cv::Mat& img) const noexcept -> void override;

 private:
  cv::Scalar mean_   = cv::Scalar(0); //!< The mean.
  cv::Scalar stddev_ = cv::Scalar(1); //!< Standard deviation.
};

/// Converts an image to a torch Tensor with float data in the range 0.0 - 1.0.
/// The resulting tensor is C x H x W. To create a batch with size 1 for the
/// result then call `unsqueeze(0)` on the result to get a tensor with
/// shape 1 x C x H x W.
///
/// This checks if the image has float data, and if not, applies the
/// ConvertImageDType transform to convert the data to float data and RGB
/// format. If the data is already floating point, then C is ordered as per the
/// input image.
struct ToTensor final : public Transform {
  /// Normalizes the \p img.
  /// \param img The image to normalize.
  auto create(cv::Mat& img) const noexcept -> torch::Tensor override;
};

} // namespace flame::transforms

#endif // FLAME_TRANSFORMS_TRANSFORMS_HPP

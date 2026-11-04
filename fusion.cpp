#include <ATen/ATen.h>
#include <ATen/native/ConvUtils.h>
#include <torch/library.h>
#include <mudnn.h>
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
namespace {
using namespace at;
using namespace at::musa;
void ConfigConv(
    ::musa::dnn::Convolution& c,
    const at::ScalarType& dtype,
    IntArrayRef str,
    IntArrayRef pad,
    IntArrayRef dil,
    int64_t groups) {
  auto sz = str.size();
  if (sz == 2) {
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]), static_cast<int>(pad[1])},
            {static_cast<int>(str[0]), static_cast<int>(str[1])},
            {static_cast<int>(dil[0]), static_cast<int>(dil[1])}),
        "SetNdInfo");
  } else {
    // conv3d
    CHECK_MUDNN_STATUS(
        c.SetNdInfo(
            {static_cast<int>(pad[0]),
             static_cast<int>(pad[1]),
             static_cast<int>(pad[2])},
            {static_cast<int>(str[0]),
             static_cast<int>(str[1]),
             static_cast<int>(str[2])},
            {static_cast<int>(dil[0]),
             static_cast<int>(dil[1]),
             static_cast<int>(dil[2])}),
        "SetNdInfo");
  }
  CHECK_MUDNN_STATUS(
      c.SetComputeMode(at::musa::GetComputeModeFromCtx(dtype)),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(c.SetGroups(groups), "SetGroups");
}

// Adapted from torch_musa commit 42b33842280cbc1f6a4bbb4d1de3c53e0cbb88fc.
Tensor Conv2dBiasActAdd(
    const Tensor& input,
    const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& add_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool act_relu) {
  TORCH_CHECK(
      input.dim() == 4 && weight.dim() == 4,
      "conv2d_bias_act_add only supports 4-D NCHW input/weight");
  TORCH_CHECK(
      weight.scalar_type() == at::ScalarType::Float &&
          input.scalar_type() == at::ScalarType::Float,
      "conv2d_bias_act_add only supports float32 (fp32/tf32) input/weight");
  TORCH_CHECK(
      weight.device() == input.device(),
      "conv2d_bias_act_add: weight and input must be on the same device");
  c10::musa::MUSAGuard device_guard(input.device());

  const auto weight_memory_format = weight.suggest_memory_format();

  Tensor contiguous_input = FormatContiguous(input, weight_memory_format);
  Tensor contiguous_weight = FormatContiguous(weight, weight_memory_format);

  auto input_shape = contiguous_input.sizes();
  auto weight_shape = contiguous_weight.sizes();

  auto output_size = at::native::conv_output_size(
      input_shape, weight_shape, padding, stride, dilation);
  Tensor output =
      at::empty(output_size, contiguous_input.options(), weight_memory_format);

  auto in = CreateMUTensor(contiguous_input);
  auto out = CreateMUTensor(output);
  auto ke = CreateMUTensor(contiguous_weight);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Convolution c;
  ConfigConv(c, input.scalar_type(), stride, padding, dilation, groups);

  ::musa::dnn::Convolution::Algorithm algo;
  c.GetRecommendForwardAlgorithm(h, algo, out, in, ke);

  ::musa::dnn::Convolution::FusedActivationDesc act;
  act.SetMode(
      act_relu
          ? ::musa::dnn::Convolution::FusedActivationDesc::Mode::RELU
          : ::musa::dnn::Convolution::FusedActivationDesc::Mode::IDENTITY);

  // Residual add tensor: must match `output`'s shape & memory format so muDNN
  // can fuse  out = act(conv + bias + add).  empty -> identity (no add).
  Tensor contiguous_add;
  muTensor add;
  if (add_opt.has_value() && add_opt.value().numel() > 0) {
    contiguous_add = FormatContiguous(add_opt.value(), weight_memory_format);
    TORCH_CHECK(
        contiguous_add.sizes() == output.sizes(),
        "conv2d_bias_act_add: add tensor shape ",
        contiguous_add.sizes(),
        " must match conv output shape ",
        output.sizes());
    add = CreateMUTensor(contiguous_add);
  } else {
    add = CreateEmptyMUTensor();
  }

  muTensor bias = (bias_opt.has_value() && bias_opt.value().numel() > 0)
      ? CreateMUTensor(bias_opt.value())
      : CreateEmptyMUTensor();

  CHECK_MUDNN_STATUS(
      c.RunFusion(h, out, in, ke, bias, add, act, algo, InternalMemAlloc),
      "RunFusion(conv2d_bias_act_add)");

  return output;
}

} // namespace
TORCH_LIBRARY_FRAGMENT(musa_fused, m) {
  m.def("conv2d_bias_act_add(Tensor input, Tensor weight, Tensor? bias, "
        "Tensor? add, int[] stride, int[] padding, int[] dilation, int groups, "
        "bool act_relu) -> Tensor");
}
TORCH_LIBRARY_IMPL(musa_fused, PrivateUse1, m) {
  m.impl("conv2d_bias_act_add", TORCH_FN(Conv2dBiasActAdd));
}

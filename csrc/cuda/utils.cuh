#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define AT_DISPATCH_CUSPARSE_TYPES(TYPE, ...)                                  \
  [&] {                                                                        \
    switch (TYPE) {                                                            \
    case torch::ScalarType::Float: {                                           \
      using scalar_t = float;                                                  \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseScsrgemm2_bufferSizeExt;                                     \
      const auto &cusparsecsrgemm2 = cusparseScsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case torch::ScalarType::Double: {                                          \
      using scalar_t = double;                                                 \
      const auto &cusparsecsrgemm2_bufferSizeExt =                             \
          cusparseDcsrgemm2_bufferSizeExt;                                     \
      const auto &cusparsecsrgemm2 = cusparseDcsrgemm2;                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      AT_ERROR("Not implemented for '", toString(TYPE), "'");                  \
    }                                                                          \
  }()


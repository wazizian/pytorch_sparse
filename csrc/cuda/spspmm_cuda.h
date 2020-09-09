#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_cuda(torch::Tensor rowptrA, torch::Tensor colA,
            torch::optional<torch::Tensor> optional_valueA,
            torch::Tensor rowptrB, torch::Tensor colB,
            torch::optional<torch::Tensor> optional_valueB,
            torch::optional<torch::Tensor> optional_rowptrC,
            torch::optional<torch::Tensor> optional_colC,
            torch::optional<torch::Tensor> optional_valueC, int64_t K,
            std::string reduce);

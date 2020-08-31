#include "spspmm_out_cpu.h"

#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
spspmm_out_cpu(torch::Tensor rowptrA, torch::Tensor colA,
               torch::optional<torch::Tensor> optional_valueA,
               torch::Tensor rowptrB, torch::Tensor colB,
               torch::optional<torch::Tensor> optional_valueB,
               torch::Tensor rowptrC, torch::Tensor colC,
               torch::optional<torch::Tensor> optional_valueC, int64_t K,
               std::string reduce) {

  CHECK_CPU(rowptrA);
  CHECK_CPU(colA);
  if (optional_valueA.has_value())
    CHECK_CPU(optional_valueA.value());
  CHECK_CPU(rowptrB);
  CHECK_CPU(colB);
  if (optional_valueB.has_value())
    CHECK_CPU(optional_valueB.value());
  CHECK_CPU(rowptrC);
  CHECK_CPU(colC);
  if (optional_valueC.has_value())
    CHECK_CPU(optional_valueC.value());

  CHECK_INPUT(rowptrA.dim() == 1);
  CHECK_INPUT(colA.dim() == 1);
  if (optional_valueA.has_value()) {
    CHECK_INPUT(optional_valueA.value().dim() == 1);
    CHECK_INPUT(optional_valueA.value().size(0) == colA.size(0));
  }
  CHECK_INPUT(rowptrB.dim() == 1);
  CHECK_INPUT(colB.dim() == 1);
  if (optional_valueB.has_value()) {
    CHECK_INPUT(optional_valueB.value().dim() == 1);
    CHECK_INPUT(optional_valueB.value().size(0) == colB.size(0));
  }
  CHECK_INPUT(rowptrC.dim() == 1);
  CHECK_INPUT(colC.dim() == 1);
  if (optional_valueC.has_value()) {
    CHECK_INPUT(optional_valueC.value().dim() == 1);
    CHECK_INPUT(optional_valueC.value().size(0) == colC.size(0));
  }
  
  if (!optional_valueA.has_value() && !optional_valueB.has_value())
    return std::make_tuple(rowptrC, colC, optional_valueC);

  if (!optional_valueA.has_value() && optional_valueB.has_value())
    optional_valueA =
        torch::ones(colA.numel(), optional_valueB.value().options());

  if (!optional_valueB.has_value() && !optional_valueA.has_value())
    optional_valueB =
        torch::ones(colB.numel(), optional_valueA.value().options());

  if (!optional_valueC.has_value())
      optional_valueC = 
        torch::zeros(colC.numel(), optional_valueA.value().options());

  auto scalar_type = torch::ScalarType::Float;
  if (optional_valueA.has_value())
    scalar_type = optional_valueA.value().scalar_type();

  auto rowptrA_data = rowptrA.data_ptr<int64_t>();
  auto colA_data = colA.data_ptr<int64_t>();
  auto rowptrB_data = rowptrB.data_ptr<int64_t>();
  auto colB_data = colB.data_ptr<int64_t>();
  auto rowptrC_data = rowptrC.data_ptr<int64_t>();
  auto colC_data = colC.data_ptr<int64_t>();


  AT_DISPATCH_ALL_TYPES(scalar_type, "spspmm", [&] {
    scalar_t *valA_data = nullptr, *valB_data = nullptr, *valC_data = nullptr;
    valA_data = optional_valueA.value().data_ptr<scalar_t>();
    valB_data = optional_valueB.value().data_ptr<scalar_t>();
    valC_data = optional_valueC.value().data_ptr<scalar_t>();

    int64_t cA, eB, eC;
    std::vector<scalar_t> tmp_vals(K, 0);
    std::vector<int64_t> cols;
    std::vector<scalar_t> vals;

    for (auto rA = 0; rA < rowptrA.numel() - 1; rA++) {
      for (auto eA = rowptrA_data[rA]; eA < rowptrA_data[rA + 1]; eA++) {
        cA = colA_data[eA];
        eB = rowptrB_data[cA];
        eC = rowptrC_data[rA];
        while (eB < rowptrB_data[cA + 1] && eC < rowptrC_data[rA + 1]) {
          if (colB_data[eB] < colC_data[eC])
            eB++;
          else if (colB_data[eB] > colC_data[eC])
            eC++;
          else {
            valC_data[eC] += valA_data[eA] * valB_data[eB];
            eB++;
            eC++;
          }
        }
      }
    }
  });

  return std::make_tuple(rowptrC, colC, optional_valueC);
}

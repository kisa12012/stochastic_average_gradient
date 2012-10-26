#ifndef LCSGD_LCSGD_H_
#define LCSGD_LCSGD_H_

#include <vector>
#include <random>
#include "tools.h"

namespace lcsgd {
class LCSGD {
 public:
  explicit LCSGD();
  ~LCSGD();

  void LoadData(const data_t& data);
  int Update(int UpdateN);

 private:
  void Initialize();

  void UpdateOnce();
  void UpdateAverageSubgradient(int index);
  void AddSubgradient2AS(int index, double coeff);

  void CalcSubgradient(int index);
  double CalcScore(const datum_t& datum);

  void ResizeAll(int size);

  int updateN_;
  double alpha_; // alpha = (1.0 / (lambda + max \| x \|))
  double lambda_;

  std::mt19937 engine_;

  data_t data_;
  dense_vector_t weight_;
  dense_vector_t average_subgradient_;

  std::vector<sparse_vector_t> subgradients_;
};

} //namespace lcsgd

#endif //LCSGD_LCSGD_H_


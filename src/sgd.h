#ifndef LCSGD_SGD_H_
#define LCSGD_SGD_H_

#include <vector>
#include <random>
#include "tools.h"

namespace lcsgd {
class SGD {
 public:
  explicit SGD();
  ~SGD();

  void LoadData(const data_t& data);
  int Update(int UpdateN);
  double Evaluation();

  void SetAlpha(double a) { alpha_ = a; }
  void SetLambda(double l) { lambda_ = l; }

 private:
  void Initialize();

  void UpdateOnce();
  void UpdateWeight(int index);

  void CalcSubgradient(int index);
  double CalcScore(const datum_t& datum);

  int updateN_;
  double alpha_;
  double lambda_;

  std::mt19937 engine_;

  data_t data_;
  dense_vector_t weight_;
  sparse_vector_t subgradient_;
};

} //namespace lcsgd

#endif //LCSGD_SGD_H_


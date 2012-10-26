#ifndef LCSGD_TOOLS_H_
#define LCSGD_TOOLS_H_

#include <unordered_map>
#include <Eigen/Dense>

namespace lcsgd {
typedef Eigen::VectorXd dense_vector_t;
typedef std::unordered_map<int, double> sparse_vector_t;
typedef int binary_label_t;

typedef std::pair<int, double> feature_t;
typedef std::vector<feature_t> features_t;
struct datum_t {
  features_t features;
  binary_label_t label;
};

struct data_t {
  std::vector<datum_t> examples;
  int data_size;
  int max_feature_id;

  data_t() : data_size(0), max_feature_id(0) {}
};

} // namespace lcsgd

#endif //LCSGD_TOOLS_H_

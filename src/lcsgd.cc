#include "lcsgd.h"

#include <cmath>
#include <iostream>

namespace lcsgd {
LCSGD::LCSGD() : updateN_(0), alpha_(0.1), lambda_(0.1) {}

LCSGD::~LCSGD() {}

void LCSGD::LoadData(const data_t& data) {
  data_ = data;
  Initialize();
}

void LCSGD::Initialize() {
  std::vector<sparse_vector_t>(data_.data_size).swap(subgradients_);
  weight_ = dense_vector_t::Zero(data_.max_feature_id + 1);
  average_subgradient_ = dense_vector_t::Zero(data_.max_feature_id + 1);
}

int LCSGD::Update(int iterN) {
  for (int iter = 0; iter < iterN; ++iter) {
    UpdateOnce();
    ++updateN_;
  }

  return 0;
}

void LCSGD::UpdateOnce() {
  std::uniform_int_distribution<int> dist(1, data_.data_size);
  int index = dist(engine_) - 1;
  UpdateAverageSubgradient(index);
  weight_ = (1.0 - alpha_ * lambda_) * weight_
      - alpha_ / data_.data_size * average_subgradient_;
}

void LCSGD::UpdateAverageSubgradient(int index) {
  AddSubgradient2AS(index, -1);
  CalcSubgradient(index);
  AddSubgradient2AS(index, 1);
}

void LCSGD::AddSubgradient2AS(int index, double coeff) {
  const sparse_vector_t& sv = subgradients_[index];
  for (auto it = sv.begin(); it != sv.end(); ++it) {
    average_subgradient_(it->first) += coeff * it->second;
  }
}

void LCSGD::CalcSubgradient(int index) {
  const datum_t& datum = data_.examples[index];
  double score = CalcScore(datum);

  // std::cout << weight_.transpose() << std::endl;
  std::cout << score << std::endl;

  double coeff = - datum.label * (1.0 / (std::exp(score) + 1.0));
  std::cout << "coeff : " << coeff << std::endl;

  sparse_vector_t& sv = subgradients_[index];
  const features_t& features = datum.features;
  for (auto it = features.begin(); it != features.end(); ++it) {
    sv[it->first] = coeff * it->second;
  }
}

double LCSGD::CalcScore(const datum_t& datum) {
  const features_t& features = datum.features;
  const binary_label_t& label = datum.label;
  
  double score = 0.0;
  for (auto it = features.begin(); it != features.end(); ++it) {
    score += weight_(it->first) * it->second;
  }

  return label * score;
}

} //namespace lcsgd

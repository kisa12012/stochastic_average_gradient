#include "lcsgd.h"

#include <cmath>
#include <iostream>

namespace lcsgd {
LCSGD::LCSGD() : updateN_(0), alpha_(20.0), lambda_(0.00001) {}

LCSGD::~LCSGD() {}

void LCSGD::LoadData(const data_t& data) {
  data_ = data;
  Initialize();
}

void LCSGD::Initialize() {
  std::vector<double>(data_.data_size).swap(latest_coeffs_);
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
  double prev_coeff = latest_coeffs_[index];
  double current_coeff = CalcNewCoeff(index);
  latest_coeffs_[index] = current_coeff;
  AddSubgradient2AS(index, current_coeff - prev_coeff);
}

void LCSGD::AddSubgradient2AS(int index, double coeff) {
  const features_t& fv = data_.examples[index].features;
  for (auto it = fv.begin(); it != fv.end(); ++it) {
    average_subgradient_(it->first) += coeff * it->second;
  }
}

double LCSGD::CalcNewCoeff(int index) {
  const datum_t& datum = data_.examples[index];
  double score = CalcScore(datum);

  double coeff = - datum.label * (1.0 / (std::exp(score) + 1.0));
  return coeff;
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

double LCSGD::Evaluation() {
  double result = 0.0;
  for (auto it = data_.examples.begin(); it != data_.examples.end(); ++it) {
    double score = CalcScore(*it);
    result += std::log(1.0 + std::exp(-score));
  }

  result += weight_.norm();
  return result;
}

} //namespace lcsgd

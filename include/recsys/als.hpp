#pragma once
#include <Eigen/Dense>
#include <vector>
#include "types.hpp"

// Train ALS (baseline) to initialize U,V (used by hybrid ALS -> SGD finetune)
void train_als(const std::vector<Rating>& train,
               int num_users, int num_items, int k,
               float lambda, int iters,
               Eigen::MatrixXf& U, Eigen::MatrixXf& V,
               unsigned seed);

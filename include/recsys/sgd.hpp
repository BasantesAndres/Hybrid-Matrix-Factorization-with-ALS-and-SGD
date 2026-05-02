#pragma once
#include <Eigen/Dense>
#include <vector>
#include "types.hpp"

// SGD training (optionally reinit=false to finetune from existing U,V)
void train_sgd(const std::vector<Rating>& train,
               int num_users, int num_items, int k,
               float lambda, float lr, int epochs,
               Eigen::MatrixXf& U, Eigen::MatrixXf& V,
               bool reinit, unsigned seed);

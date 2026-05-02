#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include "types.hpp"

// Error metrics
std::pair<double,double> compute_rmse_mae(const std::vector<Rating>& data,
                                          const Eigen::MatrixXf& U,
                                          const Eigen::MatrixXf& V);

// Top-N report
struct TopKReport {
    double recall = 0.0;
    double ndcg = 0.0;
    double map = 0.0;
    double coverage = 0.0;
    int users_evaluated = 0;
};

// Evaluate Top-N with negative sampling per user
TopKReport eval_topk(const std::vector<Rating>& test,
                     const std::vector<std::vector<int>>& test_items_by_user,
                     const std::vector<std::vector<int>>& observed_items_by_user,
                     const Eigen::MatrixXf& U, const Eigen::MatrixXf& V,
                     int K, int negatives_per_user, unsigned seed);

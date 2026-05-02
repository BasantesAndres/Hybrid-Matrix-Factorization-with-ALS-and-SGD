#pragma once
#include <vector>
#include <random>
#include "types.hpp"

// Split container
struct Split {
    std::vector<Rating> train, val, test;
};

// User-based split (random per user). Ratios in [0,1], sum <= 1.
Split make_user_splits(const std::vector<Rating>& R,
                       double val_ratio, double test_ratio, unsigned seed);

// For Top-N: items by user from a rating vector
std::vector<std::vector<int>> build_user_items(const std::vector<Rating>& R, int num_users);

// Observed items by user (train+val+test) to avoid sampling seen items
std::vector<std::vector<int>> build_user_observed(const std::vector<Rating>& all, int num_users);

#pragma once
#include <vector>
#include <string>
#include <cstdint>

// Basic rating triple (user, item, rating)
struct Rating {
    int user;
    int item;
    float rating;
};

// Simple stats for convenience
struct DatasetStats {
    int num_users = 0;
    int num_items = 0;
    size_t num_ratings = 0;
    double density = 0.0; // ~ num_ratings / (num_users * num_items)
};

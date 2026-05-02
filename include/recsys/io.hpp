#pragma once
#include <string>
#include <vector>
#include "types.hpp"

// Loaders
std::vector<Rating> load_csv_ratings(const std::string& filename, bool has_header=true);
std::vector<Rating> load_movielens_dat(const std::string& filename); // "::" (ML-1M/10M)

// Infer counts from ratings (max id + 1)
DatasetStats infer_stats(const std::vector<Rating>& R);

// Save single CSV line (append). Creates file if not exists.
void append_csv_line(const std::string& filename, const std::string& header_if_new,
                     const std::string& line);

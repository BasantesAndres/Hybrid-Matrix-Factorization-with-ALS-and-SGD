#pragma once
#include <string>
#include <vector>

std::string now_timestamp();                 // "YYYY-MM-DD_HH-MM-SS"
bool ensure_dir(const std::string& path);   // mkdir -p
bool file_exists(const std::string& path);
std::string join_path(const std::string& a, const std::string& b);

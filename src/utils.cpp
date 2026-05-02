#include "recsys/utils.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

std::string now_timestamp() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d_%02d-%02d-%02d",
                  tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return std::string(buf);
}

bool ensure_dir(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::exists(path, ec)) return true;
    return fs::create_directories(path, ec);
}

bool file_exists(const std::string& path) {
    return std::filesystem::exists(path);
}

std::string join_path(const std::string& a, const std::string& b) {
    namespace fs = std::filesystem;
    return (fs::path(a) / fs::path(b)).string();
}

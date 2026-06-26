#include "recsys/io.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <unordered_map>

std::vector<Rating> load_csv_ratings(const std::string& filename, bool has_header) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    if (has_header && std::getline(in, line)) {
        // consume header
    }
    std::vector<Rating> R;
    R.reserve(100000);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string suser, sitem, srating;
        if (!std::getline(ss, suser, ',')) continue;
        if (!std::getline(ss, sitem, ',')) continue;
        if (!std::getline(ss, srating, ',')) continue;
        Rating r;
        r.user = std::stoi(suser);
        r.item = std::stoi(sitem);
        r.rating = std::stof(srating);
        R.push_back(r);
    }
    return R;
}

static inline bool split4_dat(const std::string& s, std::string& a, std::string& b,
                              std::string& c, std::string& d) {
    // Split "a::b::c::d"
    size_t p1 = s.find("::"); if (p1 == std::string::npos) return false;
    size_t p2 = s.find("::", p1+2); if (p2 == std::string::npos) return false;
    size_t p3 = s.find("::", p2+2); if (p3 == std::string::npos) return false;
    a = s.substr(0, p1);
    b = s.substr(p1+2, p2 - (p1+2));
    c = s.substr(p2+2, p3 - (p2+2));
    d = s.substr(p3+2);
    return true;
}

std::vector<Rating> load_movielens_dat(const std::string& filename) {
    // Parses ML-1M/10M ratings.dat with '::'
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    std::vector<Rating> R; R.reserve(1000000);
    std::string line;
    std::string su, si, sr, ts;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (!split4_dat(line, su, si, sr, ts)) continue;
        Rating r;
        r.user   = std::stoi(su) - 1;  // to 0-based
        r.item   = std::stoi(si) - 1;  // to 0-based
        r.rating = std::stof(sr);
        R.push_back(r);
    }
    return R;
}

// NUEVA: Para ML-100K (u.data con tabs/espacios)
std::vector<Rating> load_movielens_100k(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open file: " + filename);
    std::vector<Rating> R;
    R.reserve(100000);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string su, si, sr, st;
        if (!(ss >> su >> si >> sr >> st)) continue; // formato: user item rating timestamp
        Rating r;
        r.user   = std::stoi(su) - 1;
        r.item   = std::stoi(si) - 1;
        r.rating = std::stof(sr);
        R.push_back(r);
    }
    return R;
}

DatasetStats infer_stats(const std::vector<Rating>& R) {
    DatasetStats st;
    st.num_ratings = R.size();
    int maxu = -1, maxi = -1;
    for (auto& r : R) {
        if (r.user > maxu) maxu = r.user;
        if (r.item > maxi) maxi = r.item;
    }
    st.num_users = maxu + 1;
    st.num_items = maxi + 1;
    if ((long long)st.num_users * (long long)st.num_items > 0) {
        st.density = double(st.num_ratings) /
                     double((long long)st.num_users * (long long)st.num_items);
    }
    return st;
}

void append_csv_line(const std::string& filename, const std::string& header_if_new,
                     const std::string& line) {
    bool new_file = !std::filesystem::exists(filename);
    std::ofstream out(filename, std::ios::app);
    if (!out) throw std::runtime_error("Cannot write file: " + filename);
    if (new_file && !header_if_new.empty()) out << header_if_new << "\n";
    out << line << "\n";
}
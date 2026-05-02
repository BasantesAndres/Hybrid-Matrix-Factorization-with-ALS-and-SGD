#include "recsys/split.hpp"
#include <unordered_map>
#include <algorithm>

Split make_user_splits(const std::vector<Rating>& R,
                       double val_ratio, double test_ratio, unsigned seed) {
    std::unordered_map<int, std::vector<Rating>> by_user;
    by_user.reserve(R.size()/5 + 1);
    int maxu = -1;
    for (auto& r : R) {
        by_user[r.user].push_back(r);
        if (r.user > maxu) maxu = r.user;
    }
    std::mt19937 rng(seed);
    Split sp;

    for (auto& kv : by_user) {
        auto& vec = kv.second;
        std::shuffle(vec.begin(), vec.end(), rng);
        size_t n = vec.size();
        size_t n_val  = static_cast<size_t>(n * val_ratio);
        size_t n_test = static_cast<size_t>(n * test_ratio);
        size_t n_train = (n > n_val + n_test) ? (n - n_val - n_test) : 0;

        sp.train.insert(sp.train.end(), vec.begin(), vec.begin()+n_train);
        sp.val.insert(sp.val.end(), vec.begin()+n_train, vec.begin()+n_train+n_val);
        sp.test.insert(sp.test.end(), vec.begin()+n_train+n_val, vec.end());
    }
    return sp;
}

std::vector<std::vector<int>> build_user_items(const std::vector<Rating>& R, int num_users) {
    std::vector<std::vector<int>> items(num_users);
    for (auto& r : R) items[r.user].push_back(r.item);
    return items;
}

std::vector<std::vector<int>> build_user_observed(const std::vector<Rating>& all, int num_users) {
    std::vector<std::vector<int>> obs(num_users);
    // Use vector and deduplicate later (cheap)
    for (auto& r : all) obs[r.user].push_back(r.item);
    // Deduplicate each user list
    for (auto& v : obs) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return obs;
}

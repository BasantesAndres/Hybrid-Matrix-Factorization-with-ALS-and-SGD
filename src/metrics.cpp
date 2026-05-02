#include "recsys/metrics.hpp"
#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>

std::pair<double,double> compute_rmse_mae(const std::vector<Rating>& data,
                                          const Eigen::MatrixXf& U,
                                          const Eigen::MatrixXf& V) {
    if (data.empty()) return {0.0, 0.0};
    double se = 0.0, ae = 0.0;
    for (auto& r : data) {
        float pred = U.row(r.user).dot(V.row(r.item).transpose());
        double diff = (double)r.rating - (double)pred;
        se += diff * diff;
        ae += std::abs(diff);
    }
    double rmse = std::sqrt(se / (double)data.size());
    double mae  = ae / (double)data.size();
    return {rmse, mae};
}

static std::vector<int> sample_negatives_one_user(
        int user, int num_items,
        const std::vector<int>& observed_sorted, // sorted ascending
        int n_neg, std::mt19937& rng) {
    // Rejection sampling with hash set to avoid duplicates.
    // Works well for n_neg up to ~1000. observed_sorted is usually small vs num_items.
    std::uniform_int_distribution<int> uni(0, num_items - 1);
    std::unordered_set<int> used;
    used.reserve(n_neg * 2 + 8);

    // Add observed into "used" to avoid sampling seen items
    for (int it : observed_sorted) used.insert(it);

    std::vector<int> neg;
    neg.reserve(n_neg);
    while ((int)neg.size() < n_neg) {
        int cand = uni(rng);
        if (used.find(cand) == used.end()) {
            neg.push_back(cand);
            used.insert(cand);
        }
    }
    return neg;
}

TopKReport eval_topk(const std::vector<Rating>& test,
                     const std::vector<std::vector<int>>& test_items_by_user,
                     const std::vector<std::vector<int>>& observed_items_by_user,
                     const Eigen::MatrixXf& U, const Eigen::MatrixXf& V,
                     int K, int negatives_per_user, unsigned seed) {
    const int num_users = U.rows();
    const int num_items = V.rows();

    // Coverage: which items appear in any user's Top-K
    std::vector<char> covered(num_items, 0);

    std::mt19937 rng(seed);

    double sum_recall = 0.0, sum_ndcg = 0.0, sum_map = 0.0;
    int users_eval = 0;

    for (int u = 0; u < num_users; ++u) {
        const auto& positives = test_items_by_user[u];
        if (positives.empty()) continue;

        // Observed (train+val+test) to avoid sampling seen items
        std::vector<int> observed = observed_items_by_user[u]; // already sorted
        // Negatives
        auto negatives = sample_negatives_one_user(u, num_items, observed, negatives_per_user, rng);

        // Build candidate set = positives union negatives
        std::vector<int> candidates = positives;
        candidates.insert(candidates.end(), negatives.begin(), negatives.end());

        // Score candidates
        std::vector<std::pair<float,int>> scored;
        scored.reserve(candidates.size());
        for (int it : candidates) {
            float s = U.row(u).dot(V.row(it).transpose());
            scored.push_back({s, it});
        }
        // Top-K by score desc
        if ((int)scored.size() > K) {
            std::partial_sort(scored.begin(), scored.begin()+K, scored.end(),
                              [](auto& a, auto& b){ return a.first > b.first; });
            scored.resize(K);
        } else {
            std::sort(scored.begin(), scored.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
        }

        // Prepare a fast membership test for positives
        std::unordered_set<int> pos_set(positives.begin(), positives.end());

        // Compute Recall@K
        int hits = 0;
        for (auto& pr : scored) if (pos_set.count(pr.second)) ++hits;
        double denom = std::min((int)K, (int)positives.size());
        double recall = (denom > 0) ? (double)hits / denom : 0.0;

        // Compute NDCG@K (binary relevance)
        auto log2i = [](int i){ return std::log2((double)(i+2)); }; // positions start at 0
        double dcg = 0.0;
        for (int rank = 0; rank < (int)scored.size(); ++rank) {
            int item_id = scored[rank].second;
            if (pos_set.count(item_id)) {
                dcg += 1.0 / log2i(rank);
            }
        }
        // Ideal DCG with all positives at top
        int ideal_hits = std::min((int)positives.size(), K);
        double idcg = 0.0;
        for (int rank = 0; rank < ideal_hits; ++rank) {
            idcg += 1.0 / log2i(rank);
        }
        double ndcg = (idcg > 0.0) ? (dcg / idcg) : 0.0;

        // MAP@K (binary relevance)
        double ap = 0.0;
        int found = 0;
        for (int rank = 0; rank < (int)scored.size(); ++rank) {
            int item_id = scored[rank].second;
            if (pos_set.count(item_id)) {
                ++found;
                double prec_at_r = (double)found / (double)(rank+1);
                ap += prec_at_r;
            }
        }
        double mapk = (denom > 0) ? (ap / denom) : 0.0;

        // Coverage: mark items in Top-K
        for (auto& pr : scored) covered[pr.second] = 1;

        sum_recall += recall;
        sum_ndcg   += ndcg;
        sum_map    += mapk;
        ++users_eval;
    }

    TopKReport rep;
    rep.users_evaluated = users_eval;
    if (users_eval > 0) {
        rep.recall = sum_recall / users_eval;
        rep.ndcg   = sum_ndcg   / users_eval;
        rep.map    = sum_map    / users_eval;
    } else {
        rep.recall = rep.ndcg = rep.map = 0.0;
    }
    // Coverage fraction
    int covered_cnt = 0;
    for (char c : covered) if (c) ++covered_cnt;
    rep.coverage = (num_items > 0) ? (double)covered_cnt / (double)num_items : 0.0;
    return rep;
}

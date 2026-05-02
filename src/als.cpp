#include "recsys/als.hpp"
#include <random>

// Simple ALS with normal equations and LDLT
// A_u = sum_{j in R(u)} v_j v_j^T + lambda I
// b_u = sum_{j in R(u)} r_uj v_j
// Solve A_u x = b_u, then set U[u] = x
// Symmetric for items.

void train_als(const std::vector<Rating>& train,
               int num_users, int num_items, int k,
               float lambda, int iters,
               Eigen::MatrixXf& U, Eigen::MatrixXf& V,
               unsigned seed) {
    // Build incidence lists once
    std::vector<std::vector<std::pair<int,float>>> user2items(num_users);
    std::vector<std::vector<std::pair<int,float>>> item2users(num_items);
    user2items.reserve(num_users);
    item2users.reserve(num_items);
    for (auto& r : train) {
        user2items[r.user].push_back({r.item, r.rating});
        item2users[r.item].push_back({r.user, r.rating});
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.1f);

    U = Eigen::MatrixXf::Zero(num_users, k);
    V = Eigen::MatrixXf::Zero(num_items, k);
    for (int u = 0; u < num_users; ++u)
        for (int f = 0; f < k; ++f) U(u,f) = nd(rng);
    for (int i = 0; i < num_items; ++i)
        for (int f = 0; f < k; ++f) V(i,f) = nd(rng);

    Eigen::MatrixXf I = lambda * Eigen::MatrixXf::Identity(k, k);

    for (int it = 1; it <= iters; ++it) {
        // Update users
        for (int u = 0; u < num_users; ++u) {
            auto& vec = user2items[u];
            if (vec.empty()) continue;
            Eigen::MatrixXf A = I;
            Eigen::VectorXf b = Eigen::VectorXf::Zero(k);
            for (auto& p : vec) {
                int j = p.first; float ruj = p.second;
                Eigen::RowVectorXf vj = V.row(j);
                A.noalias() += vj.transpose() * vj;     // kxk
                b.noalias() += ruj * vj.transpose();    // k
            }
            Eigen::VectorXf x = A.ldlt().solve(b);
            U.row(u) = x.transpose();
        }
        // Update items
        for (int j = 0; j < num_items; ++j) {
            auto& vec = item2users[j];
            if (vec.empty()) continue;
            Eigen::MatrixXf A = I;
            Eigen::VectorXf b = Eigen::VectorXf::Zero(k);
            for (auto& p : vec) {
                int u = p.first; float ruj = p.second;
                Eigen::RowVectorXf uu = U.row(u);
                A.noalias() += uu.transpose() * uu;
                b.noalias() += ruj * uu.transpose();
            }
            Eigen::VectorXf x = A.ldlt().solve(b);
            V.row(j) = x.transpose();
        }
        // (Optional) you can print progress outside, in main(), after RMSE on val
    }
}

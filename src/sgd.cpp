#include "recsys/sgd.hpp"
#include <random>
#include <algorithm>

// Basic SGD with L2 regularization
// rating r ~ U[u] . V[i]
// Update:
//   e = r - dot(Uu, Vi)
//   Uu += lr * (e * Vi - lambda * Uu)
//   Vi += lr * (e * Uu_old - lambda * Vi)

void train_sgd(const std::vector<Rating>& train,
               int num_users, int num_items, int k,
               float lambda, float lr, int epochs,
               Eigen::MatrixXf& U, Eigen::MatrixXf& V,
               bool reinit, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 0.1f);

    if (reinit || U.rows() != num_users || U.cols() != k) {
        U = Eigen::MatrixXf::Zero(num_users, k);
        for (int u = 0; u < num_users; ++u)
            for (int f = 0; f < k; ++f) U(u,f) = nd(rng);
    }
    if (reinit || V.rows() != num_items || V.cols() != k) {
        V = Eigen::MatrixXf::Zero(num_items, k);
        for (int i = 0; i < num_items; ++i)
            for (int f = 0; f < k; ++f) V(i,f) = nd(rng);
    }

    std::vector<Rating> data = train;

    for (int ep = 1; ep <= epochs; ++ep) {
        std::shuffle(data.begin(), data.end(), rng);

        for (const auto& r : data) {
            int u = r.user, i = r.item;
            float pred = U.row(u).dot(V.row(i).transpose());
            float e = r.rating - pred;

            // keep a copy for Vi update
            Eigen::RowVectorXf Uu = U.row(u);
            U.row(u) += lr * (e * V.row(i) - lambda * U.row(u));
            V.row(i) += lr * (e * Uu        - lambda * V.row(i));
        }
        // You can decay lr if you wish (not required)
        // lr *= 0.98f;
    }
}

#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <Eigen/Dense>

#include "recsys/types.hpp"
#include "recsys/io.hpp"
#include "recsys/split.hpp"
#include "recsys/als.hpp"
#include "recsys/sgd.hpp"
#include "recsys/metrics.hpp"
#include "recsys/utils.hpp"

// Simple CLI parsing
struct Args {
    std::string data_path = "";
    std::string format = "csv";    // csv | ml1m | ml10m (ml1m/10m both "::")
    std::string model  = "hybrid"; // als | hybrid
    int k = 50;
    float lambda = 0.2f;
    int iters = 6;                 // ALS iters
    int epochs_sgd = 2;            // finetune epochs
    float lr = 0.01f;
    int K_top = 10;
    int neg_per_user = 500;
    unsigned seed = 42;
    double val_ratio = 0.1;
    double test_ratio = 0.1;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1; i<argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int i){ if (i+1>=argc) throw std::runtime_error("Missing value for " + s); return std::string(argv[i+1]); };
        if (s=="--data") a.data_path = need(i++);
        else if (s=="--format") a.format = need(i++);
        else if (s=="--model") a.model = need(i++);
        else if (s=="--k") a.k = std::stoi(need(i++));
        else if (s=="--lambda") a.lambda = std::stof(need(i++));
        else if (s=="--iters") a.iters = std::stoi(need(i++));
        else if (s=="--epochs_sgd") a.epochs_sgd = std::stoi(need(i++));
        else if (s=="--lr") a.lr = std::stof(need(i++));
        else if (s=="--K_top") a.K_top = std::stoi(need(i++));
        else if (s=="--neg_per_user") a.neg_per_user = std::stoi(need(i++));
        else if (s=="--seed") a.seed = (unsigned)std::stoul(need(i++));
        else if (s=="--val_ratio") a.val_ratio = std::stod(need(i++));
        else if (s=="--test_ratio") a.test_ratio = std::stod(need(i++));
        else {
            std::cerr << "Unknown arg: " << s << "\n";
        }
    }
    if (a.data_path.empty()) throw std::runtime_error("--data is required");
    return a;
}

int main(int argc, char** argv) {
    std::cout << std::fixed << std::setprecision(4);
    try {
        Args args = parse_args(argc, argv);

        // Load ratings
        std::vector<Rating> ratings;
        if (args.format == "csv") {
            ratings = load_csv_ratings(args.data_path, /*has_header=*/true);
        } else if (args.format == "ml1m" || args.format=="ml10m") {
            ratings = load_movielens_dat(args.data_path);
        } else {
            throw std::runtime_error("Unsupported --format (use csv|ml1m|ml10m)");
        }
        auto stats = infer_stats(ratings);

        std::cout << "==============================================\n";
        std::cout << "Recommender — seed=" << args.seed << "\n";
        std::cout << "Users=" << stats.num_users
                  << "  Items=" << stats.num_items
                  << "  Ratings=" << ratings.size()
                  << "  Density=" << 100.0*stats.density << "%\n";
        std::cout << "k=" << args.k << "  lambda=" << args.lambda
                  << "  K_top=" << args.K_top
                  << "  neg_per_user=" << args.neg_per_user << "\n";
        std::cout << "Split per-user: val=" << args.val_ratio*100 << "%  test=" << args.test_ratio*100 << "%\n";
        std::cout << "==============================================\n";

        // Split
        auto S = make_user_splits(ratings, args.val_ratio, args.test_ratio, args.seed);
        // Build helpers for TopN
        auto test_items_by_user = build_user_items(S.test, stats.num_users);
        auto observed_by_user   = build_user_observed(ratings, stats.num_users);

        // Prepare results directory
        std::string root = "results";
        std::string dataset_dir = (args.format=="csv" ? "csv" : (args.format=="ml1m" ? "ml1m" : "ml10m"));
        std::string stamp = now_timestamp();
        std::string run_dir = join_path(join_path(root, dataset_dir), stamp);
        ensure_dir(run_dir);

        // Files
        std::string err_csv  = join_path(run_dir, "error_metrics.csv");
        std::string topn_csv = join_path(run_dir, "topn_metrics.csv");

        // ---- ALS ----
        Eigen::MatrixXf U, V;
        train_als(S.train, stats.num_users, stats.num_items, args.k,
                  args.lambda, args.iters, U, V, args.seed);

        auto val_err = compute_rmse_mae(S.val, U, V);
        std::cout << "[ALS][VAL] rmse=" << val_err.first << "  mae=" << val_err.second << "\n";

        auto test_err_als = compute_rmse_mae(S.test, U, V);
        std::cout << "[ALS][TEST] rmse=" << test_err_als.first
                  << "  mae=" << test_err_als.second << "\n";
        append_csv_line(err_csv,
                        "model,split,rmse,mae",
                        "ALS,TEST," + std::to_string(test_err_als.first) + "," + std::to_string(test_err_als.second));

        auto rep_als = eval_topk(S.test, test_items_by_user, observed_by_user,
                                 U, V, args.K_top, args.neg_per_user, args.seed);
        std::cout << "[ALS][TEST][TopN@" << args.K_top << "] "
                  << "recall=" << rep_als.recall
                  << "  ndcg="   << rep_als.ndcg
                  << "  map="    << rep_als.map
                  << "  coverage=" << rep_als.coverage*100 << "%  "
                  << "users=" << rep_als.users_evaluated << "\n";
        append_csv_line(topn_csv,
            "model,split,K,recall,ndcg,map,coverage,users_evaluated,neg_per_user",
            "ALS,TEST," + std::to_string(args.K_top) + "," +
            std::to_string(rep_als.recall) + "," + std::to_string(rep_als.ndcg) + "," +
            std::to_string(rep_als.map) + "," + std::to_string(rep_als.coverage) + "," +
            std::to_string(rep_als.users_evaluated) + "," + std::to_string(args.neg_per_user));

        if (args.model == "hybrid") {
            // ---- HYBRID: ALS -> SGD finetune (no reinit) ----
            train_sgd(S.train, stats.num_users, stats.num_items, args.k,
                      args.lambda, args.lr, args.epochs_sgd,
                      U, V, /*reinit=*/false, args.seed);

            auto test_err_h = compute_rmse_mae(S.test, U, V);
            std::cout << "[HYBRID][TEST] rmse=" << test_err_h.first
                      << "  mae=" << test_err_h.second << "\n";
            append_csv_line(err_csv, "", // header already written
                            "HYBRID,TEST," + std::to_string(test_err_h.first) + "," + std::to_string(test_err_h.second));

            auto rep_h = eval_topk(S.test, test_items_by_user, observed_by_user,
                                   U, V, args.K_top, args.neg_per_user, args.seed);
            std::cout << "[HYBRID][TEST][TopN@" << args.K_top << "] "
                      << "recall=" << rep_h.recall
                      << "  ndcg="   << rep_h.ndcg
                      << "  map="    << rep_h.map
                      << "  coverage=" << rep_h.coverage*100 << "%  "
                      << "users=" << rep_h.users_evaluated << "\n";
            append_csv_line(topn_csv, "",
                "HYBRID,TEST," + std::to_string(args.K_top) + "," +
                std::to_string(rep_h.recall) + "," + std::to_string(rep_h.ndcg) + "," +
                std::to_string(rep_h.map) + "," + std::to_string(rep_h.coverage) + "," +
                std::to_string(rep_h.users_evaluated) + "," + std::to_string(args.neg_per_user));
        }

        std::cout << "[SAVE] " << err_csv  << "\n";
        std::cout << "[SAVE] " << topn_csv << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FATAL: " << ex.what() << "\n";
        return 1;
    }
}

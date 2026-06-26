#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <chrono>
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
    std::string format = "csv";    // csv | ml100k | ml1m | ml10m
    std::string model  = "hybrid"; // als | sgd | hybrid
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
        } else if (args.format == "ml1m" || args.format == "ml10m") {
            ratings = load_movielens_dat(args.data_path);
        } else if (args.format == "ml100k") {
            ratings = load_movielens_100k(args.data_path);
        } else {
            throw std::runtime_error("Unsupported --format (use csv|ml100k|ml1m|ml10m)");
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
        std::string dataset_dir = (args.format=="csv" ? "csv" : (args.format=="ml100k" ? "ml100k" : (args.format=="ml1m" ? "ml1m" : "ml10m")));
        std::string stamp = now_timestamp();
        std::string run_dir = join_path(join_path(root, dataset_dir), stamp);
        ensure_dir(run_dir);

        // Files
        std::string err_csv  = join_path(run_dir, "error_metrics.csv");
        std::string topn_csv = join_path(run_dir, "topn_metrics.csv");
        std::string time_csv = join_path(run_dir, "timing.csv");

        // ---- Variables para tiempos ----
        double time_als = 0.0;
        double time_sgd = 0.0;
        double time_eval = 0.0;
        
        // ---- Inicializar U,V (se usarán para ALS y Hybrid) ----
        Eigen::MatrixXf U, V;

        // ============================================================
        // 1. ENTRENAMIENTO (dependiendo del modelo)
        // ============================================================
        if (args.model == "als") {
            // ----- ALS ONLY -----
            auto start = std::chrono::high_resolution_clock::now();
            train_als(S.train, stats.num_users, stats.num_items, args.k,
                      args.lambda, args.iters, U, V, args.seed);
            auto end = std::chrono::high_resolution_clock::now();
            time_als = std::chrono::duration<double>(end - start).count();
            time_sgd = 0.0;

        } else if (args.model == "sgd") {
            // ----- SGD ONLY (from random init) -----
            auto start = std::chrono::high_resolution_clock::now();
            train_sgd(S.train, stats.num_users, stats.num_items, args.k,
                      args.lambda, args.lr, args.epochs_sgd,
                      U, V, /*reinit=*/true, args.seed);
            auto end = std::chrono::high_resolution_clock::now();
            time_sgd = std::chrono::duration<double>(end - start).count();
            time_als = 0.0;

        } else if (args.model == "hybrid") {
            // ----- HYBRID: ALS + SGD finetune -----
            auto start_als = std::chrono::high_resolution_clock::now();
            train_als(S.train, stats.num_users, stats.num_items, args.k,
                      args.lambda, args.iters, U, V, args.seed);
            auto end_als = std::chrono::high_resolution_clock::now();
            time_als = std::chrono::duration<double>(end_als - start_als).count();

            auto start_sgd = std::chrono::high_resolution_clock::now();
            train_sgd(S.train, stats.num_users, stats.num_items, args.k,
                      args.lambda, args.lr, args.epochs_sgd,
                      U, V, /*reinit=*/false, args.seed);
            auto end_sgd = std::chrono::high_resolution_clock::now();
            time_sgd = std::chrono::duration<double>(end_sgd - start_sgd).count();
        }

        // ============================================================
        // 2. EVALUACIÓN (se mide por separado)
        // ============================================================
        auto start_eval = std::chrono::high_resolution_clock::now();

        auto test_err = compute_rmse_mae(S.test, U, V);
        std::cout << "[MODEL][TEST] rmse=" << test_err.first
                  << "  mae=" << test_err.second << "\n";
        append_csv_line(err_csv,
                        "model,split,rmse,mae",
                        args.model + ",TEST," + std::to_string(test_err.first) + "," + std::to_string(test_err.second));

        auto rep = eval_topk(S.test, test_items_by_user, observed_by_user,
                             U, V, args.K_top, args.neg_per_user, args.seed);
        std::cout << "[MODEL][TEST][TopN@" << args.K_top << "] "
                  << "recall=" << rep.recall
                  << "  ndcg="   << rep.ndcg
                  << "  map="    << rep.map
                  << "  coverage=" << rep.coverage*100 << "%  "
                  << "users=" << rep.users_evaluated << "\n";
        append_csv_line(topn_csv,
            "model,split,K,recall,ndcg,map,coverage,users_evaluated,neg_per_user",
            args.model + ",TEST," + std::to_string(args.K_top) + "," +
            std::to_string(rep.recall) + "," + std::to_string(rep.ndcg) + "," +
            std::to_string(rep.map) + "," + std::to_string(rep.coverage) + "," +
            std::to_string(rep.users_evaluated) + "," + std::to_string(args.neg_per_user));

        auto end_eval = std::chrono::high_resolution_clock::now();
        time_eval = std::chrono::duration<double>(end_eval - start_eval).count();

        // ============================================================
        // 3. GUARDAR TIEMPOS (timing.csv)
        // ============================================================
        double time_total = time_als + time_sgd + time_eval;
        std::cout << "[TIMING] ALS=" << time_als << "s, SGD=" << time_sgd << "s, Eval=" << time_eval << "s, Total=" << time_total << "s\n";
        
        std::string header_time = "model,dataset,als_time,sgd_time,eval_time,total_time,seed,k,iters,epochs_sgd";
        std::string line_time = args.model + "," + args.format + "," +
                                std::to_string(time_als) + "," + std::to_string(time_sgd) + "," +
                                std::to_string(time_eval) + "," + std::to_string(time_total) + "," +
                                std::to_string(args.seed) + "," + std::to_string(args.k) + "," +
                                std::to_string(args.iters) + "," + std::to_string(args.epochs_sgd);
        append_csv_line(time_csv, header_time, line_time);

        std::cout << "[SAVE] " << err_csv  << "\n";
        std::cout << "[SAVE] " << topn_csv << "\n";
        std::cout << "[SAVE] " << time_csv << "\n";
        std::cout << "==============================================\n";
        std::cout << "RUN FINISHED SUCCESSFULLY.\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FATAL: " << ex.what() << "\n";
        return 1;
    }
}
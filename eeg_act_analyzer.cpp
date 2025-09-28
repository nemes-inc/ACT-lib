#include "ACT.h"
#include "ACT_CPU.h"
#include "ACT_Accelerate.h"
#include "ACT_MLX.h"
#include "ACT_MLX_MT.h"
#include "muse_osc_receiver.h"
#include "ring_buffer.h"
#include "logging/ndjson_logger.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <memory>
#include <cmath>
#include "linenoise.h"
#include <chrono>
#include <atomic>
#include <array>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <ctime>

// --- Global State ---
static std::vector<std::vector<double>> csv_data;
static std::vector<double> selected_signal;
static std::vector<std::string> csv_headers;
static int selected_column_index = -1;
static int start_sample = 0;
static ACT::ParameterRanges param_ranges;
// Backends
enum class BackendSel { ACT, CPU, ACCEL, MLX };
static BackendSel backend_sel = BackendSel::CPU; // default
static std::unique_ptr<ACT> act_legacy;           // Legacy ACT
static std::unique_ptr<ACT_CPU> act_cpu;          // ACT_CPU (double)
static std::unique_ptr<ACT_Accelerate> act_accel; // ACT_Accelerate (double)
static std::unique_ptr<ACT_Accelerate_f> act_accel_f; // Float32 path (MLX or Accelerate_f)
static bool coarse_only = false;                  // skip BFGS if true
static double sampling_frequency = 256.0; // Default Muse sampling rate
static std::string current_filename;

// --- Live OSC/Streaming State ---
static std::unique_ptr<MuseOSCReceiver> muse_rx;
static std::ofstream live_csv;
static std::mutex live_csv_mu;
static std::atomic<bool> live_running{false};
static int live_port = 0;
static std::string live_csv_path;
static std::string live_json_base;
static int live_window = 0; // must equal dictionary length
static int live_hop = 64;
static int live_order = 10;
static bool live_refine = true;
static int live_feedback_top = 1;
static bool live_quiet = true; // suppress live per-window console output by default
static size_t live_feedback_interval_ms = 1000; // throttle live console feedback
enum class LiveBackend { AUTO, MLX, CPU };
static LiveBackend live_backend = LiveBackend::AUTO;

// Ring buffers per channel
static RingBuffer<float> rb_tp9, rb_af7, rb_af8, rb_tp10;
static size_t rb_capacity = 0;

// Quality indicators (sample-and-hold)
static std::array<std::atomic<int>,4> hs_last = {1,1,1,1};
static std::atomic<int> blink_pending{0};
static std::atomic<int> jaw_pending{0};

// Analysis thread
static std::thread analysis_th;
static std::atomic<int64_t> sample_counter{0};
static int64_t last_processed = 0;
static std::string live_session_id;
static const char* kChannels[4] = {"TP9","AF7","AF8","TP10"};

// --- Forward Declarations ---
void handle_load_csv(const std::vector<std::string>& args);
void handle_select_data(const std::vector<std::string>& args);
void handle_set_params(const std::vector<std::string>& args);
void handle_create_dictionary(const std::vector<std::string>& args);
void handle_save_dictionary(const std::vector<std::string>& args);
void handle_load_dictionary(const std::vector<std::string>& args);
void handle_analyze(const std::vector<std::string>& args);
void handle_analyze_samples(const std::vector<std::string>& args);
void print_analysis_results(const ACT::TransformResult& result, int time_offset = 0);
void print_signal_stats(const std::vector<double>& signal, const std::string& title);
void print_estimated_size();
void print_help();
void print_current_params();
void print_dict_summary();
void save_analysis_to_json(const std::string& filename, const ACT::TransformResult& result, 
                          int num_chirplets, double residual_threshold, bool is_single = true, 
                          int window_start = 0, int end_sample = 0, int overlap = 0, int num_chirps = 0);
// Combined multi-window JSON saver
void save_multiwindow_combined_to_json(
    const std::string& filename,
    const std::vector<ACT::TransformResult>& results,
    const std::vector<int>& window_starts,
    int end_sample,
    int overlap,
    int num_chirps);

// Live mode handlers
void handle_muse_start(const std::vector<std::string>& args);
void handle_muse_stop(const std::vector<std::string>& args);
void handle_muse_status(const std::vector<std::string>& args);
static std::string iso8601_now();
static void live_analysis_loop();

// --- Main Application Loop ---
int main() {
    const char* history_file = ".eeg_analyzer_history.txt";
    linenoiseHistoryLoad(history_file);
    linenoiseHistorySetMaxLen(100);

    char* line_c;
    std::cout << "Welcome to the Interactive EEG ACT Analyzer." << std::endl;
    print_help();

    while ((line_c = linenoise("> ")) != nullptr) {
        std::string line(line_c);
        if (!line.empty()) {
            linenoiseHistoryAdd(line_c);
            linenoiseHistorySave(history_file);
        }

#if 0
// --- Live mode implementation ---

static std::string iso8601_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

static void live_analysis_loop() {
    // Determine backend
    bool use_mlx = false;
    ACT_MLX_f* act_mlx = nullptr;
    ACT_CPU* act_cpu_ptr = nullptr;
    ACT_Accelerate* act_accel_ptr = nullptr;

    if (live_backend == LiveBackend::MLX || (live_backend == LiveBackend::AUTO && backend_sel == BackendSel::MLX)) {
        if (act_accel_f) {
            act_mlx = dynamic_cast<ACT_MLX_f*>(act_accel_f.get());
            if (act_mlx) use_mlx = true;
        }
    }
    if (!use_mlx) {
        if (act_cpu) act_cpu_ptr = act_cpu.get();
        else if (act_accel) act_accel_ptr = act_accel.get();
    }

    ACT_CPU::TransformOptions opts; opts.order = live_order; opts.refine = live_refine; opts.residual_threshold = 1e-6;
    using clock = std::chrono::steady_clock;
    auto last_feedback_time = clock::now();

    while (live_running.load()) {
        int64_t sc = sample_counter.load();
        if (sc - last_processed < live_hop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        // Need enough data buffered
        if (rb_tp9.size() < (size_t)live_window || rb_af7.size() < (size_t)live_window || rb_af8.size() < (size_t)live_window || rb_tp10.size() < (size_t)live_window) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const int64_t window_start = sc - live_window;

        // Gather windows
        std::vector<float> v_tp9, v_af7, v_af8, v_tp10;
        if (!rb_tp9.latestWindow(live_window, v_tp9)) continue;
        if (!rb_af7.latestWindow(live_window, v_af7)) continue;
        if (!rb_af8.latestWindow(live_window, v_af8)) continue;
        if (!rb_tp10.latestWindow(live_window, v_tp10)) continue;

        // Convert to Eigen::VectorXd (double)
        std::vector<Eigen::VectorXd> xs;
        xs.reserve(4);
        auto copy_to_eig = [](const std::vector<float>& v){
            Eigen::VectorXd x(v.size());
            for (size_t i = 0; i < v.size(); ++i) x[(int)i] = (double)v[i];
            return x;
        };
        xs.emplace_back(copy_to_eig(v_tp9));
        xs.emplace_back(copy_to_eig(v_af7));
        xs.emplace_back(copy_to_eig(v_af8));
        xs.emplace_back(copy_to_eig(v_tp10));

        // Run batched analysis
        std::vector<ACT_CPU::TransformResult> results_cpu;
        std::vector<ACT_CPU_f::TransformResult> results_mlx;
        if (use_mlx && act_mlx) {
            auto r = actmlx::transform_batch(*act_mlx, xs, opts);
            results_mlx = std::move(r);
        } else if (act_cpu_ptr) {
            results_cpu = actmt::transform_batch(*act_cpu_ptr, xs, opts);
        } else if (act_accel_ptr) {
            results_cpu = actmt::transform_batch(*act_accel_ptr, xs, opts);
        } else {
            std::cout << "[live] No suitable backend available; stopping analysis." << std::endl;
            live_running = false;
            break;
        }

        // Emit NDJSON and CLI feedback
        for (int ch = 0; ch < 4; ++ch) {
            const char* cname = kChannels[ch];
            int used_order = 0;
            double err = 0.0;
            std::vector<logging::ChirpletJson> chirps;
            if (use_mlx) {
                const auto& R = results_mlx[ch];
                used_order = (int)R.params.rows();
                err = (double)R.error;
                chirps.reserve(used_order);
                for (int i = 0; i < used_order; ++i) {
                    double tc_local = (double)R.params(i,0);
                    double tc_global = tc_local + (double)window_start;
                    chirps.push_back({
                        tc_global,
                        tc_global / sampling_frequency,
                        (double)R.params(i,1),
                        1000.0 * std::exp((double)R.params(i,2)),
                        (double)R.params(i,3),
                        (double)R.coeffs[i]
                    });
                }
            } else {
                const auto& R = results_cpu[ch];
                used_order = (int)R.params.rows();
                err = R.error;
                chirps.reserve(used_order);
                for (int i = 0; i < used_order; ++i) {
                    double tc_local = R.params(i,0);
                    double tc_global = tc_local + (double)window_start;
                    chirps.push_back({
                        tc_global,
                        tc_global / sampling_frequency,
                        R.params(i,1),
                        1000.0 * std::exp(R.params(i,2)),
                        R.params(i,3),
                        R.coeffs[i]
                    });
                }
            }

            // NDJSON line
            logging::NDJSONLogger::log_window_result(cname, window_start, err, used_order, chirps, iso8601_now());
        }

        // Throttled/optional CLI feedback
        if (!live_quiet) {
            auto now = clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_feedback_time).count();
            if (elapsed_ms >= static_cast<long long>(live_feedback_interval_ms)) {
                auto print_top = [&](auto& R, const char* cname){
                    int k = std::min(live_feedback_top, (int)R.params.rows());
                    std::cout << cname << " found=" << R.params.rows() << "/" << live_order;
                    for (int i = 0; i < k; ++i) {
                        double tc_s = ((double)R.params(i,0) + (double)window_start) / sampling_frequency;
                        double fc = (double)R.params(i,1);
                        double dur_ms = 1000.0 * std::exp((double)R.params(i,2));
                        double c = (double)R.params(i,3);
                        double a = (double)R.coeffs[i];
                        std::cout << " | tc=" << std::fixed << std::setprecision(3) << tc_s
                                  << "s fc=" << std::setprecision(1) << fc
                                  << "Hz dur=" << std::setprecision(0) << dur_ms
                                  << "ms c=" << std::setprecision(1) << c
                                  << "Hz/s a=" << std::setprecision(3) << a;
                    }
                };

                std::cout << "[WIN start=" << window_start << "] ";
                if (use_mlx) {
                    print_top(results_mlx[0], "TP9"); std::cout << "  ";
                    print_top(results_mlx[1], "AF7"); std::cout << "  ";
                    print_top(results_mlx[2], "AF8"); std::cout << "  ";
                    print_top(results_mlx[3], "TP10");
                } else {
                    print_top(results_cpu[0], "TP9"); std::cout << "  ";
                    print_top(results_cpu[1], "AF7"); std::cout << "  ";
                    print_top(results_cpu[2], "AF8"); std::cout << "  ";
                    print_top(results_cpu[3], "TP10");
                }
                std::array<int,4> hs = {hs_last[0].load(), hs_last[1].load(), hs_last[2].load(), hs_last[3].load()};
                std::cout << "  HS=[" << hs[0] << "," << hs[1] << "," << hs[2] << "," << hs[3] << "]";
                std::cout << " blink=" << blink_pending.load() << " jaw=" << jaw_pending.load();
                std::cout << std::endl;
                last_feedback_time = now;
            }
        }

        last_processed = sc;
    }
}

void handle_muse_start(const std::vector<std::string>& args) {
    if (live_running.load()) { std::cout << "Live mode already running." << std::endl; return; }
    if (args.size() < 3) {
        std::cout << "Usage: muse_start <port> <csv_path> <json_dir> [--window L] [--hop H] [--order K] [--refine 0|1] [--backend auto|mlx|cpu] [--feedback_top N] [--json_max_mb MB] [--json_max_files N] [--json_flush_interval_ms MS]" << std::endl;
        return;
    }

    // Validate existing backend & dictionary
    int dict_len = 0;
    int dict_size = 0;
    std::string backend_name = "";
    if (act_accel_f && backend_sel == BackendSel::MLX) { dict_len = act_accel_f->get_length(); dict_size = act_accel_f->get_dict_size(); backend_name = "MLX(f32)"; }
    else if (act_cpu) { dict_len = act_cpu->get_length(); dict_size = act_cpu->get_dict_size(); backend_name = "CPU"; }
    else if (act_accel) { dict_len = act_accel->get_length(); dict_size = act_accel->get_dict_size(); backend_name = "ACCEL"; }
    else { std::cout << "Error: No dictionary loaded. Use 'create_dictionary' or 'load_dictionary' first (CPU or MLX)." << std::endl; return; }

    try {
        live_port = std::stoi(args[0]);
    } catch (...) { std::cout << "Invalid port." << std::endl; return; }
    live_csv_path = args[1];
    live_json_base = args[2];

    // Defaults
    live_window = dict_len;
    live_hop = 64;
    live_order = 10;
    live_refine = true;
    live_feedback_top = 1;
    live_quiet = true;
    live_feedback_interval_ms = 1000;
    live_backend = LiveBackend::AUTO;
    size_t json_max_mb = 25, json_max_files = 10, json_flush_ms = 1000;

    // Parse optional flags
    for (size_t i = 3; i + 1 < args.size(); i += 2) {
        const std::string& k = args[i]; const std::string& v = args[i+1];
        if (k == "--window") live_window = std::stoi(v);
        else if (k == "--hop") live_hop = std::stoi(v);
        else if (k == "--order") live_order = std::stoi(v);
        else if (k == "--refine") live_refine = (v == "1" || v == "true" || v == "TRUE");
        else if (k == "--backend") {
            if (v == "mlx") live_backend = LiveBackend::MLX; else if (v == "cpu") live_backend = LiveBackend::CPU; else live_backend = LiveBackend::AUTO;
        }
        else if (k == "--feedback_top") live_feedback_top = std::stoi(v);
        else if (k == "--feedback_interval_ms") live_feedback_interval_ms = (size_t)std::stoul(v);
        else if (k == "--quiet") live_quiet = (v == "1" || v == "true" || v == "TRUE");
        else if (k == "--json_max_mb") json_max_mb = (size_t)std::stoul(v);
        else if (k == "--json_max_files") json_max_files = (size_t)std::stoul(v);
        else if (k == "--json_flush_interval_ms") json_flush_ms = (size_t)std::stoul(v);
        else { std::cout << "Unknown flag: " << k << std::endl; return; }
    }

    if (live_window != dict_len) {
        std::cout << "Error: --window (" << live_window << ") must match dictionary length (" << dict_len << ")." << std::endl;
        return;
    }

    // Init NDJSON logger (assumes json_dir exists)
    std::string session_ts = iso8601_now();
    live_session_id = session_ts; // simple session id
    std::string ndjson_path = live_json_base + "/session_" + session_ts + ".ndjson";
    try {
        logging::NDJSONLogger::init_rotating(ndjson_path, json_max_mb, json_max_files, json_flush_ms);
    } catch (const std::exception& e) {
        std::cerr << "Failed to init logger: " << e.what() << std::endl;
        return;
    }

    logging::ParamRangesJson pr{param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step};
    logging::NDJSONLogger::log_session_meta(live_session_id, sampling_frequency, live_window, live_hop, live_order, live_window - live_hop, pr, backend_name, dict_size);

    // Prepare ring buffers
    rb_capacity = (size_t)live_window * 8;
    rb_tp9.reset(rb_capacity); rb_af7.reset(rb_capacity); rb_af8.reset(rb_capacity); rb_tp10.reset(rb_capacity);

    // Open CSV
    {
        std::lock_guard<std::mutex> lk(live_csv_mu);
        live_csv.open(live_csv_path);
        if (!live_csv.is_open()) { std::cout << "Error: cannot open CSV path." << std::endl; return; }
        live_csv << "timestamp,TP9,AF7,AF8,TP10,horseshoe_TP9,horseshoe_AF7,horseshoe_AF8,horseshoe_TP10,blink,jaw_clench\n";
    }

    // Start OSC receiver
    muse_rx.reset(new MuseOSCReceiver());
    muse_rx->on_eeg([](double ts_sec, float tp9, float af7, float af8, float tp10){
        rb_tp9.push(tp9); rb_af7.push(af7); rb_af8.push(af8); rb_tp10.push(tp10);
        int b = blink_pending.exchange(0);
        int j = jaw_pending.exchange(0);
        int h0 = hs_last[0].load(); int h1 = hs_last[1].load(); int h2 = hs_last[2].load(); int h3 = hs_last[3].load();
        {
            std::lock_guard<std::mutex> lk(live_csv_mu);
            if (live_csv.is_open()) {
                live_csv << std::fixed << std::setprecision(6) << ts_sec << ","
                         << tp9 << "," << af7 << "," << af8 << "," << tp10 << ","
                         << h0 << "," << h1 << "," << h2 << "," << h3 << ","
                         << b << "," << j << "\n";
            }
        }
        sample_counter.fetch_add(1);
    });
    muse_rx->on_horseshoe([](double /*ts*/, const std::array<int,4>& hs){
        hs_last[0].store(hs[0]); hs_last[1].store(hs[1]); hs_last[2].store(hs[2]); hs_last[3].store(hs[3]);
        logging::NDJSONLogger::log_quality_event(hs, blink_pending.load(), jaw_pending.load(), iso8601_now());
    });
    muse_rx->on_blink([](double /*ts*/, int blink){ blink_pending.store(blink ? 1 : 0); });
    muse_rx->on_jaw([](double /*ts*/, int jaw){ jaw_pending.store(jaw ? 1 : 0); });

    if (!muse_rx->start(live_port)) { std::cout << "Failed to start OSC receiver." << std::endl; return; }

    live_running = true;
    last_processed = 0;
    analysis_th = std::thread(live_analysis_loop);
    std::cout << "Live OSC started on port " << live_port << ", window=" << live_window << ", hop=" << live_hop << ", order=" << live_order << ", refine=" << (live_refine?"1":"0") << "." << std::endl;
}

void handle_muse_stop(const std::vector<std::string>& /*args*/) {
    if (!live_running.load()) { std::cout << "Live mode not running." << std::endl; return; }
    live_running = false;
    if (muse_rx) muse_rx->stop();
    if (analysis_th.joinable()) analysis_th.join();
    {
        std::lock_guard<std::mutex> lk(live_csv_mu);
        if (live_csv.is_open()) live_csv.close();
    }
    logging::NDJSONLogger::shutdown();
    std::cout << "Live OSC stopped." << std::endl;
}

void handle_muse_status(const std::vector<std::string>& /*args*/) {
    std::cout << "Live: " << (live_running.load()?"ON":"OFF") << ", port=" << live_port << std::endl;
    std::cout << "RB fill: TP9=" << rb_tp9.size() << "/" << rb_capacity
              << " AF7=" << rb_af7.size() << "/" << rb_capacity
              << " AF8=" << rb_af8.size() << "/" << rb_capacity
              << " TP10=" << rb_tp10.size() << "/" << rb_capacity << std::endl;
    std::array<int,4> hs = {hs_last[0].load(), hs_last[1].load(), hs_last[2].load(), hs_last[3].load()};
    std::cout << "Horseshoe: [" << hs[0] << "," << hs[1] << "," << hs[2] << "," << hs[3] << "]" << std::endl;
    std::cout << "Pending: blink=" << blink_pending.load() << " jaw=" << jaw_pending.load() << std::endl;
}
#endif
        linenoiseFree(line_c);

        std::stringstream ss(line);
        std::string command;
        ss >> command;

        std::vector<std::string> args;
        std::string arg;
        while (ss >> arg) {
            args.push_back(arg);
        }

        if (command == "load_csv") {
            handle_load_csv(args);
        } else if (command == "select") {
            handle_select_data(args);
        } else if (command == "params") {
            handle_set_params(args);
        } else if (command == "backend") {
            if (args.size() != 1) { std::cout << "Usage: backend <act|cpu|accel|mlx>" << std::endl; continue; }
            std::string b = args[0];
            if (b == "act") backend_sel = BackendSel::ACT;
            else if (b == "cpu") backend_sel = BackendSel::CPU;
            else if (b == "accel") backend_sel = BackendSel::ACCEL;
            else if (b == "mlx") backend_sel = BackendSel::MLX;
            else { std::cout << "Unknown backend. Use act|cpu|accel|mlx" << std::endl; continue; }
            // Clear any existing instances
            act_legacy.reset(); act_cpu.reset(); act_accel.reset(); act_accel_f.reset();
            std::cout << "Backend set to '" << b << "'." << std::endl;
        } else if (command == "coarse_only") {
            if (args.size() != 1) { std::cout << "Usage: coarse_only <0|1>" << std::endl; continue; }
            coarse_only = (args[0] == "1" || args[0] == "true" || args[0] == "TRUE");
            std::cout << "Coarse-only set to " << (coarse_only ? "ON" : "OFF") << std::endl;
        } else if (command == "create_dictionary") {
            handle_create_dictionary(args);
        } else if (command == "save_dictionary") {
            handle_save_dictionary(args);
        } else if (command == "load_dictionary") {
            handle_load_dictionary(args);
        } else if (command == "analyze") {
            handle_analyze(args);
        } else if (command == "analyze_samples") {
            handle_analyze_samples(args);
        } else if (command == "show_params") {
            print_current_params();
        } else if (command == "muse_start") {
            handle_muse_start(args);
        } else if (command == "muse_stop") {
            handle_muse_stop(args);
        } else if (command == "muse_status") {
            handle_muse_status(args);
        } else if (command == "help") {
            print_help();
        } else if (command == "exit") {
            break;
        } else if (!command.empty()) {
            std::cout << "Unknown command. Type 'help' for a list of commands." << std::endl;
        }
    }

    return 0;
}

// --- Command Implementations (Stubs) ---

void handle_load_csv(const std::vector<std::string>& args) {
    if (args.size() != 1) {
        std::cout << "Usage: load_csv <filepath>" << std::endl;
        return;
    }

    const std::string& filename = args[0];
    current_filename = filename; // Store filename for JSON output
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Clear previous data
    csv_data.clear();
    csv_headers.clear();
    selected_signal.clear();

    std::string line;
    // Read header
    if (std::getline(file, line)) {
        std::stringstream header_ss(line);
        std::string header;
        while (std::getline(header_ss, header, ',')) {
            csv_headers.push_back(header);
        }
    }

    // Read data rows
    std::vector<std::vector<double>> temp_data(csv_headers.size());
    size_t row_count = 0;
    while (std::getline(file, line)) {
        std::stringstream line_ss(line);
        std::string cell;
        size_t col_idx = 0;
        while (std::getline(line_ss, cell, ',') && col_idx < csv_headers.size()) {
            try {
                temp_data[col_idx].push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                temp_data[col_idx].push_back(NAN); // Use NAN for non-numeric data
            }
            col_idx++;
        }
        row_count++;
    }

    csv_data = temp_data;
    file.close();

    std::cout << "Successfully loaded " << row_count << " samples with " << csv_headers.size() << " columns." << std::endl;
    std::cout << "Columns: ";
    for(size_t i = 0; i < csv_headers.size(); ++i) {
        std::cout << i << ":" << csv_headers[i] << " ";
    }
    std::cout << std::endl;
}

void handle_select_data(const std::vector<std::string>& args) {
    if (args.size() != 3) {
        std::cout << "Usage: select <column_idx> <start_sample> <num_samples>" << std::endl;
        return;
    }
    if (csv_data.empty()) {
        std::cout << "No CSV data loaded. Use 'load_csv' first." << std::endl;
        return;
    }

    try {
        size_t col_idx = std::stoul(args[0]);
        size_t start_idx = std::stoul(args[1]);
        size_t num_samples = std::stoul(args[2]);

        if (col_idx >= csv_data.size()) {
            std::cout << "Error: Column index out of bounds." << std::endl;
            return;
        }
        if (start_idx >= csv_data[col_idx].size()) {
            std::cout << "Error: Start sample out of bounds." << std::endl;
            return;
        }
        if (num_samples <= 0 || (start_idx + num_samples) > csv_data[col_idx].size()) {
            std::cout << "Error: Invalid number of samples." << std::endl;
            return;
        }

        selected_column_index = col_idx;
        start_sample = start_idx;

        selected_signal.assign(csv_data[col_idx].begin() + start_idx, 
                               csv_data[col_idx].begin() + start_idx + num_samples);

        size_t original_size = selected_signal.size();
        selected_signal.erase(
            std::remove_if(selected_signal.begin(), selected_signal.end(),
                           [](double val) { return std::isnan(val); }),
            selected_signal.end());
        size_t removed_count = original_size - selected_signal.size();

        if (removed_count > 0) {
            std::cout << "Cleaned signal: Removed " << removed_count << " invalid (NaN) samples." << std::endl;
        }

        if (selected_signal.empty()) {
            std::cout << "Error: No valid data points in the selected range." << std::endl;
            return;
        }

        double sum = std::accumulate(selected_signal.begin(), selected_signal.end(), 0.0);
        double mean = sum / selected_signal.size();
        for (double& val : selected_signal) {
            val -= mean;
        }

        std::cout << "Selected signal from column '" << csv_headers[col_idx] << "'." << std::endl;
        print_signal_stats(csv_data[col_idx], "Full Column Stats");
        print_signal_stats(selected_signal, "Selected Segment Stats");

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    }
}

void print_signal_stats(const std::vector<double>& signal, const std::string& title) {
    if (signal.empty()) {
        std::cout << "No signal selected to print stats for." << std::endl;
        return;
    }

    std::vector<double> clean_signal;
    clean_signal.reserve(signal.size());
    for(double val : signal) {
        if (!std::isnan(val)) {
            clean_signal.push_back(val);
        }
    }

    if (clean_signal.empty()) {
        std::cout << "Signal contains only NaN values." << std::endl;
        return;
    }

    double min_val = *std::min_element(clean_signal.begin(), clean_signal.end());
    double max_val = *std::max_element(clean_signal.begin(), clean_signal.end());
    double sum = std::accumulate(clean_signal.begin(), clean_signal.end(), 0.0);
    double mean = sum / clean_signal.size();
    
    double sq_sum = 0.0;
    for (double val : clean_signal) {
        sq_sum += (val - mean) * (val - mean);
    }
    double std_dev = std::sqrt(sq_sum / clean_signal.size());

    std::cout << "\n--- " << title << " ---" << std::endl;
    std::cout << "Size: " << clean_signal.size() << " samples" << std::endl;
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Std Dev: " << std_dev << std::endl;
    std::cout << "-------------------------" << std::endl;
}

void print_current_params() {
    std::cout << "\n--- Current ACT Parameter Ranges ---" << std::endl;
    std::cout << "Time Center (tc):   min=" << param_ranges.tc_min 
              << ", max=" << param_ranges.tc_max 
              << ", step=" << param_ranges.tc_step << std::endl;
    std::cout << "Frequency (fc):     min=" << param_ranges.fc_min 
              << ", max=" << param_ranges.fc_max 
              << ", step=" << param_ranges.fc_step << std::endl;
    std::cout << "Log Duration (logDt): min=" << param_ranges.logDt_min 
              << ", max=" << param_ranges.logDt_max 
              << ", step=" << param_ranges.logDt_step << std::endl;
    std::cout << "Chirp Rate (c):     min=" << param_ranges.c_min 
              << ", max=" << param_ranges.c_max 
              << ", step=" << param_ranges.c_step << std::endl;
    std::cout << "------------------------------------" << std::endl;
    print_estimated_size();
}

void handle_set_params(const std::vector<std::string>& args) {
    if (args.empty()) {
        print_current_params();
        return;
    }
    if (args.size() != 4) {
        std::cout << "Usage: params <tc|fc|logDt|c> <min> <max> <step>" << std::endl;
        return;
    }

    const std::string& param_type = args[0];
    try {
        double min_val = std::stod(args[1]);
        double max_val = std::stod(args[2]);
        double step_val = std::stod(args[3]);

        if (param_type == "tc") {
            param_ranges.tc_min = min_val; param_ranges.tc_max = max_val; param_ranges.tc_step = step_val;
        } else if (param_type == "fc") {
            param_ranges.fc_min = min_val; param_ranges.fc_max = max_val; param_ranges.fc_step = step_val;
        } else if (param_type == "logDt") {
            param_ranges.logDt_min = min_val; param_ranges.logDt_max = max_val; param_ranges.logDt_step = step_val;
        } else if (param_type == "c") {
            param_ranges.c_min = min_val; param_ranges.c_max = max_val; param_ranges.c_step = step_val;
        } else {
            std::cout << "Unknown parameter type. Use one of: tc, fc, logDt, c" << std::endl;
            return;
        }

        std::cout << "Parameter '" << param_type << "' updated." << std::endl;
        print_current_params();

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    }
}

void handle_create_dictionary(const std::vector<std::string>& /*args*/) {
    if (selected_signal.empty()) {
        std::cout << "No signal selected. Use 'select' command first." << std::endl;
        return;
    }

    std::cout << "Creating dictionary in memory..." << std::endl;
    try {
        int len = static_cast<int>(selected_signal.size());
        // Build per-backend
        act_legacy.reset(); act_cpu.reset(); act_accel.reset(); act_accel_f.reset();
        if (backend_sel == BackendSel::ACT) {
            act_legacy.reset(new ACT(sampling_frequency, len, param_ranges, false, false));
            int size = act_legacy->generate_chirplet_dictionary();
            std::cout << "Dictionary created (ACT). Size=" << size << std::endl;
        } else if (backend_sel == BackendSel::MLX) {
            // MLX float32 path uses float ParameterRanges
            ACT_CPU_f::ParameterRanges cpu_ranges_f(
                param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step
            );
            act_accel_f.reset(new ACT_MLX_f(sampling_frequency, len, cpu_ranges_f, true));
            int size = act_accel_f->generate_chirplet_dictionary();
            std::cout << "Dictionary created (MLX). Size=" << size << std::endl;
        } else if (backend_sel == BackendSel::ACCEL) {
            ACT_CPU::ParameterRanges cpu_ranges(
                param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step
            );
            act_accel.reset(new ACT_Accelerate(sampling_frequency, len, cpu_ranges, false));
            int size = act_accel->generate_chirplet_dictionary();
            std::cout << "Dictionary created (ACCEL). Size=" << size << std::endl;
        } else { // CPU
            ACT_CPU::ParameterRanges cpu_ranges(
                param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step
            );
            act_cpu.reset(new ACT_CPU(sampling_frequency, len, cpu_ranges, false));
            int size = act_cpu->generate_chirplet_dictionary();
            std::cout << "Dictionary created (CPU). Size=" << size << std::endl;
        }
        print_dict_summary();
    } catch (const std::exception& e) {
        std::cerr << "Error creating dictionary: " << e.what() << std::endl;
    }
}

void handle_save_dictionary(const std::vector<std::string>& args) {
    if (!act_legacy && !act_cpu && !act_accel && !act_accel_f) {
        std::cout << "No dictionary in memory. Use 'create_dictionary' or 'load_dictionary' first." << std::endl;
        return;
    }
    if (args.size() != 1) {
        std::cout << "Usage: save_dictionary <filepath>" << std::endl;
        return;
    }
    const std::string& path = args[0];
    std::cout << "Saving dictionary to '" << path << "'..." << std::flush;
    bool ok = false;
    if (act_legacy) ok = act_legacy->save_dictionary(path);
    else if (act_cpu) ok = act_cpu->save_dictionary(path);
    else if (act_accel_f) ok = act_accel_f->save_dictionary(path);
    else if (act_accel) ok = act_accel->save_dictionary(path);
    if (ok) {
        std::cout << " done." << std::endl;
    } else {
        std::cout << " FAILED." << std::endl;
    }
}

void handle_load_dictionary(const std::vector<std::string>& args) {
    if (args.size() != 1) {
        std::cout << "Usage: load_dictionary <filepath>" << std::endl;
        return;
    }
    const std::string& path = args[0];
    std::cout << "Loading dictionary from '" << path << "'..." << std::flush;
    act_legacy.reset(); act_cpu.reset(); act_accel.reset(); act_accel_f.reset();
    bool ok = false;
    if (backend_sel == BackendSel::ACT) {
        auto loaded = ACT::load_dictionary<ACT>(path, false);
        if (loaded) { act_legacy = std::move(loaded); ok = true; }
    } else if (backend_sel == BackendSel::MLX) {
        // Load as CPU-style dictionary into MLX float backend
        auto loaded = ACT_CPU_f::load_dictionary<ACT_MLX_f>(path, false);
        if (loaded) { act_accel_f = std::move(loaded); ok = true; }
    } else if (backend_sel == BackendSel::ACCEL) {
        auto loaded = ACT_CPU::load_dictionary<ACT_Accelerate>(path, false);
        if (loaded) { act_accel = std::move(loaded); ok = true; }
    } else { // CPU
        auto loaded = ACT_CPU::load_dictionary<ACT_CPU>(path, false);
        if (loaded) { act_cpu = std::move(loaded); ok = true; }
    }
    if (!ok) {
        std::cout << " FAILED." << std::endl;
        std::cerr << "\nError: Could not load dictionary from file." << std::endl;
        return;
    }
    std::cout << " done." << std::endl;
    print_dict_summary();
}

void handle_analyze(const std::vector<std::string>& args) {
    if (!act_legacy && !act_cpu && !act_accel && !act_accel_f) {
        std::cout << "Dictionary not created. Use 'create_dictionary' first." << std::endl;
        return;
    }
    if (args.size() != 2 && args.size() != 4) {
        std::cout << "Usage: analyze <num_chirplets> <residual_threshold> [save <filename>]" << std::endl;
        return;
    }

    try {
        int num_chirplets = std::stoi(args[0]);
        double residual_threshold = std::stod(args[1]);
        std::string save_filename = "";
        bool do_save = false;

        if (args.size() == 4) {
            if (args[2] == "save") {
                save_filename = args[3];
                do_save = true;
            } else {
                std::cout << "Usage: analyze <num_chirplets> <residual_threshold> [save <filename>]" << std::endl;
                return;
            }
        }

        std::cout << "\nPerforming ACT analysis to find top " << num_chirplets << " chirplets..." << std::endl;
        auto __analysis_t0 = std::chrono::high_resolution_clock::now();

        ACT::TransformResult result;
        if (act_legacy) {
            result = act_legacy->transform(selected_signal, num_chirplets, residual_threshold);
        } else if (act_accel_f) {
            if (coarse_only) {
                ACT_CPU_f::TransformOptions opts; opts.order = num_chirplets; opts.refine = false; opts.residual_threshold = residual_threshold;
                // Build a float view without extra allocation
                std::vector<float> sf(selected_signal.size());
                for (size_t j = 0; j < selected_signal.size(); ++j) sf[j] = static_cast<float>(selected_signal[j]);
                Eigen::Map<const act::VecX<float>> xmap(sf.data(), (int)sf.size());
                auto rf = act_accel_f->transform(xmap, opts);
                // Convert
                result.params.resize(rf.params.rows());
                for (int rr = 0; rr < rf.params.rows(); ++rr) result.params[rr] = { rf.params(rr,0), rf.params(rr,1), rf.params(rr,2), rf.params(rr,3) };
                result.coeffs.resize(rf.coeffs.size()); for (int k = 0; k < rf.coeffs.size(); ++k) result.coeffs[k] = rf.coeffs[k];
                result.signal.resize(rf.signal.size()); for (int k = 0; k < rf.signal.size(); ++k) result.signal[k] = rf.signal[k];
                result.residue.resize(rf.residue.size()); for (int k = 0; k < rf.residue.size(); ++k) result.residue[k] = rf.residue[k];
                result.approx.resize(rf.approx.size()); for (int k = 0; k < rf.approx.size(); ++k) result.approx[k] = rf.approx[k];
                result.error = rf.error;
            } else {
                std::vector<float> sf(selected_signal.size());
                for (size_t j = 0; j < selected_signal.size(); ++j) sf[j] = static_cast<float>(selected_signal[j]);
                auto rf = act_accel_f->transform(sf, num_chirplets, static_cast<float>(residual_threshold));
                result.params.resize(rf.params.rows());
                for (int rr = 0; rr < rf.params.rows(); ++rr) result.params[rr] = { rf.params(rr,0), rf.params(rr,1), rf.params(rr,2), rf.params(rr,3) };
                result.coeffs.resize(rf.coeffs.size()); for (int k = 0; k < rf.coeffs.size(); ++k) result.coeffs[k] = rf.coeffs[k];
                result.signal.resize(rf.signal.size()); for (int k = 0; k < rf.signal.size(); ++k) result.signal[k] = rf.signal[k];
                result.residue.resize(rf.residue.size()); for (int k = 0; k < rf.residue.size(); ++k) result.residue[k] = rf.residue[k];
                result.approx.resize(rf.approx.size()); for (int k = 0; k < rf.approx.size(); ++k) result.approx[k] = rf.approx[k];
                result.error = rf.error;
            }
        } else {
            ACT_CPU::TransformResult r;
            if (coarse_only) {
                ACT_CPU::TransformOptions opts; opts.order = num_chirplets; opts.refine = false; opts.residual_threshold = residual_threshold;
                Eigen::Map<const Eigen::VectorXd> x(selected_signal.data(), (int)selected_signal.size());
                r = (act_accel ? act_accel->transform(x, opts) : act_cpu->transform(x, opts));
            } else {
                r = (act_accel ? act_accel->transform(selected_signal, num_chirplets, residual_threshold)
                               : act_cpu->transform(selected_signal, num_chirplets, residual_threshold));
            }
            // Convert to ACT::TransformResult
            result.params.resize(r.params.rows());
            for (int i = 0; i < r.params.rows(); ++i) result.params[i] = { r.params(i,0), r.params(i,1), r.params(i,2), r.params(i,3) };
            result.coeffs.resize(r.coeffs.size()); for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
            result.signal.resize(r.signal.size()); for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
            result.residue.resize(r.residue.size()); for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
            result.approx.resize(r.approx.size()); for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
            result.error = r.error;
        }
        
        double __analysis_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::high_resolution_clock::now() - __analysis_t0
                               ).count() / 1000.0;
        std::cout << "\n--- Analysis Results ---" << std::endl;
        std::cout << "Analysis Time: " << std::fixed << std::setprecision(2) << __analysis_ms << " ms" << std::endl;
        std::cout << "Final Error: " << result.error << std::endl;
        std::cout << "Number of chirplets: " << result.params.size() << std::endl;
        std::cout << "------------------------" << std::endl;
        for (size_t i = 0; i < result.params.size(); ++i) {
            double tc_sec = result.params[i][0] / sampling_frequency;
            double fc_hz = result.params[i][1];
            double duration_ms = 1000.0 * std::exp(result.params[i][2]);
            double chirp_rate = result.params[i][3];
            double coeff = result.coeffs[i];
            double residual = result.residue[i];

            std::cout << "Chirplet " << (i + 1) << ":\n";
            std::cout << "  Time: " << std::fixed << std::setprecision(3) << tc_sec << " s\n";
            std::cout << "  Frequency: " << std::setprecision(1) << fc_hz << " Hz\n";
            std::cout << "  Duration: " << std::setprecision(0) << duration_ms << " ms\n";
            std::cout << "  Chirp Rate: " << std::setprecision(1) << chirp_rate << " Hz/s\n";
            std::cout << "  Coefficient: " << std::setprecision(4) << coeff << std::endl;
            std::cout << "  Residual: " << std::setprecision(4) << residual << std::endl;
        }

        if (do_save) {
            save_analysis_to_json(save_filename, result, num_chirplets, residual_threshold, true);
        }


    } catch (const std::exception& e) {
        std::cerr << "Error during analysis: " << e.what() << std::endl;
    }
}

void print_estimated_size() {
    if (selected_signal.empty()) {
        std::cout << "Cannot estimate size: No signal selected." << std::endl;
        return;
    }

    // Ensure tc_max is signal length dependent if not explicitly set by user
    if (param_ranges.tc_max == 0) {
        param_ranges.tc_max = selected_signal.size() - 1;
    }

    int tc_count = static_cast<int>((param_ranges.tc_max - param_ranges.tc_min) / param_ranges.tc_step) + 1;
    int fc_count = static_cast<int>((param_ranges.fc_max - param_ranges.fc_min) / param_ranges.fc_step) + 1;
    int logDt_count = static_cast<int>((param_ranges.logDt_max - param_ranges.logDt_min) / param_ranges.logDt_step) + 1;
    int c_count = static_cast<int>((param_ranges.c_max - param_ranges.c_min) / param_ranges.c_step) + 1;

    long long dict_size = (long long)tc_count * fc_count * logDt_count * c_count;
    double mem_size_mb = (dict_size * selected_signal.size() * sizeof(double)) / (1024.0 * 1024.0);

    std::cout << "\n--- Estimated Dictionary Size ---" << std::endl;
    std::cout << "Chirplets: " << dict_size << std::endl;
    std::cout << "Memory: " << std::fixed << std::setprecision(2) << mem_size_mb << " MB" << std::endl;
    std::cout << "---------------------------------" << std::endl;
}

void print_analysis_results(const ACT::TransformResult& result, int time_offset) {
    for (size_t i = 0; i < result.params.size(); ++i) {
        double tc_sec = (result.params[i][0] + time_offset) / sampling_frequency;
        double fc_hz = result.params[i][1];
        double duration_ms = 1000.0 * std::exp(result.params[i][2]);
        double chirp_rate = result.params[i][3];
        double coeff = result.coeffs[i];

        std::cout << "Chirplet " << (i + 1) << ":\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << tc_sec << " s\n";
        std::cout << "  Frequency: " << std::setprecision(1) << fc_hz << " Hz\n";
        std::cout << "  Duration: " << std::setprecision(0) << duration_ms << " ms\n";
        std::cout << "  Chirp Rate: " << std::setprecision(1) << chirp_rate << " Hz/s\n";
        std::cout << "  Coefficient: " << std::setprecision(4) << coeff << std::endl;
    }
}

void handle_analyze_samples(const std::vector<std::string>& args) {
    if (!act_legacy && !act_cpu && !act_accel) {
        std::cout << "Dictionary not created. Use 'create_dictionary' first." << std::endl;
        return;
    }
    if (selected_column_index == -1) {
        std::cout << "No signal selected. Use 'select' command first." << std::endl;
        return;
    }
    if (args.size() != 3 && args.size() != 5) {
        std::cout << "Usage: analyze_samples <num_chirps> <end_sample> <overlap> [save <filename>]" << std::endl;
        return;
    }

    try {
        int num_chirps = std::stoi(args[0]);
        int end_sample = std::stoi(args[1]);
        int overlap = std::stoi(args[2]);
        std::string save_filename = "";
        bool do_save = false;

        if (args.size() == 5) {
            if (args[3] == "save") {
                save_filename = args[4];
                do_save = true;
            } else {
                std::cout << "Usage: analyze_samples <num_chirps> <end_sample> <overlap> [save <filename>]" << std::endl;
                return;
            }
        }

        int window_size = 0;
        if (act_legacy) window_size = act_legacy->get_dictionary_length();
        else if (act_cpu) window_size = act_cpu->get_length();
        else if (act_accel_f) window_size = act_accel_f->get_length();
        else if (act_accel) window_size = act_accel->get_length();

        if (end_sample <= start_sample || overlap < 0 || overlap >= window_size) {
            std::cout << "Invalid arguments. End sample must be after start sample, and overlap must be less than window size." << std::endl;
            return;
        }

        const auto& full_signal = csv_data[selected_column_index];
        if (static_cast<size_t>(end_sample) > full_signal.size()) {
            std::cout << "End sample is out of bounds for the selected column." << std::endl;
            return;
        }

        std::cout << "Starting sample-based analysis from " << start_sample << " to " << end_sample 
                  << " with window size " << window_size << " and overlap " << overlap << "..." << std::endl;

        // Collect results to save a single combined JSON if requested
        std::vector<ACT::TransformResult> collected_results;
        std::vector<int> collected_starts;

        for (int current_start = start_sample; current_start + window_size <= end_sample; current_start += (window_size - overlap)) {
            std::cout << "\n[DEBUG] Processing window starting at sample " << current_start << "..." << std::endl;

            std::vector<double> raw_window(full_signal.begin() + current_start, full_signal.begin() + current_start + window_size);
            std::cout << "[DEBUG]   Raw window size: " << raw_window.size() << std::endl;

            std::vector<double> cleaned_window;
            for (double val : raw_window) {
                if (!std::isnan(val)) {
                    cleaned_window.push_back(val);
                }
            }
            std::cout << "[DEBUG]   Cleaned window size: " << cleaned_window.size() << std::endl;

            if (cleaned_window.size() != static_cast<size_t>(window_size)) {
                std::cout << "[INFO] Skipping window at " << current_start << ": size after cleaning (" << cleaned_window.size() 
                          << ") does not match dictionary size (" << window_size << ")." << std::endl;
                continue;
            }

            double sum = std::accumulate(cleaned_window.begin(), cleaned_window.end(), 0.0);
            double mean = sum / cleaned_window.size();
            for (double& val : cleaned_window) {
                val -= mean;
            }
            std::cout << "[DEBUG]   DC offset removed. Calling transform..." << std::endl;

            ACT::TransformResult result;
            if (act_legacy) {
                result = act_legacy->transform(cleaned_window, num_chirps);
            } else if (act_accel_f) {
                if (coarse_only) {
                    ACT_CPU_f::TransformOptions opts; opts.order = num_chirps; opts.refine = false; opts.residual_threshold = 1e-6f;
                    std::vector<float> sf(cleaned_window.size());
                    for (size_t j = 0; j < cleaned_window.size(); ++j) sf[j] = static_cast<float>(cleaned_window[j]);
                    Eigen::Map<const act::VecX<float>> x_f(sf.data(), (int)sf.size());
                    auto rf = act_accel_f->transform(x_f, opts);
                    result.params.resize(rf.params.rows());
                    for (int rr = 0; rr < rf.params.rows(); ++rr) result.params[rr] = { rf.params(rr,0), rf.params(rr,1), rf.params(rr,2), rf.params(rr,3) };
                    result.coeffs.resize(rf.coeffs.size()); for (int k = 0; k < rf.coeffs.size(); ++k) result.coeffs[k] = rf.coeffs[k];
                    result.signal.resize(rf.signal.size()); for (int k = 0; k < rf.signal.size(); ++k) result.signal[k] = rf.signal[k];
                    result.residue.resize(rf.residue.size()); for (int k = 0; k < rf.residue.size(); ++k) result.residue[k] = rf.residue[k];
                    result.approx.resize(rf.approx.size()); for (int k = 0; k < rf.approx.size(); ++k) result.approx[k] = rf.approx[k];
                    result.error = rf.error;
                } else {
                    std::vector<float> sf(cleaned_window.size());
                    for (size_t j = 0; j < cleaned_window.size(); ++j) sf[j] = static_cast<float>(cleaned_window[j]);
                    auto rf = act_accel_f->transform(sf, num_chirps, 1e-6f);
                    result.params.resize(rf.params.rows());
                    for (int rr = 0; rr < rf.params.rows(); ++rr) result.params[rr] = { rf.params(rr,0), rf.params(rr,1), rf.params(rr,2), rf.params(rr,3) };
                    result.coeffs.resize(rf.coeffs.size()); for (int k = 0; k < rf.coeffs.size(); ++k) result.coeffs[k] = rf.coeffs[k];
                    result.signal.resize(rf.signal.size()); for (int k = 0; k < rf.signal.size(); ++k) result.signal[k] = rf.signal[k];
                    result.residue.resize(rf.residue.size()); for (int k = 0; k < rf.residue.size(); ++k) result.residue[k] = rf.residue[k];
                    result.approx.resize(rf.approx.size()); for (int k = 0; k < rf.approx.size(); ++k) result.approx[k] = rf.approx[k];
                    result.error = rf.error;
                }
            } else {
                ACT_CPU::TransformResult r;
                if (coarse_only) {
                    ACT_CPU::TransformOptions opts; opts.order = num_chirps; opts.refine = false; opts.residual_threshold = 1e-6;
                    Eigen::Map<const Eigen::VectorXd> x(cleaned_window.data(), (int)cleaned_window.size());
                    r = (act_accel ? act_accel->transform(x, opts) : act_cpu->transform(x, opts));
                } else {
                    r = (act_accel ? act_accel->transform(cleaned_window, num_chirps, 1e-6)
                                   : act_cpu->transform(cleaned_window, num_chirps, 1e-6));
                }
                result.params.resize(r.params.rows());
                for (int i = 0; i < r.params.rows(); ++i) result.params[i] = { r.params(i,0), r.params(i,1), r.params(i,2), r.params(i,3) };
                result.coeffs.resize(r.coeffs.size()); for (int k = 0; k < r.coeffs.size(); ++k) result.coeffs[k] = r.coeffs[k];
                result.signal.resize(r.signal.size()); for (int k = 0; k < r.signal.size(); ++k) result.signal[k] = r.signal[k];
                result.residue.resize(r.residue.size()); for (int k = 0; k < r.residue.size(); ++k) result.residue[k] = r.residue[k];
                result.approx.resize(r.approx.size()); for (int k = 0; k < r.approx.size(); ++k) result.approx[k] = r.approx[k];
                result.error = r.error;
            }

            std::cout << "--- Window starting at sample " << current_start << " ---" << std::endl;
            print_analysis_results(result, current_start);

            if (do_save) {
                collected_results.push_back(result);
                collected_starts.push_back(current_start);
            }
        }

        // Save a single combined JSON file after processing all windows
        if (do_save) {
            // Compute combined file in one go
            save_multiwindow_combined_to_json(save_filename, collected_results, collected_starts, end_sample, overlap, num_chirps);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    }
}

void save_analysis_to_json(const std::string& filename, const ACT::TransformResult& result, 
                          int num_chirplets, double residual_threshold, bool is_single, 
                          int window_start, int end_sample, int overlap, int num_chirps) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    // Use consistent high precision for numeric values
    file << std::setprecision(12) << std::fixed;

    file << "{\n";
    file << "  \"source_file\": \"" << current_filename << "\",\n";
    file << "  \"column_name\": \"" << (selected_column_index >= 0 ? csv_headers[selected_column_index] : "") << "\",\n";
    file << "  \"column_index\": " << selected_column_index << ",\n";
    file << "  \"start_sample\": " << start_sample << ",\n";
    file << "  \"num_samples\": " << selected_signal.size() << ",\n";
    file << "  \"sampling_frequency\": " << sampling_frequency << ",\n";
    file << "  \"param_ranges\": {\n";
    file << "    \"tc_min\": " << param_ranges.tc_min << ",\n";
    file << "    \"tc_max\": " << param_ranges.tc_max << ",\n";
    file << "    \"tc_step\": " << param_ranges.tc_step << ",\n";
    file << "    \"fc_min\": " << param_ranges.fc_min << ",\n";
    file << "    \"fc_max\": " << param_ranges.fc_max << ",\n";
    file << "    \"fc_step\": " << param_ranges.fc_step << ",\n";
    file << "    \"logDt_min\": " << param_ranges.logDt_min << ",\n";
    file << "    \"logDt_max\": " << param_ranges.logDt_max << ",\n";
    file << "    \"logDt_step\": " << param_ranges.logDt_step << ",\n";
    file << "    \"c_min\": " << param_ranges.c_min << ",\n";
    file << "    \"c_max\": " << param_ranges.c_max << ",\n";
    file << "    \"c_step\": " << param_ranges.c_step << "\n";
    file << "  },\n";

    if (is_single) {
        file << "  \"analysis_type\": \"single\",\n";
        file << "  \"num_chirplets\": " << num_chirplets << ",\n";
        file << "  \"residual_threshold\": " << residual_threshold << ",\n";
        file << "  \"result\": {\n";
        file << "    \"error\": " << result.error << ",\n";
        file << "    \"chirplets\": [\n";
        for (size_t i = 0; i < result.params.size(); ++i) {
            file << "      {\n";
            file << "        \"index\": " << (i + 1) << ",\n";
            file << "        \"time_center_samples\": " << result.params[i][0] << ",\n";
            file << "        \"time_center_seconds\": " << (result.params[i][0] / sampling_frequency) << ",\n";
            file << "        \"frequency_hz\": " << result.params[i][1] << ",\n";
            file << "        \"duration_ms\": " << (1000.0 * std::exp(result.params[i][2])) << ",\n";
            file << "        \"chirp_rate_hz_per_s\": " << result.params[i][3] << ",\n";
            file << "        \"coefficient\": " << result.coeffs[i] << "\n";
            file << "      }";
            if (i < result.params.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ],\n";
        // Export approximation and residue arrays for reconstruction/validation
        file << "    \"approx\": [\n";
        for (size_t j = 0; j < result.approx.size(); ++j) {
            file << "      " << result.approx[j];
            if (j < result.approx.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ],\n";
        file << "    \"residue\": [\n";
        for (size_t j = 0; j < result.residue.size(); ++j) {
            file << "      " << result.residue[j];
            if (j < result.residue.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ]\n";
        file << "  }\n";
    } else {
        file << "  \"analysis_type\": \"multi_sample\",\n";
        file << "  \"num_chirps\": " << num_chirps << ",\n";
        file << "  \"end_sample\": " << end_sample << ",\n";
        file << "  \"overlap\": " << overlap << ",\n";
        int ws = 0;
        if (act_legacy) ws = act_legacy->get_dictionary_length();
        else if (act_cpu) ws = act_cpu->get_length();
        else if (act_accel) ws = act_accel->get_length();
        file << "  \"window_size\": " << ws << ",\n";
        file << "  \"window_start\": " << window_start << ",\n";
        file << "  \"result\": {\n";
        file << "    \"error\": " << result.error << ",\n";
        file << "    \"chirplets\": [\n";
        for (size_t i = 0; i < result.params.size(); ++i) {
            file << "      {\n";
            file << "        \"index\": " << (i + 1) << ",\n";
            file << "        \"time_center_samples\": " << result.params[i][0] << ",\n";
            file << "        \"time_center_seconds\": " << (result.params[i][0] / sampling_frequency) << ",\n";
            file << "        \"frequency_hz\": " << result.params[i][1] << ",\n";
            file << "        \"duration_ms\": " << (1000.0 * std::exp(result.params[i][2])) << ",\n";
            file << "        \"chirp_rate_hz_per_s\": " << result.params[i][3] << ",\n";
            file << "        \"coefficient\": " << result.coeffs[i] << "\n";
            file << "      }";
            if (i < result.params.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ],\n";
        // Export approximation and residue arrays for reconstruction/validation
        file << "    \"approx\": [\n";
        for (size_t j = 0; j < result.approx.size(); ++j) {
            file << "      " << result.approx[j];
            if (j < result.approx.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ],\n";
        file << "    \"residue\": [\n";
        for (size_t j = 0; j < result.residue.size(); ++j) {
            file << "      " << result.residue[j];
            if (j < result.residue.size() - 1) file << ",";
            file << "\n";
        }
        file << "    ]\n";
        file << "  }\n";
    }

    file << "}\n";

    file.close();
    std::cout << "Analysis results saved to " << filename << std::endl;
}

// Save all analyze_samples windows into a single combined JSON file with global time centers
void save_multiwindow_combined_to_json(
    const std::string& filename,
    const std::vector<ACT::TransformResult>& results,
    const std::vector<int>& window_starts,
    int end_sample,
    int overlap,
    int num_chirps
) {
    if (results.size() != window_starts.size()) {
        std::cerr << "Error: results and window_starts size mismatch." << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    file << std::setprecision(12) << std::fixed;

    int window_size = 0;
    if (act_legacy) window_size = act_legacy->get_dictionary_length();
    else if (act_cpu) window_size = act_cpu->get_length();
    else if (act_accel) window_size = act_accel->get_length();
    else window_size = results.empty() ? 0 : (int)results.front().approx.size();
    int total_len = end_sample - start_sample;
    if (total_len < 0) total_len = 0;

    file << "{\n";
    file << "  \"source_file\": \"" << current_filename << "\",\n";
    file << "  \"column_name\": \"" << (selected_column_index >= 0 ? csv_headers[selected_column_index] : "") << "\",\n";
    file << "  \"column_index\": " << selected_column_index << ",\n";
    file << "  \"start_sample\": " << start_sample << ",\n";
    file << "  \"num_samples\": " << total_len << ",\n";
    file << "  \"sampling_frequency\": " << sampling_frequency << "\n";
    file << "  ,\n  \"param_ranges\": {\n";
    file << "    \"tc_min\": " << param_ranges.tc_min << ",\n";
    file << "    \"tc_max\": " << param_ranges.tc_max << ",\n";
    file << "    \"tc_step\": " << param_ranges.tc_step << ",\n";
    file << "    \"fc_min\": " << param_ranges.fc_min << ",\n";
    file << "    \"fc_max\": " << param_ranges.fc_max << ",\n";
    file << "    \"fc_step\": " << param_ranges.fc_step << ",\n";
    file << "    \"logDt_min\": " << param_ranges.logDt_min << ",\n";
    file << "    \"logDt_max\": " << param_ranges.logDt_max << ",\n";
    file << "    \"logDt_step\": " << param_ranges.logDt_step << ",\n";
    file << "    \"c_min\": " << param_ranges.c_min << ",\n";
    file << "    \"c_max\": " << param_ranges.c_max << ",\n";
    file << "    \"c_step\": " << param_ranges.c_step << "\n";
    file << "  },\n";

    file << "  \"analysis_type\": \"multi_sample_combined\",\n";
    file << "  \"num_chirps_per_window\": " << num_chirps << ",\n";
    file << "  \"end_sample\": " << end_sample << ",\n";
    file << "  \"overlap\": " << overlap << ",\n";
    file << "  \"window_size\": " << window_size << ",\n";
    file << "  \"window_starts\": [";
    for (size_t i = 0; i < window_starts.size(); ++i) {
        file << window_starts[i];
        if (i + 1 < window_starts.size()) file << ", ";
    }
    file << "],\n";

    file << "  \"result\": {\n";
    file << "    \"error_per_window\": [";
    for (size_t w = 0; w < results.size(); ++w) {
        file << results[w].error;
        if (w + 1 < results.size()) file << ", ";
    }
    file << "],\n";

    file << "    \"chirplets\": [\n";
    bool first = true;
    for (size_t w = 0; w < results.size(); ++w) {
        const auto& res = results[w];
        int wstart = window_starts[w];
        for (size_t i = 0; i < res.params.size(); ++i) {
            if (!first) file << ",\n";
            first = false;
            file << "      {\n";
            file << "        \"index\": " << (i + 1) << ",\n";
            file << "        \"time_center_samples\": " << (res.params[i][0] + wstart) << ",\n";
            file << "        \"time_center_seconds\": " << ((res.params[i][0] + wstart) / sampling_frequency) << ",\n";
            file << "        \"frequency_hz\": " << res.params[i][1] << ",\n";
            file << "        \"duration_ms\": " << (1000.0 * std::exp(res.params[i][2])) << ",\n";
            file << "        \"chirp_rate_hz_per_s\": " << res.params[i][3] << ",\n";
            file << "        \"coefficient\": " << res.coeffs[i] << "\n";
            file << "      }";
        }
    }
    file << "\n    ]\n";

    file << "  }\n";
    file << "}\n";

    file.close();
    std::cout << "Combined analysis results saved to " << filename << std::endl;
}

void print_help() {
    std::cout << "\nAvailable Commands:\n";
    std::cout << "  load_csv <filepath>                               - Load EEG data from a CSV file.\n";
    std::cout << "  select <column_idx> <start_sample> <num_samples>  - Select a signal segment for analysis.\n";
    std::cout << "  params <tc|fc|logDt|c> <min> <max> <step>       - Set parameter ranges for the dictionary.\n";
    std::cout << "  backend <act|cpu|accel|mlx>                      - Select implementation backend.\n";
    std::cout << "  coarse_only <0|1>                                - Toggle coarse-only (skip BFGS refinement) for CPU/ACCEL.\n";
    std::cout << "  create_dictionary                               - Generate a chirplet dictionary in memory.\n";
    std::cout << "  save_dictionary <filepath>                      - Save the in-memory dictionary to a file.\n";
    std::cout << "  load_dictionary <filepath>                      - Load a dictionary from a file and print its summary.\n";
    std::cout << "  analyze <num_chirplets> <residual_threshold> [save <filename>] - Run ACT analysis to find the top N chirplets.\n";
    std::cout << "  analyze_samples <num_chirps> <end_sample> <overlap> [save <filename>] - Analyze sequence of samples with overlap.\n";
    std::cout << "  muse_start <port> <csv_path> <json_dir> [--window L] [--hop H] [--order K] [--refine 0|1] [--backend auto|mlx|cpu] [--feedback_top N] [--feedback_interval_ms MS] [--quiet 0|1] [--json_max_mb MB] [--json_max_files N] [--json_flush_interval_ms MS] - Start live OSC analysis.\n";
    std::cout << "  muse_status                                     - Show live capture/analyze status.\n";
    std::cout << "  muse_stop                                       - Stop live OSC analysis.\n";
    std::cout << "  help                                            - Show this help message.\n";
    std::cout << "  exit                                            - Exit the application.\n" << std::endl;
}

void print_dict_summary() {
    if (!act_legacy && !act_cpu && !act_accel && !act_accel_f) {
        std::cout << "No dictionary loaded/created." << std::endl;
        return;
    }
    std::cout << "\n--- Dictionary Summary ---" << std::endl;
    double fs = sampling_frequency; int length = 0; ACT::ParameterRanges pr = param_ranges; bool complex = false; const char* bname = "CPU";
    if (act_legacy) {
        fs = act_legacy->get_FS(); length = act_legacy->get_length(); complex = act_legacy->get_complex_mode(); pr = act_legacy->get_param_ranges(); bname = "ACT";
    }
    else if (act_cpu) {
        fs = act_cpu->get_FS(); length = act_cpu->get_length(); pr = ACT::ParameterRanges(pr.tc_min, pr.tc_max, pr.tc_step, pr.fc_min, pr.fc_max, pr.fc_step, pr.logDt_min, pr.logDt_max, pr.logDt_step, pr.c_min, pr.c_max, pr.c_step); bname = "CPU";
    }
    else if (act_accel_f) {
        fs = act_accel_f->get_FS(); length = act_accel_f->get_length(); pr = ACT::ParameterRanges(pr.tc_min, pr.tc_max, pr.tc_step, pr.fc_min, pr.fc_max, pr.fc_step, pr.logDt_min, pr.logDt_max, pr.logDt_step, pr.c_min, pr.c_max, pr.c_step);
        bname = (backend_sel == BackendSel::MLX ? "MLX(f32)" : "ACCEL(f32)");
    }
    else if (act_accel) {
        fs = act_accel->get_FS(); length = act_accel->get_length(); pr = ACT::ParameterRanges(pr.tc_min, pr.tc_max, pr.tc_step, pr.fc_min, pr.fc_max, pr.fc_step, pr.logDt_min, pr.logDt_max, pr.logDt_step, pr.c_min, pr.c_max, pr.c_step); bname = "ACCEL";
    }
    std::cout << "Backend: " << bname << std::endl;
    std::cout << "FS: " << fs << " Hz" << std::endl;
    std::cout << "Length: " << length << " samples" << std::endl;
    std::cout << "Complex mode: " << (complex ? "true" : "false") << std::endl;
    
    std::cout << "Parameter Ranges:" << std::endl;
    std::cout << "  tc: min=" << pr.tc_min << ", max=" << pr.tc_max << ", step=" << pr.tc_step << std::endl;
    std::cout << "  fc: min=" << pr.fc_min << ", max=" << pr.fc_max << ", step=" << pr.fc_step << std::endl;
    std::cout << "  logDt: min=" << pr.logDt_min << ", max=" << pr.logDt_max << ", step=" << pr.logDt_step << std::endl;
    std::cout << "  c: min=" << pr.c_min << ", max=" << pr.c_max << ", step=" << pr.c_step << std::endl;
    int dsz = (act_legacy ? act_legacy->get_dict_size() : (act_cpu ? act_cpu->get_dict_size() : (act_accel_f ? act_accel_f->get_dict_size() : (act_accel ? act_accel->get_dict_size() : 0))));
    std::cout << "Dictionary size: " << dsz << std::endl;
    std::cout << "--------------------------\n";
}

// --- Live mode implementation (file-scope definitions) ---
static std::string iso8601_now() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return std::string(buf);
}

static void live_analysis_loop() {
    // Determine backend
    bool use_mlx = false;
    ACT_MLX_f* act_mlx = nullptr;
    ACT_CPU* act_cpu_ptr = nullptr;
    ACT_Accelerate* act_accel_ptr = nullptr;

    if (live_backend == LiveBackend::MLX || (live_backend == LiveBackend::AUTO && backend_sel == BackendSel::MLX)) {
        if (act_accel_f) {
            act_mlx = dynamic_cast<ACT_MLX_f*>(act_accel_f.get());
            if (act_mlx) use_mlx = true;
        }
    }
    if (!use_mlx) {
        if (act_cpu) act_cpu_ptr = act_cpu.get();
        else if (act_accel) act_accel_ptr = act_accel.get();
    }

    ACT_CPU::TransformOptions opts; opts.order = live_order; opts.refine = live_refine; opts.residual_threshold = 1e-6;

    while (live_running.load()) {
        int64_t sc = sample_counter.load();
        if (sc - last_processed < live_hop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        // Need enough data buffered
        if (rb_tp9.size() < (size_t)live_window || rb_af7.size() < (size_t)live_window || rb_af8.size() < (size_t)live_window || rb_tp10.size() < (size_t)live_window) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        const int64_t window_start = sc - live_window;

        // Gather windows
        std::vector<float> v_tp9, v_af7, v_af8, v_tp10;
        if (!rb_tp9.latestWindow(live_window, v_tp9)) continue;
        if (!rb_af7.latestWindow(live_window, v_af7)) continue;
        if (!rb_af8.latestWindow(live_window, v_af8)) continue;
        if (!rb_tp10.latestWindow(live_window, v_tp10)) continue;

        // Convert to Eigen::VectorXd (double)
        std::vector<Eigen::VectorXd> xs;
        xs.reserve(4);
        auto copy_to_eig = [](const std::vector<float>& v){
            Eigen::VectorXd x(v.size());
            for (size_t i = 0; i < v.size(); ++i) x[(int)i] = (double)v[i];
            return x;
        };
        xs.emplace_back(copy_to_eig(v_tp9));
        xs.emplace_back(copy_to_eig(v_af7));
        xs.emplace_back(copy_to_eig(v_af8));
        xs.emplace_back(copy_to_eig(v_tp10));

        // Run batched analysis
        std::vector<ACT_CPU::TransformResult> results_cpu;
        std::vector<ACT_CPU_f::TransformResult> results_mlx;
        if (use_mlx && act_mlx) {
            auto r = actmlx::transform_batch(*act_mlx, xs, opts);
            results_mlx = std::move(r);
        } else if (act_cpu_ptr) {
            results_cpu = actmt::transform_batch(*act_cpu_ptr, xs, opts);
        } else if (act_accel_ptr) {
            results_cpu = actmt::transform_batch(*act_accel_ptr, xs, opts);
        } else {
            std::cout << "[live] No suitable backend available; stopping analysis." << std::endl;
            live_running = false;
            break;
        }

        // Emit NDJSON and CLI feedback
        for (int ch = 0; ch < 4; ++ch) {
            const char* cname = kChannels[ch];
            int used_order = 0;
            double err = 0.0;
            std::vector<logging::ChirpletJson> chirps;
            if (use_mlx) {
                const auto& R = results_mlx[ch];
                used_order = (int)R.params.rows();
                err = (double)R.error;
                chirps.reserve(used_order);
                for (int i = 0; i < used_order; ++i) {
                    double tc_local = (double)R.params(i,0);
                    double tc_global = tc_local + (double)window_start;
                    chirps.push_back({
                        tc_global,
                        tc_global / sampling_frequency,
                        (double)R.params(i,1),
                        1000.0 * std::exp((double)R.params(i,2)),
                        (double)R.params(i,3),
                        (double)R.coeffs[i]
                    });
                }
            } else {
                const auto& R = results_cpu[ch];
                used_order = (int)R.params.rows();
                err = R.error;
                chirps.reserve(used_order);
                for (int i = 0; i < used_order; ++i) {
                    double tc_local = R.params(i,0);
                    double tc_global = tc_local + (double)window_start;
                    chirps.push_back({
                        tc_global,
                        tc_global / sampling_frequency,
                        R.params(i,1),
                        1000.0 * std::exp(R.params(i,2)),
                        R.params(i,3),
                        R.coeffs[i]
                    });
                }
            }

            // NDJSON line
            logging::NDJSONLogger::log_window_result(cname, window_start, err, used_order, chirps, iso8601_now());
        }

        // Live CLI feedback (top N)
        auto print_top = [&](auto& R, const char* cname){
            int k = std::min(live_feedback_top, (int)R.params.rows());
            std::cout << cname << " found=" << R.params.rows() << "/" << live_order;
            for (int i = 0; i < k; ++i) {
                double tc_s = ((double)R.params(i,0) + (double)window_start) / sampling_frequency;
                double fc = (double)R.params(i,1);
                double dur_ms = 1000.0 * std::exp((double)R.params(i,2));
                double c = (double)R.params(i,3);
                double a = (double)R.coeffs[i];
                std::cout << " | tc=" << std::fixed << std::setprecision(3) << tc_s
                          << "s fc=" << std::setprecision(1) << fc
                          << "Hz dur=" << std::setprecision(0) << dur_ms
                          << "ms c=" << std::setprecision(1) << c
                          << "Hz/s a=" << std::setprecision(3) << a;
            }
            std::cout << "";
        };

        std::cout << "[WIN start=" << window_start << "] ";
        if (use_mlx) {
            print_top(results_mlx[0], "TP9"); std::cout << "  ";
            print_top(results_mlx[1], "AF7"); std::cout << "  ";
            print_top(results_mlx[2], "AF8"); std::cout << "  ";
            print_top(results_mlx[3], "TP10");
        } else {
            print_top(results_cpu[0], "TP9"); std::cout << "  ";
            print_top(results_cpu[1], "AF7"); std::cout << "  ";
            print_top(results_cpu[2], "AF8"); std::cout << "  ";
            print_top(results_cpu[3], "TP10");
        }
        std::array<int,4> hs = {hs_last[0].load(), hs_last[1].load(), hs_last[2].load(), hs_last[3].load()};
        std::cout << "  HS=[" << hs[0] << "," << hs[1] << "," << hs[2] << "," << hs[3] << "]";
        std::cout << " blink=" << blink_pending.load() << " jaw=" << jaw_pending.load();
        std::cout << std::endl;

        last_processed = sc;
    }
}

void handle_muse_start(const std::vector<std::string>& args) {
    if (live_running.load()) { std::cout << "Live mode already running." << std::endl; return; }
    if (args.size() < 3) {
        std::cout << "Usage: muse_start <port> <csv_path> <json_dir> [--window L] [--hop H] [--order K] [--refine 0|1] [--backend auto|mlx|cpu] [--feedback_top N] [--json_max_mb MB] [--json_max_files N] [--json_flush_interval_ms MS]" << std::endl;
        return;
    }

    // Validate existing backend & dictionary
    int dict_len = 0;
    int dict_size = 0;
    std::string backend_name = "";
    if (act_accel_f && backend_sel == BackendSel::MLX) { dict_len = act_accel_f->get_length(); dict_size = act_accel_f->get_dict_size(); backend_name = "MLX(f32)"; }
    else if (act_cpu) { dict_len = act_cpu->get_length(); dict_size = act_cpu->get_dict_size(); backend_name = "CPU"; }
    else if (act_accel) { dict_len = act_accel->get_length(); dict_size = act_accel->get_dict_size(); backend_name = "ACCEL"; }
    else { std::cout << "Error: No dictionary loaded. Use 'create_dictionary' or 'load_dictionary' first (CPU or MLX)." << std::endl; return; }

    try {
        live_port = std::stoi(args[0]);
    } catch (...) { std::cout << "Invalid port." << std::endl; return; }
    live_csv_path = args[1];
    live_json_base = args[2];

    // Defaults
    live_window = dict_len;
    live_hop = 64;
    live_order = 10;
    live_refine = true;
    live_feedback_top = 1;
    live_backend = LiveBackend::AUTO;
    size_t json_max_mb = 25, json_max_files = 10, json_flush_ms = 1000;

    // Parse optional flags
    for (size_t i = 3; i + 1 < args.size(); i += 2) {
        const std::string& k = args[i]; const std::string& v = args[i+1];
        if (k == "--window") live_window = std::stoi(v);
        else if (k == "--hop") live_hop = std::stoi(v);
        else if (k == "--order") live_order = std::stoi(v);
        else if (k == "--refine") live_refine = (v == "1" || v == "true" || v == "TRUE");
        else if (k == "--backend") {
            if (v == "mlx") live_backend = LiveBackend::MLX; else if (v == "cpu") live_backend = LiveBackend::CPU; else live_backend = LiveBackend::AUTO;
        }
        else if (k == "--feedback_top") live_feedback_top = std::stoi(v);
        else if (k == "--json_max_mb") json_max_mb = (size_t)std::stoul(v);
        else if (k == "--json_max_files") json_max_files = (size_t)std::stoul(v);
        else if (k == "--json_flush_interval_ms") json_flush_ms = (size_t)std::stoul(v);
        else { std::cout << "Unknown flag: " << k << std::endl; return; }
    }

    if (live_window != dict_len) {
        std::cout << "Error: --window (" << live_window << ") must match dictionary length (" << dict_len << ")." << std::endl;
        return;
    }

    // Init NDJSON logger (assumes json_dir exists)
    std::string session_ts = iso8601_now();
    live_session_id = session_ts; // simple session id
    std::string ndjson_path = live_json_base + "/session_" + session_ts + ".ndjson";
    try {
        logging::NDJSONLogger::init_rotating(ndjson_path, json_max_mb, json_max_files, json_flush_ms);
    } catch (const std::exception& e) {
        std::cerr << "Failed to init logger: " << e.what() << std::endl;
        return;
    }

    logging::ParamRangesJson pr{param_ranges.tc_min, param_ranges.tc_max, param_ranges.tc_step,
                                param_ranges.fc_min, param_ranges.fc_max, param_ranges.fc_step,
                                param_ranges.logDt_min, param_ranges.logDt_max, param_ranges.logDt_step,
                                param_ranges.c_min, param_ranges.c_max, param_ranges.c_step};
    logging::NDJSONLogger::log_session_meta(live_session_id, sampling_frequency, live_window, live_hop, live_order, live_window - live_hop, pr, backend_name, dict_size);

    // Prepare ring buffers
    rb_capacity = (size_t)live_window * 8;
    rb_tp9.reset(rb_capacity); rb_af7.reset(rb_capacity); rb_af8.reset(rb_capacity); rb_tp10.reset(rb_capacity);

    // Open CSV
    {
        std::lock_guard<std::mutex> lk(live_csv_mu);
        live_csv.open(live_csv_path);
        if (!live_csv.is_open()) { std::cout << "Error: cannot open CSV path." << std::endl; return; }
        live_csv << "timestamp,TP9,AF7,AF8,TP10,horseshoe_TP9,horseshoe_AF7,horseshoe_AF8,horseshoe_TP10,blink,jaw_clench\n";
    }

    // Start OSC receiver
    muse_rx.reset(new MuseOSCReceiver());
    muse_rx->on_eeg([](double ts_sec, float tp9, float af7, float af8, float tp10){
        rb_tp9.push(tp9); rb_af7.push(af7); rb_af8.push(af8); rb_tp10.push(tp10);
        int b = blink_pending.exchange(0);
        int j = jaw_pending.exchange(0);
        int h0 = hs_last[0].load(); int h1 = hs_last[1].load(); int h2 = hs_last[2].load(); int h3 = hs_last[3].load();
        {
            std::lock_guard<std::mutex> lk(live_csv_mu);
            if (live_csv.is_open()) {
                live_csv << std::fixed << std::setprecision(6) << ts_sec << ","
                         << tp9 << "," << af7 << "," << af8 << "," << tp10 << ","
                         << h0 << "," << h1 << "," << h2 << "," << h3 << ","
                         << b << "," << j << "\n";
            }
        }
        sample_counter.fetch_add(1);
    });
    muse_rx->on_horseshoe([](double /*ts*/, const std::array<int,4>& hs){
        hs_last[0].store(hs[0]); hs_last[1].store(hs[1]); hs_last[2].store(hs[2]); hs_last[3].store(hs[3]);
        logging::NDJSONLogger::log_quality_event(hs, blink_pending.load(), jaw_pending.load(), iso8601_now());
    });
    muse_rx->on_blink([](double /*ts*/, int blink){ blink_pending.store(blink ? 1 : 0); });
    muse_rx->on_jaw([](double /*ts*/, int jaw){ jaw_pending.store(jaw ? 1 : 0); });

    if (!muse_rx->start(live_port)) { std::cout << "Failed to start OSC receiver." << std::endl; return; }

    live_running = true;
    last_processed = 0;
    analysis_th = std::thread(live_analysis_loop);
    std::cout << "Live OSC started on port " << live_port << ", window=" << live_window << ", hop=" << live_hop << ", order=" << live_order << ", refine=" << (live_refine?"1":"0") << "." << std::endl;
}

void handle_muse_stop(const std::vector<std::string>& /*args*/) {
    if (!live_running.load()) { std::cout << "Live mode not running." << std::endl; return; }
    live_running = false;
    if (muse_rx) muse_rx->stop();
    if (analysis_th.joinable()) analysis_th.join();
    {
        std::lock_guard<std::mutex> lk(live_csv_mu);
        if (live_csv.is_open()) live_csv.close();
    }
    logging::NDJSONLogger::shutdown();
    std::cout << "Live OSC stopped." << std::endl;
}

void handle_muse_status(const std::vector<std::string>& /*args*/) {
    std::cout << "Live: " << (live_running.load()?"ON":"OFF") << ", port=" << live_port << std::endl;
    std::cout << "RB fill: TP9=" << rb_tp9.size() << "/" << rb_capacity
              << " AF7=" << rb_af7.size() << "/" << rb_capacity
              << " AF8=" << rb_af8.size() << "/" << rb_capacity
              << " TP10=" << rb_tp10.size() << "/" << rb_capacity << std::endl;
    std::array<int,4> hs = {hs_last[0].load(), hs_last[1].load(), hs_last[2].load(), hs_last[3].load()};
    std::cout << "Horseshoe: [" << hs[0] << "," << hs[1] << "," << hs[2] << "," << hs[3] << "]" << std::endl;
    std::cout << "Pending: blink=" << blink_pending.load() << " jaw=" << jaw_pending.load() << std::endl;
}

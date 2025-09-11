#include "ACT_SIMD.h"
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

// --- Global State ---
static std::vector<std::vector<double>> csv_data;
static std::vector<double> selected_signal;
static std::vector<std::string> csv_headers;
static int selected_column_index = -1;
static int start_sample = 0;
static ACT::ParameterRanges param_ranges;
static std::unique_ptr<ACT_SIMD> act_analyzer;
static double sampling_frequency = 256.0; // Default Muse sampling rate
static std::string current_filename;

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
        act_analyzer = std::make_unique<ACT_SIMD>(
            sampling_frequency,
            static_cast<int>(selected_signal.size()),
            param_ranges,
            false,  // complex_mode
            false   // verbose
        );
        int size = act_analyzer->generate_chirplet_dictionary();
        std::cout << "Dictionary created successfully. Size=" << size << std::endl;
        print_dict_summary();
    } catch (const std::exception& e) {
        std::cerr << "Error creating dictionary: " << e.what() << std::endl;
    }
}

void handle_save_dictionary(const std::vector<std::string>& args) {
    if (!act_analyzer) {
        std::cout << "No dictionary in memory. Use 'create_dictionary' or 'load_dictionary' first." << std::endl;
        return;
    }
    if (args.size() != 1) {
        std::cout << "Usage: save_dictionary <filepath>" << std::endl;
        return;
    }
    const std::string& path = args[0];
    std::cout << "Saving dictionary to '" << path << "'..." << std::flush;
    if (act_analyzer->save_dictionary(path)) {
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
    auto loaded = ACT::load_dictionary<ACT_SIMD>(path, false);
    if (!loaded) {
        std::cout << " FAILED." << std::endl;
        std::cerr << "\nError: Could not load dictionary from file." << std::endl;
        return;
    }
    act_analyzer = std::move(loaded);
    std::cout << " done." << std::endl;
    print_dict_summary();
}

void handle_analyze(const std::vector<std::string>& args) {
    if (!act_analyzer) {
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

        auto result = act_analyzer->transform(selected_signal, num_chirplets, residual_threshold);
        
        std::cout << "\n--- Analysis Results ---" << std::endl;
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
    double mem_size_mb = (dict_size * selected_signal.size() * sizeof(float)) / (1024.0 * 1024.0);

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
    if (!act_analyzer) {
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

        int window_size = act_analyzer->get_dictionary_length();

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

            auto result = act_analyzer->transform(cleaned_window, num_chirps);

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
        file << "  \"window_size\": " << act_analyzer->get_dictionary_length() << ",\n";
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

    int window_size = act_analyzer ? act_analyzer->get_dictionary_length() : (results.empty() ? 0 : (int)results.front().approx.size());
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
    std::cout << "  create_dictionary                               - Generate a chirplet dictionary in memory.\n";
    std::cout << "  save_dictionary <filepath>                      - Save the in-memory dictionary to a file.\n";
    std::cout << "  load_dictionary <filepath>                      - Load a dictionary from a file and print its summary.\n";
    std::cout << "  analyze <num_chirplets> <residual_threshold> [save <filename>] - Run ACT analysis to find the top N chirplets.\n";
    std::cout << "  analyze_samples <num_chirps> <end_sample> <overlap> [save <filename>] - Analyze sequence of samples with overlap.\n";
    std::cout << "  help                                            - Show this help message.\n";
    std::cout << "  exit                                            - Exit the application.\n" << std::endl;
}

void print_dict_summary() {
    if (!act_analyzer) {
        std::cout << "No dictionary loaded/created." << std::endl;
        return;
    }
    std::cout << "\n--- Dictionary Summary ---" << std::endl;
    std::cout << "FS: " << act_analyzer->get_FS() << " Hz" << std::endl;
    std::cout << "Length: " << act_analyzer->get_length() << " samples" << std::endl;
    std::cout << "Complex mode: " << (act_analyzer->get_complex_mode() ? "true" : "false") << std::endl;
    const auto& pr = act_analyzer->get_param_ranges();
    std::cout << "Parameter Ranges:" << std::endl;
    std::cout << "  tc: min=" << pr.tc_min << ", max=" << pr.tc_max << ", step=" << pr.tc_step << std::endl;
    std::cout << "  fc: min=" << pr.fc_min << ", max=" << pr.fc_max << ", step=" << pr.fc_step << std::endl;
    std::cout << "  logDt: min=" << pr.logDt_min << ", max=" << pr.logDt_max << ", step=" << pr.logDt_step << std::endl;
    std::cout << "  c: min=" << pr.c_min << ", max=" << pr.c_max << ", step=" << pr.c_step << std::endl;
    std::cout << "Dictionary size: " << act_analyzer->get_dict_size() << std::endl;
    std::cout << "--------------------------\n";
}


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

// --- Forward Declarations ---
void handle_load_csv(const std::vector<std::string>& args);
void handle_select_data(const std::vector<std::string>& args);
void handle_set_params(const std::vector<std::string>& args);
void handle_create_dictionary(const std::vector<std::string>& args);
void handle_analyze(const std::vector<std::string>& args);
void handle_analyze_samples(const std::vector<std::string>& args);
void print_analysis_results(const ACT::TransformResult& result, int time_offset = 0);
void print_signal_stats(const std::vector<double>& signal, const std::string& title);
void print_estimated_size();
void print_help();
void print_current_params();

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

    std::cout << "Creating dictionary..." << std::endl;
    try {
        act_analyzer = std::make_unique<ACT_SIMD>(
            sampling_frequency, 
            selected_signal.size(), 
            "foobar.bin", 
            param_ranges, 
            false, // force_recreate
            true,  // use_simd
            false  // verbose
        );
        std::cout << "Dictionary created successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error creating dictionary: " << e.what() << std::endl;
    }
}

void handle_analyze(const std::vector<std::string>& args) {
    if (!act_analyzer) {
        std::cout << "Dictionary not created. Use 'create_dictionary' first." << std::endl;
        return;
    }
    if (args.size() != 2) {
        std::cout << "Usage: analyze <num_chirplets> <residual_threshold>" << std::endl;
        return;
    }

    try {
        int num_chirplets = std::stoi(args[0]);
        double residual_threshold = std::stod(args[1]);
        std::cout << "\nPerforming ACT analysis to find top " << num_chirplets << " chirplets..." << std::endl;

        auto result = act_analyzer->transform(selected_signal, num_chirplets, residual_threshold, true);
        
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
    if (args.size() != 3) {
        std::cout << "Usage: analyze_samples <num_chirps> <end_sample> <overlap>" << std::endl;
        return;
    }

    try {
        int num_chirps = std::stoi(args[0]);
        int end_sample = std::stoi(args[1]);
        int overlap = std::stoi(args[2]);

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

            auto result = act_analyzer->transform(cleaned_window, num_chirps, true);

            std::cout << "--- Window starting at sample " << current_start << " ---" << std::endl;
            print_analysis_results(result, current_start);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    }
}

void print_help() {
    std::cout << "\nAvailable Commands:\n";
    std::cout << "  load_csv <filepath>                               - Load EEG data from a CSV file.\n";
    std::cout << "  select <column_idx> <start_sample> <num_samples>  - Select a signal segment for analysis.\n";
    std::cout << "  params <tc|fc|logDt|c> <min> <max> <step>       - Set parameter ranges for the dictionary.\n";
    std::cout << "  create_dictionary                               - Create the chirplet dictionary based on current parameters.\n";
    std::cout << "  analyze <num_chirplets>                         - Run ACT analysis to find the top N chirplets.\n";
    std::cout << "  analyze_samples <num_chirps> <end_sample> <overlap> - Analyze sequence of samples with overlap.\n";
    std::cout << "  help                                            - Show this help message.\n";
    std::cout << "  exit                                            - Exit the application.\n" << std::endl;
}


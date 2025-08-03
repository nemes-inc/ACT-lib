#ifndef ACT_MULTITHREADED_H
#define ACT_MULTITHREADED_H

#include "ACT.h"
#include <thread>
#include <future>
#include <vector>
#include <mutex>

/**
 * Fixed Multi-threaded version of the Adaptive Chirplet Transform
 * Focus: Parallel signal processing instead of parallelizing matching pursuit
 */
class ACT_MultiThreaded : public ACT {
public:
    /**
     * Constructor - same as base ACT class
     */
    ACT_MultiThreaded(double FS = 256.0, int length = 512, 
                     const std::string& dict_addr = "mt_dict_cache.bin", 
                     const ParameterRanges& ranges = ParameterRanges(), 
                     bool complex_mode = false, 
                     bool force_regenerate = false, bool mute = true);

    /**
     * Process multiple signals in parallel - the main benefit of multi-threading
     * @param signals Vector of input signals
     * @param order Transform order for each signal
     * @param debug Enable debug output
     * @param threads Number of threads to use (0 = auto-detect)
     * @return Vector of transform results
     */
    std::vector<ACT::TransformResult> transform_batch_parallel(
        const std::vector<std::vector<double>>& signals, 
        int order = 5, bool debug = false, int threads = 0);

    /**
     * Single signal transform with optimized dictionary search
     * (Keep matching pursuit sequential but optimize dictionary operations)
     * @param signal Input signal
     * @param order Transform order
     * @param debug Enable debug output
     * @return Transform result
     */
    ACT::TransformResult transform_optimized(const std::vector<double>& signal, 
                                           int order = 5, bool debug = false);

    /**
     * Parallel dictionary search with proper chunking
     * @param signal Input signal
     * @param threads Number of threads to use
     * @return Pair of (best_index, best_value)
     */
    std::pair<int, double> search_dictionary_parallel(const std::vector<double>& signal, 
                                                           int threads = 0);

    // Getters and setters
    int get_num_threads() const { return num_threads; }
    void set_num_threads(int threads) { 
        if (threads > 0) num_threads = threads; 
    }

private:
    int num_threads;
    
    /**
     * Worker function for parallel signal processing
     */
    ACT::TransformResult process_signal_worker(const std::vector<double>& signal, 
                                             int order, bool debug);
    
    /**
     * Worker function for parallel dictionary search with proper parameter mapping
     */
    void search_worker(const std::vector<double>& signal, int start_idx, int end_idx,
                           int& best_idx, double& best_value, std::mutex& result_mutex);
};

#endif // ACT_MULTITHREADED_FIXED_H

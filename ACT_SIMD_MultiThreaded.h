#ifndef ACT_SIMD_MULTITHREADED_H
#define ACT_SIMD_MULTITHREADED_H

#include "ACT_SIMD.h"
#include <thread>
#include <future>
#include <mutex>

/**
 * Combined SIMD + Multi-threaded Adaptive Chirplet Transform
 * 
 * This class combines the best of both optimizations:
 * - SIMD vectorization for dictionary search acceleration (4.5x speedup)
 * - Multi-threading for parallel signal processing (6-7x speedup)
 * 
 * Expected combined performance: 20-30x speedup for batch processing
 */
class ACT_SIMD_MultiThreaded : public ACT_SIMD {
public:
    /**
     * Constructor - same as base ACT class
     */
    ACT_SIMD_MultiThreaded(double FS = 256.0, int length = 512, 
                          const std::string& dict_addr = "simd_mt_dict_cache.bin", 
                          const ParameterRanges& ranges = ParameterRanges(), 
                          bool complex_mode = false, 
                          bool force_regenerate = false, bool mute = true);

    /**
     * Destructor
     */
    ~ACT_SIMD_MultiThreaded();

    /**
     * Process multiple signals in parallel with SIMD-accelerated dictionary search
     * This is the main benefit - combines both optimizations
     * 
     * @param signals Vector of input signals to process
     * @param order Transform order (number of chirplets per signal)
     * @param debug Enable debug output
     * @param threads Number of threads to use (0 = auto-detect)
     * @return Vector of transform results
     */
    std::vector<ACT::TransformResult> transform_batch_simd_parallel(
        const std::vector<std::vector<double>>& signals, 
        int order = 5, bool debug = false, int threads = 0);



    // Getters
    int get_num_threads() const { return num_threads; }
    void set_num_threads(int threads) { 
        if (threads > 0) num_threads = threads; 
        else num_threads = std::thread::hardware_concurrency();
    }

private:
    int num_threads;
    
    /**
     * Worker function for parallel signal processing with SIMD
     * Each thread processes one signal with SIMD-accelerated dictionary search
     */
    ACT::TransformResult process_signal_simd_worker(
        const std::vector<double>& signal, int order, bool debug);
};

#endif // ACT_SIMD_MULTITHREADED_H

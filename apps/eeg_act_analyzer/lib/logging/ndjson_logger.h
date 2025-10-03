#ifndef NDJSON_LOGGER_H
#define NDJSON_LOGGER_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <array>
#include <stdexcept>

// spdlog (header-only)
#include "spdlog/spdlog.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/async.h"
#include "spdlog/fmt/fmt.h"

namespace logging {

struct ParamRangesJson {
    double tc_min, tc_max, tc_step;
    double fc_min, fc_max, fc_step;
    double logDt_min, logDt_max, logDt_step;
    double c_min, c_max, c_step;
};

struct ChirpletJson {
    double tc_samples;      // global sample index
    double tc_seconds;
    double fc_hz;
    double duration_ms;
    double c_hz_per_s;
    double coeff;
};

class NDJSONLogger {
public:
    // Initialize an async rotating logger writing NDJSON lines.
    // max_file_mb: size of each file before rotation.
    // max_files: number of rotated files to keep.
    // flush_every_ms: background flush interval.
    static void init_rotating(const std::string& base_filename,
                              size_t max_file_mb = 25,
                              size_t max_files = 10,
                              size_t flush_every_ms = 1000,
                              size_t queue_size = 8192,
                              bool to_console = false) {
        try {
            if (!spdlog::thread_pool()) {
                spdlog::init_thread_pool(queue_size, 1);
            }
            auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                base_filename, max_file_mb * 1024 * 1024, max_files);
            std::vector<spdlog::sink_ptr> sinks;
            sinks.push_back(rotating_sink);
            if (to_console) {
                auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                sinks.push_back(console_sink);
            }
            auto logger = std::make_shared<spdlog::async_logger>(
                "ndjson",
                sinks.begin(), sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::overrun_oldest);
            spdlog::register_logger(logger);
            spdlog::flush_every(std::chrono::milliseconds(flush_every_ms));
            logger->set_pattern("%v"); // Write raw payload (no timestamp prefix) to keep NDJSON clean
            get_logger() = logger;
        } catch (const spdlog::spdlog_ex& ex) {
            throw std::runtime_error(std::string("NDJSONLogger init failed: ") + ex.what());
        }
    }

    static void shutdown() {
        try {
            spdlog::shutdown();
        } catch (...) {}
    }

    // Emit one line: session metadata
    static void log_session_meta(const std::string& session_id,
                                 double fs,
                                 int window_size,
                                 int hop,
                                 int order,
                                 int overlap,
                                 const ParamRangesJson& pr,
                                 const std::string& backend,
                                 int dict_size) {
        std::string line;
        line.reserve(512);
        line += "{\"type\":\"session_meta\",";
        line += "\"session_id\":\"" + escape_json(session_id) + "\",";
        line += "\"fs\":" + fmt::format("{}", fs) + ",";
        line += "\"window_size\":" + fmt::format("{}", window_size) + ",";
        line += "\"hop\":" + fmt::format("{}", hop) + ",";
        line += "\"order\":" + fmt::format("{}", order) + ",";
        line += "\"overlap\":" + fmt::format("{}", overlap) + ",";
        line += "\"backend\":\"" + escape_json(backend) + "\",";
        line += "\"dict_size\":" + fmt::format("{}", dict_size) + ",";
        line += "\"param_ranges\":{";
        line += "\"tc_min\":" + fmt::format("{}", pr.tc_min) + ",";
        line += "\"tc_max\":" + fmt::format("{}", pr.tc_max) + ",";
        line += "\"tc_step\":" + fmt::format("{}", pr.tc_step) + ",";
        line += "\"fc_min\":" + fmt::format("{}", pr.fc_min) + ",";
        line += "\"fc_max\":" + fmt::format("{}", pr.fc_max) + ",";
        line += "\"fc_step\":" + fmt::format("{}", pr.fc_step) + ",";
        line += "\"logDt_min\":" + fmt::format("{}", pr.logDt_min) + ",";
        line += "\"logDt_max\":" + fmt::format("{}", pr.logDt_max) + ",";
        line += "\"logDt_step\":" + fmt::format("{}", pr.logDt_step) + ",";
        line += "\"c_min\":" + fmt::format("{}", pr.c_min) + ",";
        line += "\"c_max\":" + fmt::format("{}", pr.c_max) + ",";
        line += "\"c_step\":" + fmt::format("{}", pr.c_step) + "}";
        line += "}";
        auto& lg = get_logger(); if (lg) lg->info("{}", line);
    }

    // Emit one line: per-window result for a channel
    static void log_window_result(const std::string& channel,
                                  int64_t window_start,
                                  double error,
                                  int used_order,
                                  const std::vector<ChirpletJson>& chirps,
                                  const std::string& ts_iso8601 = std::string()) {
        std::string chirps_json;
        chirps_json.reserve(chirps.size() * 96);
        chirps_json += "[";
        for (size_t i = 0; i < chirps.size(); ++i) {
            const auto& c = chirps[i];
            std::string obj;
            obj.reserve(128);
            obj += "{\"tc_samples\":" + fmt::format("{}", c.tc_samples);
            obj += ",\"tc_seconds\":" + fmt::format("{}", c.tc_seconds);
            obj += ",\"fc_hz\":" + fmt::format("{}", c.fc_hz);
            obj += ",\"duration_ms\":" + fmt::format("{}", c.duration_ms);
            obj += ",\"c_hz_per_s\":" + fmt::format("{}", c.c_hz_per_s);
            obj += ",\"coeff\":" + fmt::format("{}", c.coeff) + "}";
            chirps_json += obj;
            if (i + 1 < chirps.size())chirps_json += ",";
        }
        chirps_json += "]";

        std::string line;
        line.reserve(256 + chirps_json.size());
        line += "{\"type\":\"window_result\",";
        line += "\"ts\":\"" + escape_json(ts_iso8601) + "\",";
        line += "\"channel\":\"" + escape_json(channel) + "\",";
        line += "\"window_start\":" + fmt::format("{}", window_start) + ",";
        line += "\"error\":" + fmt::format("{}", error) + ",";
        line += "\"used_order\":" + fmt::format("{}", used_order) + ",";
        line += "\"chirplets\":" + chirps_json + "}";
        auto& lg = get_logger(); if (lg) lg->info("{}", line);
    }

    // Emit one line: quality indicators
    static void log_quality_event(const std::array<int,4>& horseshoe,
                                  int blink,
                                  int jaw_clench,
                                  const std::string& ts_iso8601 = std::string()) {
        std::string line;
        line.reserve(160);
        line += "{\"type\":\"quality_event\",";
        line += "\"ts\":\"" + escape_json(ts_iso8601) + "\",";
        line += "\"horseshoe\":[" + fmt::format("{}", horseshoe[0]) + "," + fmt::format("{}", horseshoe[1]) + "," + fmt::format("{}", horseshoe[2]) + "," + fmt::format("{}", horseshoe[3]) + "],";
        line += "\"blink\":" + fmt::format("{}", blink) + ",";
        line += "\"jaw_clench\":" + fmt::format("{}", jaw_clench) + "}";
        auto& lg = get_logger(); if (lg) lg->info("{}", line);
    }

private:
    static std::shared_ptr<spdlog::logger>& get_logger() {
        static std::shared_ptr<spdlog::logger> s_logger;
        return s_logger;
    }
    static std::string escape_json(const std::string& s) {
        std::string out; out.reserve(s.size());
        for (char ch : s) {
            switch (ch) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += ch; break;
            }
        }
        return out;
    }
};

} // namespace logging

#endif // NDJSON_LOGGER_H

#ifndef MUSE_OSC_RECEIVER_H
#define MUSE_OSC_RECEIVER_H

#include <functional>
#include <thread>
#include <atomic>
#include <array>
#include <string>

// Forward declare OSCPP types only in implementation to keep header lightweight.

class MuseOSCReceiver {
public:
    using EEGCallback = std::function<void(double ts_sec, float tp9, float af7, float af8, float tp10)>;
    using HorseshoeCallback = std::function<void(double ts_sec, const std::array<int,4>& hs)>;
    using BlinkCallback = std::function<void(double ts_sec, int blink)>;
    using JawCallback = std::function<void(double ts_sec, int jaw)>;

    MuseOSCReceiver();
    ~MuseOSCReceiver();

    // Non-copyable
    MuseOSCReceiver(const MuseOSCReceiver&) = delete;
    MuseOSCReceiver& operator=(const MuseOSCReceiver&) = delete;

    bool start(int port);
    void stop();

    void on_eeg(EEGCallback cb) { eeg_cb_ = std::move(cb); }
    void on_horseshoe(HorseshoeCallback cb) { horseshoe_cb_ = std::move(cb); }
    void on_blink(BlinkCallback cb) { blink_cb_ = std::move(cb); }
    void on_jaw(JawCallback cb) { jaw_cb_ = std::move(cb); }

private:
    void run_loop(int port);

    std::thread th_;
    std::atomic<bool> running_{false};

    EEGCallback eeg_cb_;
    HorseshoeCallback horseshoe_cb_;
    BlinkCallback blink_cb_;
    JawCallback jaw_cb_;
};

#endif // MUSE_OSC_RECEIVER_H

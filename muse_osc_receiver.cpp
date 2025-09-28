#include "muse_osc_receiver.h"

#include <oscpp/server.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdio>

// POSIX sockets (macOS/Linux)
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>

namespace {
static double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}
}

MuseOSCReceiver::MuseOSCReceiver() {}
MuseOSCReceiver::~MuseOSCReceiver() { stop(); }

bool MuseOSCReceiver::start(int port) {
    if (running_.load()) return true;
    running_ = true;
    th_ = std::thread(&MuseOSCReceiver::run_loop, this, port);
    return true;
}

void MuseOSCReceiver::stop() {
    if (!running_.load()) return;
    running_ = false;
    if (th_.joinable()) th_.join();
}

void MuseOSCReceiver::run_loop(int port) {
    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::perror("socket");
        running_ = false;
        return;
    }
    int reuse = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        ::close(sock);
        running_ = false;
        return;
    }

    // Optional: set recv timeout to allow periodic running_ checks
    timeval tv; tv.tv_sec = 0; tv.tv_usec = 200000; // 200 ms
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    std::vector<char> buf(2048);

    while (running_.load()) {
        sockaddr_in src{}; socklen_t srclen = sizeof(src);
        ssize_t n = ::recvfrom(sock, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&src), &srclen);
        if (n < 0) {
            // timeout or error; continue
            continue;
        }
        double ts = now_sec();
        try {
            OSCPP::Server::Packet packet(buf.data(), static_cast<size_t>(n));
            auto handle_message = [&](const OSCPP::Server::Message& msg){
                const char* addr = msg.address();
                auto args = msg.args();
                if (std::strcmp(addr, "/muse/eeg") == 0) {
                    // TP9, AF7, AF8, TP10
                    float tp9 = args.float32();
                    float af7 = args.float32();
                    float af8 = args.float32();
                    float tp10 = args.float32();
                    if (eeg_cb_) eeg_cb_(ts, tp9, af7, af8, tp10);
                } else if (std::strcmp(addr, "/muse/elements/horseshoe") == 0) {
                    // 4 floats, values 1,2,4 typically
                    std::array<int,4> hs{};
                    hs[0] = static_cast<int>(args.float32());
                    hs[1] = static_cast<int>(args.float32());
                    hs[2] = static_cast<int>(args.float32());
                    hs[3] = static_cast<int>(args.float32());
                    if (horseshoe_cb_) horseshoe_cb_(ts, hs);
                } else if (std::strcmp(addr, "/muse/elements/blink") == 0) {
                    int blink = args.int32();
                    if (blink_cb_) blink_cb_(ts, blink);
                } else if (std::strcmp(addr, "/muse/elements/jaw_clench") == 0) {
                    int jaw = args.int32();
                    if (jaw_cb_) jaw_cb_(ts, jaw);
                } else {
                    // ignore other messages
                }
            };

            if (packet.isMessage()) {
                OSCPP::Server::Message m = packet;
                handle_message(m);
            } else if (packet.isBundle()) {
                OSCPP::Server::Bundle b = packet;
                auto ps = b.packets();
                while (!ps.atEnd()) {
                    OSCPP::Server::Packet p = ps.next();
                    if (p.isMessage()) {
                        OSCPP::Server::Message m = p;
                        handle_message(m);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "OSC parse error: " << e.what() << std::endl;
        }
    }

    ::close(sock);
}

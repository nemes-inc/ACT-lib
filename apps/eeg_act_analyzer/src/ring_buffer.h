#ifndef SIMPLE_RING_BUFFER_H
#define SIMPLE_RING_BUFFER_H

#include <vector>
#include <mutex>
#include <algorithm>

// Thread-safe single-producer/single-consumer friendly ring buffer for contiguous window reads.
// Stores POD values (e.g., float). Reads copy out the latest window when available.

template <typename T>
class RingBuffer {
public:
    RingBuffer() : capacity_(0), head_(0), size_(0) {}

    void reset(size_t capacity) {
        std::lock_guard<std::mutex> lock(mu_);
        data_.assign(capacity, T{});
        capacity_ = capacity;
        head_ = 0;
        size_ = 0;
    }

    void push(const T& v) {
        std::lock_guard<std::mutex> lock(mu_);
        if (capacity_ == 0) return;
        data_[head_] = v;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return size_;
    }

    size_t capacity() const {
        return capacity_;
    }

    // Copy latest 'n' elements into out. Returns false if not enough data.
    bool latestWindow(size_t n, std::vector<T>& out) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (n == 0 || n > size_) return false;
        out.resize(n);
        size_t start = (head_ + capacity_ - n) % capacity_;
        if (start + n <= capacity_) {
            std::copy_n(data_.begin() + start, n, out.begin());
        } else {
            size_t first_len = capacity_ - start;
            std::copy_n(data_.begin() + start, first_len, out.begin());
            std::copy_n(data_.begin(), n - first_len, out.begin() + first_len);
        }
        return true;
    }

private:
    mutable std::mutex mu_;
    std::vector<T> data_;
    size_t capacity_;
    size_t head_;
    size_t size_;
};

#endif // SIMPLE_RING_BUFFER_H

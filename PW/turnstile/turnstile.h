#ifndef SRC_TURNSTILE_H_
#define SRC_TURNSTILE_H_

#include <type_traits>
#include <mutex>
#include <queue>
#include <iostream>
#include <condition_variable>
#include <atomic>

const int MTX_ARR_LEN = 255;
const int MIN_TURNSTILES = 32;
class Turnstile;
class Menager;

class Mutex {
 public:
    Mutex();
    Mutex(const Mutex&) = delete;

    void lock();    // NOLINT
    void unlock();  // NOLINT
 private:
    Turnstile *turnstilePtr;
    static Menager menager;
};

class Turnstile {
 public:
    Turnstile();
 private:
    bool ready;
    uint64_t waiting;
    std::mutex mtx;
    std::condition_variable cv;
    friend class Mutex;
};

class Menager {
    uint64_t createdTurnstiles = 0;
    uint64_t unusedTurnstiles = 0;
    std::mutex mtxForTurnstilePtr[MTX_ARR_LEN];
    std::mutex mtxForMenager;
    std::queue<Turnstile*> turnstiles;
    Turnstile* dummyTurnstile;

    Menager();
    ~Menager();
    void deleteFrontTurnstile();
    friend class Mutex;
};

#endif  // SRC_TURNSTILE_H_

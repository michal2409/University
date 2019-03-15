#include <cassert>
#include <vector>

#include "turnstile.h"

Menager Mutex::menager;

Mutex::Mutex() : turnstilePtr(nullptr) {}

void Mutex::lock() {
    menager.mtxForTurnstilePtr[(size_t)this % MTX_ARR_LEN].lock();
    if (turnstilePtr == nullptr) {  // Calling lock on open Mutex.
        turnstilePtr = menager.dummyTurnstile;
    } else {
        menager.mtxForMenager.lock();
        // First lock call after Mutex was closed.
        if (turnstilePtr == menager.dummyTurnstile) {
            // No free turnstiles, creating new ones.
            if (menager.unusedTurnstiles == 0) {
                // Doubling number of created turnstiles.
                for (uint64_t i = 0; i < menager.createdTurnstiles; i++) {
                    menager.turnstiles.push(new Turnstile);
                }
                menager.unusedTurnstiles = menager.createdTurnstiles;
                menager.createdTurnstiles *= 2;
            }
            turnstilePtr = menager.turnstiles.front();
            menager.turnstiles.pop();
            menager.unusedTurnstiles--;
        }
        menager.mtxForMenager.unlock();
        turnstilePtr->waiting++;
        menager.mtxForTurnstilePtr[(size_t)this % MTX_ARR_LEN].unlock();

        std::unique_lock<std::mutex> lk(turnstilePtr->mtx);
        turnstilePtr->cv.wait(lk, [&]{return turnstilePtr->ready;});

        menager.mtxForTurnstilePtr[(size_t) this % MTX_ARR_LEN].lock();
        turnstilePtr->ready = false;
    }
    menager.mtxForTurnstilePtr[(size_t) this % MTX_ARR_LEN].unlock();
}

void Mutex::unlock() {
    menager.mtxForTurnstilePtr[(size_t)this % MTX_ARR_LEN].lock();
    // Calling unlock before someone takes turnstile.
    if (turnstilePtr == menager.dummyTurnstile) {
        turnstilePtr = nullptr;
        // No one waits on turnstile, giving back the turnstile.
    } else if (turnstilePtr->waiting == 0) {
        menager.mtxForMenager.lock();
        menager.unusedTurnstiles++;
        menager.turnstiles.push(turnstilePtr);
        // There are less than 1/4 of created turnstiles in use
        if (menager.createdTurnstiles > MIN_TURNSTILES &&
                4 * menager.unusedTurnstiles > 3 * menager.createdTurnstiles) {
            uint64_t halfOfCreatedTurnstiles = menager.createdTurnstiles / 2;
            // Removing one half of unusing turnstiles
            for (uint64_t i = 0; i < halfOfCreatedTurnstiles; i++) {
                menager.deleteFrontTurnstile();
            }
            menager.unusedTurnstiles -= halfOfCreatedTurnstiles;
            menager.createdTurnstiles -= halfOfCreatedTurnstiles;
        }
        menager.mtxForMenager.unlock();
        turnstilePtr = nullptr;
    } else {  // Someone waits on turnstile.
        turnstilePtr->waiting--;
        std::unique_lock<std::mutex> lk(turnstilePtr->mtx);
        turnstilePtr->ready = true;
        turnstilePtr->cv.notify_one();
    }
    menager.mtxForTurnstilePtr[(size_t)this % MTX_ARR_LEN].unlock();
}

Turnstile::Turnstile() : ready(false), waiting(0) {}

Menager::Menager() {
    for (int i = 0; i < MIN_TURNSTILES; i++)
        turnstiles.push(new Turnstile);
    createdTurnstiles = unusedTurnstiles = MIN_TURNSTILES;
    dummyTurnstile = new Turnstile;
}

Menager::~Menager() {
    while (!turnstiles.empty()) {
        deleteFrontTurnstile();
    }
    delete dummyTurnstile;
}

void Menager::deleteFrontTurnstile() {
    Turnstile *t = turnstiles.front();
    turnstiles.pop();
    delete t;
}

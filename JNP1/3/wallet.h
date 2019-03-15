#ifndef WALLET_H
#define WALLET_H

#include <cstdint>
#include <vector>
#include <chrono>
#include <string>
#include <boost/operators.hpp>

class Operation;

class Wallet : boost::ordered_field_operators<Wallet> {
public:
    Wallet();
    Wallet(int n);
    explicit Wallet(const char* str);
    explicit Wallet(const std::string& str);
    Wallet(Wallet && w);
    Wallet(Wallet &&w1, Wallet &&w2);
    ~Wallet();

    static Wallet fromBinary(const char* s);
    uint64_t getUnits() const;
    uint64_t opSize() const;

    const Operation& operator[](size_t index) const;

    Wallet& operator=(Wallet&& w);
    Wallet& operator+=(Wallet& w);
    Wallet& operator+=(Wallet&& w);
    Wallet& operator-=(Wallet& w);
    Wallet& operator-=(Wallet&& w);
    Wallet& operator*=(const int n);
    template <typename T> Wallet& operator*=(T t) = delete;

    friend Wallet operator+(Wallet&& w1, Wallet&& w2);
    friend Wallet operator+(Wallet&& w1, Wallet& w2);
    friend Wallet operator-(Wallet&& w1, Wallet&& w2);
    friend Wallet operator-(Wallet&& w1, Wallet& w2);
    friend Wallet operator*(Wallet&& w, int n);
    friend Wallet operator*(Wallet& w, int n);
    friend Wallet operator*(int n, Wallet&& w);
    friend Wallet operator*(int n, Wallet& w);
    template <typename T> friend Wallet operator*(T t, Wallet& w) = delete;
    template <typename T> friend Wallet operator*(Wallet& w, T t) = delete;

    friend std::ostream& operator<<(std::ostream &os, const Wallet &w);

private:
    template<typename T> Wallet(T t);
    uint64_t unitsNumber;
    std::vector<Operation> operations;
    void createOperation();
    static uint64_t numberCoinsInUse;
    static void validateCoins(uint64_t unitsToAdd);
};

class Operation : boost::ordered_field_operators<Operation> {
public:
    Operation() {};
    Operation(uint64_t units, uint64_t milisec, int year, int month, int day);
    friend std::ostream& operator<<(std::ostream &os, const Operation &o);
    uint64_t getUnits() const;
    uint64_t getMilisec() const;
private:
    uint64_t unitsAfterOperation;
    uint64_t milisec;
    int year;
    int month;
    int day;
};

bool operator==(const Wallet& w1, const Wallet& w2);
bool operator<(const Wallet& w1, const Wallet& w2);
bool operator==(const Operation& o1, const Operation& o2);
bool operator<(const Operation& o1, const Operation& o2);

const Wallet& Empty();

#endif // WALLET_H


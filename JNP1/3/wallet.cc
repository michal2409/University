#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <regex>
#include <boost/lexical_cast.hpp>
#include "wallet.h"

namespace {
    using Clock = std::chrono::system_clock;
    using operationsVec = std::vector<Operation>;
    using std::runtime_error;
    using std::string;
    using std::move;
    static const std::regex numRegex (R"(^\s*(0|[1-9]\d*)([,.](\d{1,8}))?\s*$)");
    static const std::regex binaryNumRegex (R"(^(0|1[01]*)$)");
    static const uint64_t UNIT = 1e8;
    static const uint64_t COINS_LIMIT = 2.1e15;
    static const int PRECISION = 8;

    int pow10[PRECISION] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};

    void checkAdditionOverflow(uint64_t x, uint64_t y) {
        if (x + y < x)
            runtime_error("Addition oveflow");
    }

    void checkMultOverlow(uint64_t x, uint64_t y) {
        uint64_t m = x * y;
        if (x != 0 && m / x != y)
            throw runtime_error("Multiplication overflow");
    }

    void checkSubtraction(uint64_t a, uint64_t b) {
        if (a < b)
            throw runtime_error("Wallet has a negative number of B.");
    }

    void isNatural(int n) {
        if (n < 0)
            throw runtime_error("Given number in parameter was not natural.");
    }

    /* Parsuje z napisu str liczbe zapisana w bazie base na liczbe calkowita */
    uint64_t parseNumber(const string& str, int base) {
        uint64_t number = 0;
        for (int i = 0; isdigit(str[i]); ++i) {
            checkMultOverlow(number, base);
            number = base*number + (str[i]-'0');
        }
        return number;
    }

    uint64_t parseString(const string& str, int base) {
        const std::regex& reg = (base == 2) ? binaryNumRegex : numRegex;
        std::smatch m;
        if (!std::regex_match(str, m, reg))
            throw runtime_error("Invalid str argument");
        uint64_t number = parseNumber(m[1], base);
        number *= UNIT;
        if (m.size() == 4) // Liczba z przecinkiem lub kropka, dodanie czesci ulamkowej.
            number += parseNumber(m[3], base)*pow10[PRECISION - m[3].length()];
        return number;
    }

    string getFraction(uint64_t unitsNumber) {
        string units = boost::lexical_cast<string>(unitsNumber);
        int unitsLength = units.length();
        string fraction;
        if (unitsLength < PRECISION) {
            int diff = PRECISION - unitsLength;
            string zeroes;
            for (int i = 0; i < diff; i++)
                zeroes += '0';
            fraction += zeroes + units;
        }
        else
            fraction = units.substr(unitsLength - PRECISION, PRECISION);
        int length = PRECISION;
        for (int i = PRECISION - 1; i >= 0 && fraction[i] == '0'; i--)
            length--;
        return fraction.substr(0, length);
    }
}

/* Liczba dostepnych jednostek B */
uint64_t Wallet::numberCoinsInUse = 0;

/* Tworzy pusty portfel. Historia portfela ma jeden wpis. */
Wallet::Wallet() : unitsNumber(0) {
    createOperation();
}

/* Tworzy portfel z n B, gdzie n jest liczbą
   naturalną. Historia portfela ma jeden wpis. */
Wallet::Wallet(int n) {
    isNatural(n);
    uint64_t unitsToAdd = n * UNIT;
    validateCoins(unitsToAdd);
    unitsNumber = unitsToAdd;
    createOperation();
}

/* Tworzy portfel na podstawie napisu str określającego ilość B.
   Napis może zawierać część ułamkową (do 8 miejsc) oddzieloną
   przecinkiem lub kropką. Historia portfela ma jeden wpis. */
Wallet::Wallet(const string& str) {
    uint64_t unitsToAdd = parseString(str, 10);
    validateCoins(unitsToAdd);
    unitsNumber = unitsToAdd;
    createOperation();
}

Wallet::Wallet(const char* str) : Wallet(string(str)) {}

/* Tworzy portfel z historią operacji jak w portfelu w z jedenym nowym wpisem.
   Stan B jest jak w portfelu w. Po operacji portfel w jest pusty */
Wallet::Wallet(Wallet && w) : unitsNumber(w.unitsNumber), operations(move(w.operations)) {
    createOperation();
    w.unitsNumber = 0;
    w.operations.clear();
}

/* Tworzy portfel, którego historia operacji to suma historii operacji w1
   i w2 plus jeden wpis, całość uporządkowana wg czasów wpisów. Po operacji
   w portfelu jest w1.getUnits() + w2.getUnits() jednostek, a portfele w1 i w2 są puste. */
Wallet::Wallet(Wallet &&w1, Wallet &&w2) {
    checkAdditionOverflow(w1.unitsNumber, w2.unitsNumber);
    unitsNumber = w1.unitsNumber + w2.unitsNumber;

    operationsVec v = operationsVec(w1.opSize() + w2.opSize()); // Wektor na scalanie operacji
    auto v1 = w1.operations, v2 = w2.operations;
    std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), v.begin());
    operations = v;
    createOperation();

    w1.unitsNumber = w2.unitsNumber = 0;
    w1.operations.clear();
    w2.operations.clear();
}

/* Usuwa B z portfela dodając je do puli
   dostępnych B oraz czyści historię operacji.*/
Wallet::~Wallet() {
    Wallet::numberCoinsInUse -= unitsNumber;
    operations.clear();
}

/* Tworzy portfel na podstawie napisu str, który jest zapisem
   ilości B w systemie binarnym. Kolejność bajtów jest grubokońcówkowa. */
Wallet Wallet::fromBinary(const char* str) {
    uint64_t unitsToAdd = parseString(str, 2);
    validateCoins(unitsToAdd);
    Wallet w;
    w.unitsNumber = unitsToAdd;
    return w;
}

Operation::Operation(uint64_t units, uint64_t milisec, int year, int month, int day) : unitsAfterOperation(units),
                                                        milisec(milisec), year(year), month(month), day(day) {}

/* Tworzy i dodaje operację do historii operacji portfela. */
void Wallet::createOperation() {
    auto now = Clock::now();
    uint64_t milisec = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count(); // Czas utworzenia w milisekundach.
    std::time_t now_c = Clock::to_time_t(now);
    struct tm *parts = std::localtime(&now_c); // Trzyma rok, miesiac, dzien.
    operations.emplace_back(Operation(unitsNumber, milisec, 1900 + parts->tm_year, parts->tm_mday, 1 + parts->tm_mon));
}

/* W przypadku przekoroczenia liczby dostępnych B po dodaniu unitsToAdd B rzuca wyjątkiem.
   W przeciwnym razie dodaje do puli używanych B unitsToAdd B */
void Wallet::validateCoins(uint64_t unitsToAdd) {
    checkAdditionOverflow(Wallet::numberCoinsInUse, unitsToAdd);
    if (Wallet::numberCoinsInUse + unitsToAdd > COINS_LIMIT)
        throw runtime_error("The number of all B has exceeded 21 million");
    Wallet::numberCoinsInUse += unitsToAdd;
}

uint64_t Wallet::getUnits() const {
    return unitsNumber;
}

uint64_t Wallet::opSize() const {
    return operations.size();
}

uint64_t Operation::getUnits() const {
    return unitsAfterOperation;
}

uint64_t Operation::getMilisec() const {
    return milisec;
}

/* Jeżeli portfele this i w są tym samym obiektem, to nic nie robi, wpp.
   historia operacji this to historia operacji w i jeden nowy wpis.
   Po operacji portfel w jest pusty. */
Wallet& Wallet::operator=(Wallet&& w) {
    if (this != &w) {
        numberCoinsInUse -= unitsNumber;
        unitsNumber = w.unitsNumber;
        w.unitsNumber = 0;
        operations = move(w.operations);
        createOperation();
    }
    return *this;
}

/* Po operacji portfel w ma 0 B i dodatkowy wpis w historii, a portfel this ma
   w1.getUnits() + w2.getUnits() jednostek i jeden dodatkowy wpis w historii. */
Wallet& Wallet::operator+=(Wallet&& w) {
    if (this == &w)
        throw runtime_error("Undefined operation += when called for the same object");
    checkAdditionOverflow(unitsNumber, w.unitsNumber);
    unitsNumber += w.unitsNumber;
    createOperation();
    w.unitsNumber = 0;
    w.createOperation();
    return *this;
}

Wallet& Wallet::operator+=(Wallet& w) {
    return *this += move(w);
}

/* Po operacji portfel w ma dwa razy więcej jednostek, niż było w nim przed odejmowaniem i dodatkowy wpis w historii,
   a portfel this ma w1.getUnits() - w2.getUnits() jednostek i jeden dodatkowy wpis w historii. */
Wallet& Wallet::operator-=(Wallet&& w) {
    if (this == &w)
        throw runtime_error("Undefined operation += when called for the same object");
    checkSubtraction(unitsNumber, w.unitsNumber);
    unitsNumber -= w.unitsNumber;
    createOperation();
    w.unitsNumber *= 2;
    w.createOperation();
    return *this;
}

Wallet& Wallet::operator-=(Wallet& w) {
    return *this -= move(w);
}

/* Pomnożenie zawartości portfela przez liczbę naturalną.
   Dodaje jeden wpis w historii. */
Wallet& Wallet::operator*=(const int n) {
    isNatural(n);
    if (n > 1)
        checkMultOverlow(unitsNumber, n - 1);
    uint64_t unitsToAdd = unitsNumber * (n - 1);
    validateCoins(unitsToAdd);
    unitsNumber += unitsToAdd;
    createOperation();
    return *this;
}

Wallet operator+(Wallet&& w1, Wallet&& w2) {
    checkAdditionOverflow(w1.unitsNumber, w2.unitsNumber);
    Wallet::validateCoins(w1.unitsNumber); // Monety w w2 sa zerowane w rezultacie nie musimy ich dodawac.
    Wallet w;
    w.unitsNumber = w1.unitsNumber + w2.unitsNumber;
    w.createOperation();
    w2.unitsNumber = 0;
    w2.createOperation();
    return w;
}

Wallet operator+(Wallet&& w1, Wallet& w2) {
    return move(w1) + move(w2);
}

Wallet operator-(Wallet&& w1, Wallet&& w2) {
    checkSubtraction(w1.unitsNumber, w2.unitsNumber);
    Wallet w;
    w.unitsNumber = w1.unitsNumber - w2.unitsNumber;
    w.createOperation();
    w2.unitsNumber *= 2;
    w2.createOperation();
    w1.unitsNumber = 0;
    w1.createOperation();
    return w;
}

Wallet operator-(Wallet&& w1, Wallet& w2) {
    return move(w1) - move(w2);
}

Wallet operator*(Wallet&& w, int n) {
    isNatural(n);
    checkMultOverlow(w.unitsNumber, n);
    uint64_t unitsToAdd = w.unitsNumber * n;
    Wallet::validateCoins(unitsToAdd);
    Wallet w1;
    w1.unitsNumber = unitsToAdd;
    w1.createOperation();
    return w1;
}

Wallet operator*(int n, Wallet&& w) {
    return move(w) * n;
}

Wallet operator*(int n, Wallet& w) {
    return move(w) * n;
}

Wallet operator*(Wallet& w, int n) {
    return move(w) * n;
}

/* Zwraca obiekt reprezentujący pusty
   portfel, obiektu nie mozna modyfikowac */
const Wallet& Empty() {
    static Wallet empty;
    return empty;
}

std::ostream& operator<<(std::ostream &os, const Operation &op) {
    string fraction = getFraction(op.unitsAfterOperation);
    os << "Wallet balance is " << op.unitsAfterOperation / UNIT;
    if (fraction.length() != 0)
        os  << "," << fraction;
    os << " B after operation made at day " << op.year << "-" << op.month << "-" << op.day;
    return os;
}

std::ostream& operator<<(std::ostream &os, const Wallet &w) {
    string fraction = getFraction(w.getUnits());
    os << "Wallet[" << w.getUnits() / UNIT;
    if (fraction.length() != 0)
        os << "," << fraction;
    os << " B]";
    return os;
}

const Operation& Wallet::operator[](size_t index) const {
    return operations[index];
}

bool operator==(const Wallet& w1, const Wallet& w2) {
    return w1.getUnits() == w2.getUnits();
}

bool operator<(const Wallet& w1, const Wallet& w2) {
    return w1.getUnits() < w2.getUnits();
}

bool operator==(const Operation& o1, const Operation& o2) {
    return o1.getMilisec() == o2.getMilisec();
}

bool operator<(const Operation& o1, const Operation& o2) {
    return o1.getMilisec() < o2.getMilisec();
}
#include <iostream>
#include <set>
#include <map>
#include <regex>
#include <sstream>

using namespace std;

static const char CIRCUIT_ELEMENTS[] = {'T', 'D', 'R', 'C', 'E'};
static const regex REGEX_DERC (R"(^\s*[DERC](0|[1-9]\d{0,8})\s+(\d|[A-Z])[a-zA-Z\d,/-]*\s+(0|[1-9]\d{0,8})\s+(0|[1-9]\d{0,8})\s*$)");
static const regex REGEX_T (R"(^\s*T(0|[1-9]\d{0,8})\s+(\d|[A-Z])[a-zA-Z\d,/-]*\s+(0|[1-9]\d{0,8})\s+(0|[1-9]\d{0,8})\s+(0|[1-9]\d{0,8})\s*$)");

struct designationComp{
    bool operator() (const string& s1, const string& s2) const {
        long idx1 = distance(CIRCUIT_ELEMENTS, find(begin(CIRCUIT_ELEMENTS), end(CIRCUIT_ELEMENTS), s1[0]));
        long idx2 = distance(CIRCUIT_ELEMENTS, find(begin(CIRCUIT_ELEMENTS), end(CIRCUIT_ELEMENTS), s2[0]));
        if (idx1 == idx2)
            return stoi(s1.substr(1, s1.length())) < stoi(s2.substr(1, s2.length()));
        return idx1 < idx2;
    }
};

void reportError(const string &line, const long long lineNum) {
    cerr << "Error in line " << lineNum << ": " << line << endl;
}

void reportWarning(const map<long long, long long>& nodeToCountMap) {
    bool first = true;
    for (auto &it : nodeToCountMap) {
        if (it.second < 2) {
            if (first) {
                cerr << "Warning, unconnected node(s): " << it.first;
                first = false;
            }
            else
                cerr << ", " << it.first;
        }
    }
    if (!first)
        cerr << endl;
}

bool parseLine(const string& line, map<long long, long long>& nodeToCountMap, map<string, string, designationComp>& desigToTypeMap,
                                                                map<string, set<string, designationComp>>& typeToDesigsMap) {
    if (line.empty()) // Program ignoruje puste wiersze.
        return true;

    // Sprawdzenie poprawności linii.
    if (!regex_match(line, REGEX_T) && !regex_match(line, REGEX_DERC))
        return false;

    stringstream lineStream;
    lineStream.str(line);

    string designation, type;
    lineStream >> designation >> type;

    set<long long> nodesNum;
    int numbersToRead = (designation[0] == 'T') ? 3 : 2;
    for (int i = 0; i < numbersToRead; i++) { // Wczytanie numerów węzłów do zbioru.
        long long nodeNum;
        lineStream >> nodeNum;
        nodesNum.insert(nodeNum);
    }

    if (nodesNum.size() == 1) // Wszystkie końcówki danego elementu nie mogą być podłączone do jednego węzła.
        return false;

    if (desigToTypeMap.find(designation) != desigToTypeMap.end()) // Oznaczenia elementów w obwodzie nie mogą się powtarzać.
        return false;

    desigToTypeMap[designation] = type; // Dodanie oznaczenia do obwodu.

    if (typeToDesigsMap.find(type) == typeToDesigsMap.end()) // Wczytany typ ma pusty zbiór oznaczeń.
        typeToDesigsMap[type] = set<string, designationComp>();
    typeToDesigsMap[type].insert(designation);

    for (long long i : nodesNum) { // Aktualizacja liczby występowań węzłów.
        if (nodeToCountMap.find(i) == nodeToCountMap.end())
            nodeToCountMap[i] = 1;
        else
            nodeToCountMap[i] += 1;
    }

    return true;
}

int main() {
    map<long long, long long> nodeToCountMap;
    map<string, string, designationComp> desigToTypeMap;
    map<string, set<string, designationComp>> typeToDesigsMap;
    string line;
    long long lineNum = 1;
    nodeToCountMap[0] = 0;

    while (getline(cin, line)) {
        if (!parseLine(line, nodeToCountMap, desigToTypeMap, typeToDesigsMap))
            reportError(line, lineNum);
        lineNum += 1;
    }

    while (!desigToTypeMap.empty()) { // Wypisywanie wyjścia.
        string type = desigToTypeMap.begin()->second;
        char desigChar = desigToTypeMap.begin()->first[0];
        bool first = true;

        // Ze zbioru oznaczeń wypisujemy wszystkie mające pierwszą litere oznaczenia jak desigChar.
        for (auto it = typeToDesigsMap[type].begin(); it != typeToDesigsMap[type].end(); ) {
            if ((*it)[0] != desigChar)
                break;
            if (first) {
                cout << *it;
                first = false;
            }
            else
                cout << ", " << *it;

            // Usuniecie oznaczenia z mapy i ze zbioru.
            desigToTypeMap.erase(*it);
            it = typeToDesigsMap[type].erase(it);
        }
        cout << ": " << type << endl;
    }

    reportWarning(nodeToCountMap);
    return 0;
}

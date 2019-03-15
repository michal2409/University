#include <iostream>
using namespace std;

int N = 300000;

long long p = 1e9 + 7;
long long base = 37;

inline int abs(int a) {
	return (a >= 0) ? a : (-1)*a;
}

inline long long getHash(long long pref[], long long basePow[], int i, int j) {
	long long hash = (pref[j] - pref[i-1] * basePow[j-i+1]) % p;
	return (hash >= 0) ? hash : hash + p;
}

bool comp(int arr[], long long pref[], long long basePow[], int a, int b, int c, int d) {
	int len1 = b-a+1, len2 = d-c+1, diff = abs(len1 - len2);
	bool skrA = false;
	if (len1 > len2) {
		b -= diff;
		skrA = true;
	}
	else if (len2 > len1)
		d -= diff;
	while (a < b) {
		int sr1 = (a+b)/2, sr2 = (c+d)/2;
		if (getHash(pref, basePow, a, sr1) == getHash(pref, basePow, c, sr2)) {
			a = sr1+1;
			c = sr2+1;
		}
		else {
			b = sr1;
			d = sr2;
		}
	}
	if (arr[a] == arr[c])
		return !skrA;
	return arr[a] < arr[c];
}

int main() {
	long long basePow[N+1];
	basePow[0] = 1;
	for (int i = 1; i <= N; i++)
		basePow[i] = (base*basePow[i-1]) % p;
	int n, m;
	cin >> n >> m;
	int arr[n+1]; arr[0] = 0;
	for (int i = 1; i <= n; i++) {
		char c;
		cin >> c;
		arr[i] = c - 'a' + 1;
	}
	long long pref[n+1]; pref[0] = 0;
	for (int i = 1; i <= n; i++) // uzupelnianie prefixow
		pref[i] = (base*pref[i-1]+arr[i]) % p;

	for (int i = 0; i < m; i++) {
		int a, b, c, d;
		cin >> a >> b >> c >> d;
		// cout << "hash1: " << getHash(pref, basePow, a, b) << " hash2: " << getHash(pref, basePow, c, d) << endl;
		if (getHash(pref, basePow, a, b) == getHash(pref, basePow, c, d))
			cout << "=\n";
		else if (comp(arr, pref, basePow, a, b, c, d))
			cout << "<\n";
		else
			cout << ">\n";
	}
	return 0;
}
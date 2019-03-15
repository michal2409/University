#include <iostream>

using namespace std;

int n, k, u;
const int wiersze = 17;
int potegi[wiersze] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};

bool skok(int idx_skoku, int czas, int skocznosc, int** dp) {
	if (n <= idx_skoku || 0 > idx_skoku) return false;

	if (skocznosc >= wiersze - 1) {
		int t = dp[wiersze - 1][idx_skoku];
		if (t == 0 || (t != 0 && czas < t)) {
			dp[wiersze - 1][idx_skoku] = czas;
			return true;
		}
		return false;
	}
	
	int t = dp[skocznosc][idx_skoku];
	if (t == 0 || (t != 0 && czas < t)) {
		dp[skocznosc][idx_skoku] = czas;
		return true;
	}

	return false;
}

void uzupelnij(int idx_start, int idx_koniec, char znaki[], int** dp) {
	for (int j = 0; j < wiersze; j++) {
		for (int i = idx_start; i < idx_koniec - 1; i++) {
			if (dp[j][i] == 0 || znaki[i] == '#') continue;

			switch (znaki[i]) {
				case '.':
					skok(i + 1, dp[j][i] + 1, j, dp);
					skok(i + potegi[j], dp[j][i] + 1, j, dp);
				break;

				case '*':
					skok(i + 1, dp[j][i] + 1, j+1, dp);
					skok(i + potegi[j+1], dp[j][i] + 1, j+1, dp);
				break;

				case '<':
					if (skok(i - k, dp[j][i], j, dp))
						i = i-k-1; // i-k w nast iteracji
				break;

				case '>':
					skok(i + k, dp[j][i], j, dp);
				break;

				case 'U':
					skok(i + 1, dp[j][i] + u + 1, j, dp);
					skok(i + potegi[j], dp[j][i] + u + 1, j, dp);
				break;
			}
		}
	}
}

int main() {
	cin >> n >> k >> u;

	char znaki[n];
	for (int i = 0; i < n; i ++) 
		cin >> znaki[i];

	if (n == 1) {
		cout << 0;
		return 0;	
	} 
	if (n == 2) {
		cout << 1;
		return 0;
	}

	int **dp = new int *[wiersze];
	for (int i = 0; i < wiersze; i++) {
		dp[i] = new int[n];
		for (int j = 0; j < n; j++)
			dp[i][j] = 0;
	}
	dp[0][1] = dp[0][2] = 1;

	uzupelnij(0, n, znaki, dp);

	bool init = false;
	int min = -1; 
	for (int i = 0; i < wiersze; i++) {
		if (dp[i][n-1] > 0) {
			if (!init) {
				init = true;
				min = dp[i][n-1];
			} 
			else if (dp[i][n-1] < min)
				min = dp[i][n-1];
		}
	}

	cout << min;
	return 0;
}
#include <iostream>
using namespace std;

long long rek(long long ***dp, int ciag[], int idxp, int idxk, int cyfra, int idx3) {
	if (idxp == idxk) {
		if (idx3 == 1 && cyfra > ciag[idxp]) 
			return dp[idxp][idxk][idx3] = 1;
		if (idx3 == 0 && cyfra < ciag[idxp])
			return dp[idxp][idxk][idx3] = 1;
		return dp[idxp][idxp][idx3] = 0;
	}

	int a, b;
	if (idx3 == 1) {
		a = cyfra > ciag[idxp] ? (dp[idxp + 1][idxk][0] != -1 ? dp[idxp + 1][idxk][0] : rek(dp, ciag, idxp+1, idxk, ciag[idxp], 0)) : 0;
		b = cyfra > ciag[idxk] ? (dp[idxp][idxk - 1][1] != -1 ? dp[idxp][idxk - 1][1] : rek(dp, ciag, idxp, idxk-1, ciag[idxk], 1)) : 0;
	}
	else {
		a = cyfra < ciag[idxp] ? (dp[idxp + 1][idxk][0] != -1 ? dp[idxp + 1][idxk][0] : rek(dp, ciag, idxp+1, idxk, ciag[idxp], 0)) : 0;
		b = cyfra < ciag[idxk] ? (dp[idxp][idxk - 1][1] != -1 ? dp[idxp][idxk - 1][1] : rek(dp, ciag, idxp, idxk-1, ciag[idxk], 1)) : 0;
	}

	return dp[idxp][idxk][idx3] = (a + b) % 1000000000;
}

int main() {
	ios_base::sync_with_stdio(0);

	int n;
	cin >> n;
	if (n == 1) {
		cout << 1;
		return 0;
	}
	
	int ciag[n];
	for (int i = 0; i < n; i++)
		cin >> ciag[i];

	long long ***dp = new long long**[n];
	for (int i = 0; i < n; i++) {
		dp[i] = new long long *[n];
		for (int j = 0; j < n; j++) {
			dp[i][j] = new long long [2];
			dp[i][j][0] = dp[i][j][1] = -1;
		}
	}

	rek(dp, ciag, 0, n-2, ciag[n-1], 1); 
	rek(dp, ciag, 1, n-1, ciag[0], 0);

	cout << (dp[0][n-2][1] + dp[1][n-1][0]) % 1000000000;
}
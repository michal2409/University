#include <iostream>
using namespace std;

int main() {
  ios_base::sync_with_stdio(0);

  long long int n, m;
  cin >> n;

  long long int ceny[n];
  for (long long int i = 0; i < n; i++)
    cin >> ceny[i];
  cin >> m;

  long long int suffixCena[n], maxParzPref[n], minParzSuff[n], maxNiepPref[n], minNiepSuff[n];

  suffixCena[n-1] = ceny[n-1];
  for (long long int i = n-2; i >= 0; i--)
    suffixCena[i] = suffixCena[i+1] + ceny[i];

  if (ceny[0] % 2 == 0) {
    maxParzPref[0] = ceny[0];
    maxNiepPref[0] = -1;
  }
  else {
    maxNiepPref[0] = ceny[0];
    maxParzPref[0] = -1;
  }

  for (long long int i = 1; i < n; i++) {
    if (ceny[i] % 2 == 0) {
      maxParzPref[i] = ceny[i];
      maxNiepPref[i] = maxNiepPref[i-1];
    }
    else {
      maxNiepPref[i] = ceny[i];
      maxParzPref[i] = maxParzPref[i-1];
    }
  }

  if (ceny[n-1] % 2 == 0) {
    minParzSuff[n-1] = ceny[n-1];
    minNiepSuff[n-1] = -1;
  }
  else {
    minNiepSuff[n-1] = ceny[n-1];
    minParzSuff[n-1] = -1;
  }

  for (long long int i = n-2; i >= 0; i--) {
    if (ceny[i] % 2 == 0) {
      minParzSuff[i] = ceny[i];
      minNiepSuff[i] = minNiepSuff[i+1];
    }
    else {
      minParzSuff[i] = minParzSuff[i+1];
      minNiepSuff[i] = ceny[i];
    }
  }

  long long int k , zamiana[n-1];
  zamiana[0] = -1;
  for (long long int i = 1; i < n; i++) {
    long long int strata1 = -1, strata2 = -1;
    if (minParzSuff[i] != -1 && maxNiepPref[i-1] != -1)
      strata1 = minParzSuff[i] - maxNiepPref[i-1];
    if (minNiepSuff[i] != -1 && maxParzPref[i-1] != -1)
      strata2 = minNiepSuff[i] - maxParzPref[i-1];

    if (strata1 == -1 && strata2 == -1)
      zamiana[i] = -1;
    else if (strata1 == -1)
      zamiana[i] = strata2;
    else if (strata2 == -1)
      zamiana[i] = strata1;
    else {
      if (strata1 < strata2)
        zamiana[i] = strata1;
      else
        zamiana[i] = strata2;
    }
  }

  for (long long int i = 0; i < m; i++) {
    cin >> k;
    if (suffixCena[n-k] % 2 == 1)
      cout << suffixCena[n-k] << endl;
    else if (zamiana[n-k] == -1)
      cout << -1 << endl;
    else 
      cout << suffixCena[n-k] - zamiana[n-k] << endl;
  }
}

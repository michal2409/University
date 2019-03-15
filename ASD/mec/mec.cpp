#include <iostream>
#include <algorithm>

using namespace std;

int main() {
  int n, m;
  cin >> n >> m;

  long long maska[n];
  for (int i = 0; i < n; i++)
    maska[i] = 1;

  long long wykl = 1;
  int nr;
  for (int i = 0; i < m; i++) {
    wykl *= 2;
    for (int j = 0; j < n; j++) {
      cin >> nr;
      if (j >= n / 2)
        maska[nr-1] += wykl;
    }
  }

  sort(maska, maska + n);
  for (int i = 1; i < n; i++) {
    if (maska[i-1] == maska[i]) {
      cout << "NIE";
      return 0;
    }
  }
  cout << "TAK";
  return 0;
}
#include <iostream>
#include <vector>
#include <queue>
#include <utility>
using namespace std;

int main() {
  ios_base::sync_with_stdio(0);
  int n, m, k;
  cin >> n >> m >> k;
  int prac[n];
  for (int i = 0; i < n; i++)
    cin >> prac[i];
  vector<int> v1[n];
  vector<int> v2[n];
  for (int i = 0; i < m; i++) {
    int a, b;
    cin >> a >> b;
    a--; b--;
    v1[a].push_back(b);
    v2[b].push_back(a);
  }
  priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
  int outDeg[n];
  for (int i = 0; i < n; i++) {
    outDeg[i] = v1[i].size();
    if (outDeg[i] == 0)
      q.push(pair<int, int>(prac[i], i));
  }
  int maxi = 0;
  for (int i = 0; i < k; i++) {
    pair<int, int> p = q.top();
    q.pop();
    int idx = p.second;
    maxi = (p.first > maxi) ? p.first : maxi;

    for (auto it = v2[idx].begin(); it != v2[idx].end(); ++it) {
      outDeg[*it]--;
      if (outDeg[*it] == 0)
        q.push(pair<int, int>(prac[*it], *it));
    }
  }
  cout << maxi << endl;
}
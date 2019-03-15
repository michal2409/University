#include <iostream>
#include <cmath>
using namespace std;

void calcDepth(int curr, int prev, int left[], int right[], int depth[], int parent[]) {
  int lewy = left[curr];
  int prawy = right[curr];
  parent[curr] = prev;
  if (lewy != -2) {
    depth[lewy] = depth[curr] + 1;
    calcDepth(lewy, curr, left, right, depth, parent);
  }
  if (prawy != -2) {
    depth[prawy] = depth[curr] + 1;
    calcDepth(prawy, curr, left, right, depth, parent);
  }
}

int ancestor(int v, int h, int n, int** links) {
  int res = v, i = floor(log2(n));
  while (h > 0) {
    int p = int(pow(2, i));
    if (p > h)
      i = i - 1;
    else {
      res = links[i][res];
      h = h - p;
    }
  }
  return res;
}

int lca(int u, int v, int depth[], int** links, int parent[], int n) {
  int du = depth[u], dv = depth[v];
  if (du < dv) {
    v = ancestor(v, dv-du, n, links);
    dv = depth[v];
  }
  else if (du > dv) {
    u = ancestor(u, du-dv, n, links);
    du = depth[u];
  }
  if (u==v)
    return u;

  int i = floor(log2(n)) - 1;
  while (i >= 0) {
    if (links[i][u] != links[i][v]) {
      u = links[i][u];
      v = links[i][v];
    }
    i = i-1;
  }
  return parent[u];
}

void bottomUp(int curr, int fardownV[], int fardownD[], int left[], int right[]) {
  if (left[curr] == -2 && right[curr] == -2) {
    fardownV[curr] = curr;
    fardownD[curr] = 0;
  }
  else if (left[curr] == -2) {
    bottomUp(right[curr], fardownV, fardownD, left, right);
    fardownV[curr] = fardownV[right[curr]];
    fardownD[curr] = fardownD[right[curr]]+1;
  }
  else if (right[curr] == -2) {
    bottomUp(left[curr], fardownV, fardownD, left, right);
    fardownV[curr] = fardownV[left[curr]];
    fardownD[curr] = fardownD[left[curr]]+1;
  }
  else {
    bottomUp(left[curr], fardownV, fardownD, left, right);
    bottomUp(right[curr], fardownV, fardownD, left, right);
    if (fardownD[right[curr]] > fardownD[left[curr]]) {
      fardownD[curr] = fardownD[right[curr]] + 1;
      fardownV[curr] = fardownV[right[curr]];
    }
    else {
      fardownD[curr] = fardownD[left[curr]] + 1;
      fardownV[curr] = fardownV[left[curr]];
    }
  }
}

void upDown(int curr, int farUpD[], int farUpV[], int fardownV[], int fardownD[],
                                  int parent[], int left[], int right[], bool lefish) {
  if (parent[curr] == -1) {
    farUpV[curr] = curr;
    farUpD[curr] = 0;
  }
  else {
    int d1 = farUpD[parent[curr]] + 1;
    int v1 = farUpV[parent[curr]];
    int d2 = -1;
    int v2;
    if (lefish && right[parent[curr]] != -2) {
      d2 = fardownD[right[parent[curr]]] + 2;
      v2 = fardownV[right[parent[curr]]];
    }
    else if (!lefish && left[parent[curr]] != -2) {
      d2 = fardownD[left[parent[curr]]] + 2;
      v2 = fardownV[left[parent[curr]]];
    }
    if (d2 > d1) {
      farUpD[curr] = d2;
      farUpV[curr] = v2;
    }
    else {
      farUpV[curr] = v1;
      farUpD[curr] = d1;
    }
  }
  if (left[curr] != -2)
    upDown(left[curr], farUpD, farUpV, fardownV, fardownD, parent, left, right, true);
  if (right[curr] != -2)
    upDown(right[curr], farUpD, farUpV, fardownV, fardownD, parent, left, right, false);
}

int main() {
  ios_base::sync_with_stdio(0);
  int n;
  cin >> n;
  int left[n], right[n], parent[n], depth[n];

  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    left[i] = a-1;
    right[i] = b-1;
  }
  parent[0] = depth[0] = 0;
  calcDepth(0, -1, left, right, depth, parent);

  int h = floor(log2(n));
  int **links = new int*[h];
  for (int i = 0; i < h; i++)
    links[i] = new int[n];

  for (int i = 0; i < n; i++)
    links[0][i] = parent[i];

  for (int j = 1; j < h; j++) {
    for (int i = 0; i < n; i++) {
      if (links[j-1][i] != -1)
        links[j][i] = links[j-1][links[j-1][i]];
      else
        links[j][i] = -1;
    }
  }

  int fardownV[n], fardownD[n], farUpD[n], farUpV[n];
  bottomUp(0, fardownV, fardownD, left, right);
  upDown(0, farUpD, farUpV, fardownV, fardownD, parent, left, right, true);
  int farV[n], farD[n];

  for (int i = 0; i < n; i++) {
    if (fardownD[i] > farUpD[i]) {
      farD[i] = fardownD[i];
      farV[i] = fardownV[i];
    }
    else {
      farD[i] = farUpD[i];
      farV[i] = farUpV[i];
    }
  }

  int m;
  cin >> m;
  for (int i = 0; i < m; i++) {
    int a, d;
    cin >> a >> d;
    a--;
    int d_max = farD[a];
    if (d > d_max) {
      cout << "-1\n";
      continue;
    }
    int v = farV[a];
    int l = lca(a,v, depth, links, parent, n);
    int d1 = depth[a] - depth[l];
    int d2 = depth[v] - depth[l];
    if (d <= d1)
      cout << ancestor(a, d, n, links) + 1 << endl;
    else
      cout << ancestor(v, d_max - d, n, links) + 1 << endl;
  }
}
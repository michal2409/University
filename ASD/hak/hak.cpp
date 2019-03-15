#include <iostream>
#include <set>
#include <cmath>
#include <vector>

using namespace std;

int N = 131072;
int MAX = -1;
int MIN = 1000001;

typedef struct wezel wezel;
struct wezel {
  int maxPrawy = MAX, minPrawy = MIN, maxLewy = -1, minLewy = -1;
  set<pair<int, int>> odcinki;
};

wezel* drzewo;

void aktualizuj(int idx) {
  if (drzewo[idx].odcinki.empty()) {
    drzewo[idx].minPrawy = MIN;
    drzewo[idx].maxPrawy = MAX;
  }
  else {
    drzewo[idx].minPrawy = drzewo[idx].odcinki.begin()->first;
    drzewo[idx].maxPrawy = drzewo[idx].odcinki.rbegin()->first;
  }

  for (int i = idx; i > 1; i = i/2) {
    int idx2 = (i % 2 == 0) ? i + 1 : i - 1;
    drzewo[i/2].maxPrawy = max(drzewo[i].maxPrawy, drzewo[idx2].maxPrawy);
    drzewo[i/2].minPrawy = min(drzewo[i].minPrawy, drzewo[idx2].minPrawy);
    drzewo[i/2].minLewy = (drzewo[i].minPrawy < drzewo[idx2].minPrawy) ? drzewo[i].minLewy : drzewo[idx2].minLewy;
    drzewo[i/2].maxLewy = (drzewo[i].maxPrawy > drzewo[idx2].maxPrawy) ? drzewo[i].maxLewy : drzewo[idx2].maxLewy;
  }
}

int usunMinPrawy(int idx) {
  int wyn = drzewo[idx].odcinki.begin()->second;
  drzewo[idx].odcinki.erase(drzewo[idx].odcinki.begin());
  aktualizuj(idx);
  return wyn;
}

int usunMaxPrawy(int idx) {
  int wyn = (--drzewo[idx].odcinki.end())->second;
  drzewo[idx].odcinki.erase(--drzewo[idx].odcinki.end());
  aktualizuj(idx);
  return wyn;
}

int in(int a, int b) {
   int va = N + a - 1, vb = N + b - 1;
   if (drzewo[va].minPrawy <= b) return usunMinPrawy(drzewo[va].minLewy);
   if (va != vb && drzewo[vb].minPrawy <= b) return usunMinPrawy(drzewo[vb].minLewy);

   while (va / 2 != vb / 2) {
     if (va % 2 == 0 && drzewo[va+1].minPrawy <= b) return usunMinPrawy(drzewo[va+1].minLewy);
     if (vb % 2 == 1 && drzewo[vb-1].minPrawy <= b) return usunMinPrawy(drzewo[vb-1].minLewy);
     va /= 2; vb /= 2;
   }

   return -1;
}

int over(int a, int b) {
   int va = N, vb = N + a - 1;
   if (drzewo[va].maxPrawy >= b) return usunMaxPrawy(drzewo[va].maxLewy);
   if (va != vb && drzewo[vb].maxPrawy >= b) return usunMaxPrawy(drzewo[vb].maxLewy);

   while (va / 2 != vb / 2) {
     if (va % 2 == 0 && drzewo[va+1].maxPrawy >= b) return usunMaxPrawy(drzewo[va+1].maxLewy);
     if (vb % 2 == 1 && drzewo[vb-1].maxPrawy >= b) return usunMaxPrawy(drzewo[vb-1].maxLewy);
     va /= 2; vb /= 2;
   }
   return -1;
}

int some(int a, int b) {
   int va = N, vb = N + b - 1;
   if (drzewo[va].maxPrawy >= a) return usunMaxPrawy(drzewo[va].maxLewy);
   if (va != vb && drzewo[vb].maxPrawy >= a) return usunMaxPrawy(drzewo[vb].maxLewy);

   while (va / 2 != vb / 2) {
     if (va % 2 == 0 && drzewo[va+1].maxPrawy >= a) return usunMaxPrawy(drzewo[va+1].maxLewy);
     if (vb % 2 == 1 && drzewo[vb-1].maxPrawy >= a) return usunMaxPrawy(drzewo[vb-1].maxLewy);
     va /= 2; vb /= 2;
   }
   return -1;
}

int none1(int a) {
   int va = N, vb = N + a - 1;
   if (drzewo[va].minPrawy < a) return usunMinPrawy(drzewo[va].minLewy);
   if (va != vb && drzewo[vb].minPrawy < a) return usunMinPrawy(drzewo[vb].minLewy);

   while (va / 2 != vb / 2) {
     if (va % 2 == 0 && drzewo[va+1].minPrawy < a) return usunMinPrawy(drzewo[va+1].minLewy);
     if (vb % 2 == 1 && drzewo[vb-1].minPrawy < a) return usunMinPrawy(drzewo[vb-1].minLewy);
     va /= 2; vb /= 2;
   }
   return -1;
}

int none2(int b) {
   int va = N + b, vb = 2*N - 1;
   if (drzewo[va].maxPrawy > b) return usunMaxPrawy(drzewo[va].maxLewy);
   if (va != vb && drzewo[vb].maxPrawy > b) return usunMaxPrawy(drzewo[vb].maxLewy);

   while (va / 2 != vb / 2) {
     if (va % 2 == 0 && drzewo[va+1].maxPrawy > b) return usunMaxPrawy(drzewo[va+1].maxLewy);
     if (vb % 2 == 1 && drzewo[vb-1].maxPrawy > b) return usunMaxPrawy(drzewo[vb-1].maxLewy);
     va /= 2; vb /= 2;
   }
   return -1;
}

int none(int a, int b) {
  int x = none1(a);
  if (x != -1)
    return x;
  return none2(b);
}

int main() {
  ios_base::sync_with_stdio(0);
  drzewo = new wezel[2*N];

  int n;
  cin >> n;
  for (int i = 1; i <= n; ++i) {
    int a, b;
    cin >> a >> b;
    drzewo[N + a - 1].odcinki.emplace(make_pair(b, i));
  }

  for (int i = N; i < 2*N; i++) {
    if (drzewo[i].odcinki.empty())
      continue;
    drzewo[i].minPrawy = drzewo[i].odcinki.begin()->first;
    drzewo[i].maxPrawy = drzewo[i].odcinki.rbegin()->first;
    drzewo[i].minLewy = drzewo[i].maxLewy = i;
  }

  for (int i = N-1; i > 0; --i) {
    drzewo[i].maxPrawy = max(drzewo[2*i].maxPrawy, drzewo[2*i + 1].maxPrawy);
    drzewo[i].minPrawy = min(drzewo[2*i].minPrawy, drzewo[2*i + 1].minPrawy);
    drzewo[i].minLewy = (drzewo[2*i].minPrawy < drzewo[2*i + 1].minPrawy) ? drzewo[2*i].minLewy : drzewo[2*i + 1].minLewy;
    drzewo[i].maxLewy = (drzewo[2*i].maxPrawy > drzewo[2*i + 1].maxPrawy) ? drzewo[2*i].maxLewy : drzewo[2*i + 1].maxLewy;
  }

  int q;
  cin >> q;
  for (int i = 0; i < q; ++i) {
    string s;
    int a, b;
    cin >> s >> a >> b;
    if (s.compare("in") == 0)
      cout << in(a, b) << endl;
    else if (s.compare("over") == 0)
      cout << over(a, b) << endl;
    else if (s.compare("none") == 0)
      cout << none(a, b) << endl;
    else
      cout << some(a, b) << endl;
  }
}
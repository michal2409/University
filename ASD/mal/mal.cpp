#include <iostream>
#include <cmath>

using namespace std;

void updateNode(int node, int tree[], int lazy[], int start, int end, int val) {
	tree[node] = val * (end - start + 1);
    if(start != end) {
        lazy[node*2] = val;
        lazy[node*2 + 1] = val;
    }
}

void updateRange(int node, int start, int end, int l, int r, int val, int tree[], int lazy[]) {
	if(lazy[node] != -1) {
    	updateNode(node, tree, lazy, start, end, lazy[node]);
	    lazy[node] = -1;
    }

    if(start > end || start > r || end < l)
        return;

    if(start >= l && end <= r) {
    	updateNode(node, tree, lazy, start, end, val);
        return;
    }
    int mid = start + (end - start) / 2;
    updateRange(node*2, start, mid, l, r, val, tree, lazy);
    updateRange(node*2 + 1, mid + 1, end, l, r, val, tree, lazy);
	tree[node] = tree[node*2 + 1] + tree[node*2];
}

int main() {
	int n, m;
	cin >> n >> m;
	int dl = pow(2, sizeof(int) * 8 - __builtin_clz(n) + 1) ;
	int tree[dl], lazy[dl];
	for (int i = 0; i < dl; i++) {
		lazy[i] = -1;
		tree[i] = 0;
	}

	int l, p; char d;
	for (int i = 0; i < m; i++) {
		cin >> l >> p >> d;
		int x = (d == 'C') ? 0 : 1;
		updateRange(1, 1, n, l, p, x, tree, lazy);
		cout << tree[1] << endl;
	}
}
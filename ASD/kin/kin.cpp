#include <iostream>

int n; 

void build(int tree[]) {
	for (int i = 0; i < 2*n; i++)
		tree[i] = 0;
}

void updateTreeNode(int p, int value, int tree[]) {
	tree[p + n] = value;

	for (int i = p + n; i > 1; i = i/2)
		tree[i/2] = (tree[i] + tree[(i % 2 == 0) ? i + 1 : i - 1]) % 1000000000;
}

int query(int l, int r, int tree[]) {
	long long sum = 0;
	for (l += n, r += n; l < r; l = l/2, r = r/2) {
		if (l % 2 == 1)
			sum = (sum + tree[l++]) % 1000000000;
		if (r % 2 == 1)
			sum = (sum + tree[--r]) % 1000000000;
	}
	return sum;
}

int main() {
	int k;
	std::cin >> n >> k;

	int a[n], inv[n];
	for (int i = 0; i < n; i++) {
		std::cin >> a[i];
		a[i]--;
		inv[i] = 1;
	}

	for (int j = k-1; j > 0; j--) {
		int tree[2*n];
		build(tree);

		for (int i = 0; i < n; i++) {
			int pom = inv[i];
			inv[i] = query(a[i]+1, n, tree);
			updateTreeNode(a[i], pom, tree);
		}
	}

	long long sum = 0;
	for (int i = 0; i < n; i++)
		sum = (sum + inv[i]) % 1000000000;

	std::cout << sum << std::endl;

	return 0;
}
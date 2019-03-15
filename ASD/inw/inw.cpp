#include <iostream>
#include <vector>

using namespace std;

int main() {
	ios_base::sync_with_stdio(0);
	int n;
	cin >> n;

	int arr[n];
	for (int i = 0; i < n; ++i) {
		cin >> arr[i];
	}

	vector<int> v;
	for (int i = 0; i < n;) {
		int max = arr[i], count = 0, curr = i;
		while (i < n && count < max - curr) {
			if (arr[i] > max)
				max = arr[i];
			i++;
			count++;
		}
		v.push_back(max);
	}

	cout << v.size() << endl;
	int j = 1;
	for (size_t i = 0; i < v.size(); ++i) {
		cout << v[i] - j + 1;
		while (j <= v[i]) {
			cout << " " << j;
			j++;
		}
		cout << endl;
	}

}
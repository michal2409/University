#include <iostream>
#include <climits>
using namespace std;

int main() {
  ios_base::sync_with_stdio(0);
  string line;
  getline(cin, line);
  long long minDiff = LLONG_MAX, count = 0, i = 0;

  while (line[i] == '*' && i < line.length())
  	i++;

  if (i == line.length()) {
  	cout << 1 << endl;
  	return 0;
  }

  char lastChar = line[i];

  for (int j = i+1; j < line.length(); j++) {
  	if (line[j] == '*') 
  		count++;
  	else if (line[j] != lastChar) {
  		if (count == 0) {
  			cout << line.length() << endl;
  			return 0;
  		}
		lastChar = line[j];
		if (count < minDiff)
			minDiff = count;
		count = 0;
  	} 
  	else 
  		count = 0;
  }

  if (minDiff == LLONG_MAX) {
  	cout << 1 << endl;
  	return 0;
  }

  cout << line.length() - minDiff << endl; 
}
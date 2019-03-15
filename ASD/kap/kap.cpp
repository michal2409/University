#include <queue>
#include <vector>
#include <iostream>
#include<utility>
#include<list>
#include<climits>
#include <algorithm>

using namespace std;

typedef pair<int, int> iPair;
class Graph
{
    int V;
    list< pair<int, int> > *adj;
public:
    Graph(int V);
    void addEdge(int u, int v, int w);
    void shortestPath(int s);
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new list<iPair> [V];
}

void Graph::addEdge(int u, int v, int w)
{
    adj[u].push_back(make_pair(v, w));
    adj[v].push_back(make_pair(u, w));
}

void Graph::shortestPath(int src)
{
    priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
    vector<long long> dist(V, LLONG_MAX);
    pq.push(make_pair(0, src));
    dist[src] = 0;
    while (!pq.empty())
    {
        int u = pq.top().second;
        pq.pop();
        list< pair<int, int> >::iterator i;
        for (i = adj[u].begin(); i != adj[u].end(); ++i)
        {
            int v = (*i).first;
            int weight = (*i).second;
            if (dist[v] > dist[u] + weight)
            {
                dist[v] = dist[u] + weight;
                pq.push(make_pair(dist[v], v));
            }
        }
    }
    cout << dist[V-1];
}

typedef struct punkt punkt;
struct punkt {
  int x, y, idx;
};

bool comp_x (const punkt& w1, const punkt& w2) {
  if(w1.x < w2.x)
    return true;
  return (w1.x == w2.x && w1.y < w2.y);
}

bool comp_y (const punkt& w1, const punkt& w2) {
  if(w1.y < w2.y)
    return true;
  return (w1.y == w2.y && w1.x < w2.x);
}

int main() {
  ios_base::sync_with_stdio(0);
  int n;
  cin >> n;
  Graph g(n);

  punkt w[n];
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    punkt node; node.x=a; node.y=b, node.idx = i;
    w[i] = node;
  }

  sort(w, w+n, &comp_x);
  for (int i = 0; i < n; i++) {
    if (i > 0)
      g.addEdge(w[i].idx, w[i-1].idx, abs(w[i].x - w[i-1].x));
    if (i < n-1)
      g.addEdge(w[i].idx, w[i+1].idx, abs(w[i].x - w[i+1].x));
  }

  sort(w, w+n, &comp_y);
  for (int i = 0; i < n; i++) {
    if (i > 0)
       g.addEdge(w[i].idx, w[i-1].idx, abs(w[i].y - w[i-1].y));
    if (i < n-1)
       g.addEdge(w[i].idx, w[i+1].idx, abs(w[i].y - w[i+1].y));
  }

  g.shortestPath(0); 
}
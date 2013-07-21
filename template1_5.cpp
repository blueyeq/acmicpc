                                        ACM/ICPC  CODE	 LIBRARY  FOR  HNU0314

最大流Dinic:
bool bfs() {
    queue<int> q;
    memset(dis, -1, sizeof(dis));
    dis[src] = 0;      // dis[src] = 0
    q.push(src);
    while(!q.empty()) {
        int u = q.front();
        q.pop();  //  bug q.pop()
        for(int i = head[u]; i != -1; i = edg[i].next) {
            int v = edg[i].v;
            if(dis[v] == -1 && edg[i].w > 0) {
                dis[v] = dis[u] + 1;
                q.push(v);
                if(v == sink)  return 1;
            }
        }
    }
    return 0;
}
int dinic(int x, int flow) {
    if(x == sink)  return flow;
    for(int &i = work[x]; i != -1; i = edg[i].next) {
        int v = edg[i].v;
        if(dis[v] == dis[x] + 1 && edg[i].w > 0 && flow > 0) {
            int tmp = dinic(v, min(flow, edg[i].w));
            edg[i].w -= tmp;
            edg[i ^ 1].w += tmp;
            flow -= tmp;
            if(tmp > 0) return tmp;
        }
    }
    return 0;
}
int maxflow() { // LL
    int res(0);
    while(bfs()) {
        for(int i = 0; i <= N; ++i)  work[i] = head[i];   // N not n!!!
        work[src] = head[src];
        work[sink] = head[sink];   // src, sink
        while(int tmp = dinic(src, INF))  res += tmp;   // LL
    }
    return res;
}
最小费用最大流:
spfa找一条最小费用的路径, pre记录节点前驱"边" 的编号, end 更新边的容量,
bool in[MAXN];
int pre[MAXN], low[MAXN], dis[MAXN];

// search for the min cost path
bool spfa() {
    stack<int> st;
    for(int i = 0; i <= n; ++i)  dis[i] = INF;
    st.push(0);   // stack maybe faster than queu
    dis[0] = 0;
    low[0] = INF;
    while(!st.empty()) {
        int u = st.top();
        st.pop();
        in[u] = 0;
        for(int i = head[u]; i != -1; i = edg[i].next) {
            int v = edg[i].v;
            if(edg[i].cap > 0
                    && dis[v] > dis[u] + edg[i].w) {  //  edg[i].cap > 0
                dis[v] = dis[u] + edg[i].w;
                low[v] = min(low[u], edg[i].cap);
                pre[v] = i;    //  pre is edg'id not node'id
                if(!in[v])  st.push(v);
                in[v] = 1;
            }
        }
    }
    return dis[n] != INF;
}
int end() {
    int u = n;
    while(u != 0) {
        edg[pre[u]].cap -= low[n];  // update the cap about the edg
        edg[pre[u] ^ 1].cap += low[n];
        u = edg[pre[u]].from;
    }
    return dis[n];
}



上下界网络流：
无源无汇:
添加源(s)和汇(t)  M[i] = sigma(in[i] - out[i]), 若M[i] > 0 则添加s->i 流量M[i]
 M[i] < 0 添加 i->t流量-M[i], 原边的流量edg[i].w -= low[i]. 跑一遍s到t的最大流，
 判断s和t的所有边满流则存在可行流, 最后每条边的实际流量为边的下界加上最大流时跑的流量之和,
 若是有源有汇的则照样按照无源无汇的方法添加新源和新汇建图， 最后若是求最小流则添加一条原来的汇到原来的源的边
ss->tt 容量下界：0 上界为题目所有边的流量之和: sumFlow二分答案，  若是求最大流ss->tt的上界为+oo

inline void  add(int a, int b, int c) {
    orgEdg[e_cnt].v = b;
    orgEdg[e_cnt].next = orgHead[a];
    orgEdg[e_cnt].w = c;
    orgHead[a] = e_cnt++;
}
void cop() { // 求最小流时二分多次跑最大流每次复制一遍原图
    for(int i = 0; i < MAXN; ++i)
        head[i] = orgHead[i];
    for(int i = 0; i < e_cnt; ++i)
        edg[i] = orgEdg[i];
}
int maxflow(int h) {
    cop();
    low[e_cnt / 2] = 0;
    添加原来的汇到原来的源容量为此时的二分值h的上界 edg[e_cnt].v = 1;
    edg[e_cnt].w = h;
    edg[e_cnt].next = head[n];
    head[n] = e_cnt++;
    edg[e_cnt].v = n;
    edg[e_cnt].w = 0;
    edg[e_cnt].next = head[1];
    head[1] = e_cnt--;
    int res(0);
    while(bfs()) {
        for(int i = 1; i <= n; ++i)  work[i] = head[i];
        work[src] = head[src];
        work[sink] = head[sink];   // src, sink
        while(int tmp = dinic(src, INF) > 0)  res += tmp;
    }
    return res;
}
void init() {
    memset(orgHead, -1, sizeof(orgHead));    // orgHead not head
    memset(W, 0, sizeof(W));
    e_cnt = 0;
    for(int i = 0; i < m; ++i) {
        int ta, tb;
        scanf("%d%d%d%d", &ta, &tb, &hig[i], &low[i]);
        if(low[i] == 1)  low[i] = hig[i];
        W[ta] -= low[i];   //  -out,  +in
        W[tb] += low[i];
        add(ta, tb, hig[i] - low[i]);
        add(tb, ta, 0);
    }
    for(int i = 1; i <= n; ++i) {
        if(W[i] > 0) {
            add(src, i, W[i]);
            add(i, src, 0);
        } else if(W[i] < 0) {
            add(i, sink, -W[i]);
            add(sink, i, 0);
        }
    }
}
bool ok(int mid) {  //判断源和汇是否满流
    int flow = maxflow(mid);
    for(int i = head[src]; i != -1; i = edg[i].next)
        if(edg[i].w)  return 0;
    for(int i = head[sink]; i != -1; i = edg[i].next)
        if(edg[i ^ 1].w) return 0;
    return 1;
}
//  矩形方格上下界网络流：
#include <stdio.h>
#include <cstring>
#include <string>
#include <queue>

using namespace std;

const int MAXN = 310;
const int INF = 100000000;

int src, sink, n, m;
int low[MAXN][MAXN], up[MAXN][MAXN], flow[MAXN][MAXN];

inline void normal(int x, int h, int &a, int &b){
	if(x) a = b = x;
	else a = 1, b = h;
}
void init()
{
    scanf("%d%d", &n, &m);
    memset(low, 0, sizeof(low));
    memset(up, 0, sizeof(up));
    src = n + m + 1; sink = src + 1;

    for(int i = 1; i <= n; ++i)
    for(int j = n + 1; j <= n + m; ++j)
        up[i][j] = INF;
    for(int i = 1; i <= n; ++i) {
        scanf("%d", &low[src][i]);
        up[src][i] = low[src][i];
    }
    for(int i = 1; i <= m; ++i) {
        scanf("%d", &low[i + n][sink]);
        up[i + n][sink] = low[i + n][sink];
    }
    int acc, i1, j1, i2, j2, val;   scanf("%d", &acc);
    char op[10];
    for(int k = 1; k <= acc; ++k)
    {
        scanf("%d %d %s %d", &i2, &j2, op, &val);
        normal(i2, n, i1, i2);
		normal(j2, m, j1, j2);
		for(int i = i1; i <= i2; ++i)
        for(int j = j1; j <= j2; ++j)
        {
            if(op[0] == '=') {
                up[i][j + n] = min(up[i][j + n], val);
                low[i][j + n] = max(low[i][j + n], val);
            }
            if(op[0] == '<') {
                up[i][j + n] = min(up[i][j + n], val - 1);
            }
            if(op[0] == '>') {
                low[i][j + n] = max(low[i][j + n], val + 1);
            }
        }
    }
}
int dis[MAXN];
bool bfs()
{
    queue<int> q;
    memset(dis, -1, sizeof(dis));
    q.push(src);  dis[src] = 0;
    while(!q.empty())
    {
        int u = q.front();  q.pop();
        for(int i = 1; i <= sink; ++i)
        {
//            if(up[u][i])  printf("%d -> %d  %d  %d\n", u, i, up[u][i], flow[u][i]);
            if(up[u][i] > 0 && dis[i] == -1)
            {
                q.push(i);
                dis[i] = dis[u] + 1;
                if(i == sink)  return 1;
            }
        }
    }
    return 0;
}
int dinic(int x, int ff)
{
    if(x == sink)  return ff;
    for(int i = 1; i <= sink; ++i)
    {
        if(0 < up[x][i] && dis[i] == dis[x] + 1)
        {
            int tmp = dinic(i, min(up[x][i]  , ff));  // 邻接矩阵各种蛋疼
            flow[x][i] += tmp;
            up[x][i] -= tmp;
            up[i][x] += tmp;
            flow[i][x] -= tmp;
            if(tmp > 0)  return tmp;
        }
    }
    return 0;
}
int maxflow(){
    memset(flow, 0, sizeof(flow));
    int res(0);
    while(bfs()){
        while(int tmp = dinic(src, INF))  res += tmp;
    }
    return res;
}


int xj[MAXN];
bool slove()
{
    int sumRow(0), sumCol(0);
    for(int i = 1; i <= n; ++i)
        sumRow += low[src][i];
    for(int i = n + 1; i <= n + m; ++i)
        sumCol += low[i][sink];

    if(sumRow != sumCol) return 0;
    memset(xj, 0, sizeof(xj));
    for(int i = 1; i <= sink; ++i){
        for(int j = 1; j <= sink; ++j){
            if(low[i][j] > up[i][j])  return 0;
            up[i][j] -= low[i][j];
            xj[i] -= low[i][j];
            xj[j] += low[i][j];
        }
    }
    up[sink][src] = INF;
    int nsrc = sink + 1,  nsink = sink + 2;
    int s1(0), s2(0);
    for(int i = 1; i <= sink; ++i)
    {
        if(xj[i] < 0)  up[i][nsink] = -xj[i];
        else  up[nsrc][i] = xj[i], s1 += xj[i];
    }
    src = nsrc; sink = nsink;
    int ff = maxflow();
    return ff == s1;
}

int main()
{
//	freopen("std.in", "r", stdin);
	int test, cas(1);
    scanf("%d", &test);
    while(test--)
    {
        init();
        if(!slove())  puts("IMPOSSIBLE");
        else
        {
            for(int i = 1; i <= n; ++i)
            for(int j = n + 1; j <= n + m; ++j)
                printf("%d%c", flow[i][j] + low[i][j],  j == n + m ? '\n' : ' ');
        }
        puts("");
    }
    return 0;
}


最小树形图:
#define TYP int
const int MAXN = 1010;
const int MAXM = 20000;
const TYP INF = 1000000000;
int cnt;
struct Edg {
    int f,  t;
    TYP w;
} edg[MAXM];
inline void add(int a, int b, TYP c) {
    edg[cnt].f = a;
    edg[cnt].t = b;
    edg[cnt++].w = c;
}
int id[MAXN], vis[MAXN], pre[MAXN], eid;
TYP in[MAXN];
// TYP  int LL  pre,  inistall id[], vis[],  in[root] = 0!!!!
TYP dirmst(int root, int nv, int ne) {
    TYP ret(0);
    while(1) {
        for(int i = 0; i < nv; ++i)   in[i] = INF;
        for(int i = 0; i < ne; ++i) {
            int f = edg[i].f;
            int t = edg[i].t;
            if(edg[i].w < in[t] && f != t ) {
                in[t] = edg[i].w;
                pre[t] = f;
                if(pre[t] == root) {
                    eid = i;
                }
            }
        }
        for(int i = 0; i < nv; ++i)
            if(in[i] == INF && i != root)  return -1;
        in[root] = 0;
        int cntnode(0);
        for(int i = 0; i < nv; ++i)
            vis[i] = id[i] = -1;
        for(int i = 0; i < nv; ++i) {
            ret += in[i];
            int v = i;
            while(id[v] == -1 && vis[v] != i && v != root) {
                vis[v] = i;
                v = pre[v];
            }
            if(v != root && id[v] == -1) {
                for(int u = pre[v]; u != v; u = pre[u])
                    id[u] = cntnode;
                id[v]=cntnode++;
            }
        }
        if(cntnode == 0)  break;
        for(int i = 0; i < nv; ++i)
            if(id[i] == -1)  id[i] = cntnode++;
        for(int i = 0; i < ne; ++i) {
            int u = edg[i].t;
            edg[i].f = id[edg[i].f];
            edg[i].t = id[edg[i].t];
            if(edg[i].f != edg[i].t)
                edg[i].w -= in[u];
        }
        nv = cntnode;
        root = id[root];
    }
    return ret;
}
无向图最小边割集：
int mat[600][600];
int res;//Stoer-Wagner算法，加了自己看得懂的备注//无向图全局最小割，用求prim类似方法o（n^3)，学习了一个下午……//一开始用枚举源点汇点的最大流求解，复杂度o(n^5) 超时void Mincut(int n) {    int node[600], dist[600];    bool visit[600];    int i, prev, maxj, j, k;    for (i = 0; i < n; i++)        node[i] = i;    while (n > 1) {        int maxj = 1;        for (i = 1; i < n; i++) { //初始化到已圈集合的割大小            dist[node[i]] = mat[node[0]][node[i]];            if (dist[node[i]] > dist[node[maxj]])                maxj = i;        }        prev = 0;        memset(visit, false, sizeof (visit));        visit[node[0]] = true;        for (i = 1; i < n; i++) {            if (i == n - 1) { //只剩最后一个没加入集合的点，更新最小割                res = min(res, dist[node[maxj]]);                for (k = 0; k < n; k++) //合并最后一个点以及推出它的集合中的点                    mat[node[k]][node[prev]] = (mat[node[prev]][node[k]] += mat[node[k]][node[maxj]]);                node[maxj] = node[--n]; //缩点后的图            }            visit[node[maxj]] = true;            prev = maxj;            maxj = -1;            for (j = 1; j < n; j++)                if (!visit[node[j]]) { //将上次求的maxj加入集合，合并与它相邻的边到割集                    dist[node[j]] += mat[node[prev]][node[j]];                    if (maxj == -1 || dist[node[maxj]] < dist[node[j]])                        maxj = j;                }        }    }    return;}
二分图KM匹配:
bool find(int x) {
    visx[x] = 1;
    for(int i = 0; i < boy + girl; ++i) {
        if(visy[i])  continue;
        int t = lx[x] + ly[i] - g[x][i];
        if(t == 0) {
            visy[i] = 1;
            if(matchy[i] == -1 || find(matchy[i])) {
                matchy[i] = x;
                return 1;
            }
        } else if(lack > t)
            lack = t;
    }
    return 0;
}
int KM() {
    memset(matchy, -1, sizeof(matchy));
    memset(lx, 0, sizeof(lx));
    memset(ly, 0, sizeof(ly));
    for(int i = 0; i < boy + girl; ++i)
        for(int j = 0; j < boy + girl; ++j)
            if(g[i][j] > lx[i])  lx[i] = g[i][j];

    for(int x = 0; x < boy; ++x) {
        while(1) {
            memset(visx, 0, sizeof(visx));
            memset(visy, 0, sizeof(visy));
            lack = INF;
            if(find(x))  break;
            for(int i = 0; i < boy + girl; ++i) {
                if(visx[i])  lx[i] -= lack;
                if(visy[i])  ly[i] += lack;
            }
        }
    }
    int res(0);
    for(int i = boy; i < boy + girl; ++i)
        res += g[matchy[i]][i];
    return res;
}
一般图最大匹配：
struct Graph {
    int n, match[maxn];
    bool adj[maxn][maxn];
    void clear() {
        memset(adj, 0, sizeof(adj));
        n = 0;
    }
    void insert(const int &u, const int &v) {
        if(max(u, v) + 1 > n) n = max(u, v) + 1;
        adj[u][v] = adj[v][u] = 1;
    }
    int max_match() {
        memset(match, -1, sizeof(match));
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (match[i] == -1) {
                ans += bfs(i);
            }
        }
        return ans;
    }
    int Q[maxn], pre[maxn], base[maxn];
    bool hash[maxn];
    bool in_blossom[maxn];
    int bfs(int p) {
        memset(pre, -1, sizeof(pre));
        memset(hash, 0, sizeof(hash));
        for (int i = 0; i < n; ++i) {
            base[i] = i;
        }
        Q[0] = p;
        hash[p] = 1;
        for (int s = 0, t = 1; s < t; ++s) {
            int u = Q[s];
            for (int v = 0; v < n; ++v) {
                if (adj[u][v] && base[u] != base[v] && v != match[u]) {
                    if (v == p || (match[v] != -1 && pre[match[v]] != -1)) {
                        int b = contract(u, v);
                        for (int i = 0; i < n; ++i) {
                            if (in_blossom[base[i]]) {
                                base[i] = b;
                                if (hash[i] == 0) {
                                    hash[i] = 1;
                                    Q[t++] = i;
                                }
                            }
                        }
                    } else if (pre[v] == -1) {
                        pre[v] = u;
                        if (match[v] == -1) {
                            argument(v);
                            return 1;
                        } else {
                            Q[t++] = match[v];
                            hash[match[v]] = 1;
                        }
                    }
                }
            }
        }
        return 0;
    }
    void argument(int u) {
        while (u != -1) {
            int v = pre[u];
            int k = match[v];
            match[u] = v;
            match[v] = u;
            u = k;
        }
    }
    void change_blossom(int b, int u) {
        while (base[u] != b) {
            int v = match[u];
            in_blossom[base[v]] = in_blossom[base[u]] = true;
            u = pre[v];
            if (base[u] != b) {
                pre[u] = v;
            }
        }
    }
    int contract(int u, int v) {
        memset(in_blossom, 0, sizeof(in_blossom));
        int b = find_base(base[u], base[v]);
        change_blossom(b, u);
        change_blossom(b, v);
        if (base[u] != b) {
            pre[u] = v;
        }
        if (base[v] != b) {
            pre[v] = u;
        }
        return b;
    }
    int find_base(int u, int v) {
        bool in_path[maxn] = {};
        while (true) {
            in_path[u] = true;
            if (match[u] == -1) {
                break;
            }
            u = base[pre[match[u]]];
        }
        while (!in_path[v]) {
            v = base[pre[match[v]]];
        }
        return v;
    }
};
2Sat构造解：
void topsort() {

    queue<int> q;
    for(int i = 0; i < ID; ++i) {
        if(in[i] == 0)  {
            q.push(i), color[i] = 0;
            color[re[i]] = 1;
        }
    }
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        if(color[u] == -1)  color[u] = 0, color[re[u]] = 1;
        for(int i = 0; i < g[u].size(); ++i) {
            int v = g[u][i];
            --in[v];
            if(!in[v])  q.push(v);
        }
    }
}
void slove() {

    for(int i = 0; i < ID; ++i) {
        g[i].clear();
        color[i] = -1;
        in[i] = 0;
    }
    for(int i = 0; i < n + n; ++i) {
        for(int j = head[i]; j != -1; j = edg[j].next) {
            int v = edg[j].v;
            if(belong[i] != belong[v]) {
                g[belong[v]].push_back(belong[i]);
                ++in[belong[i]];
            }
        }
    }
    topsort();
    for(int i = 1; i < n; ++i) {

        if(i != 1)  putchar(' ');
        if(color[belong[i]] == color[belong[n]]) {
            printf("%dh", i);
        } else if(color[belong[i + n]] == color[belong[n]]) {
            printf("%dw", i);
        } else while(1);
    }
    puts("");
}

AC自动机：  不要忘记调用bfs()
#define clr(a) memset(a,0,sizeof(a))
#define N 100
#define D 6
#define M 24
int n, m, ID;
const int D = 2;
int next[MAXN][D], fail[MAXN], mp[300], mask[MAXN];
bool typ[MAXN];

int newnode() {
    memset(next[ID], 0, sizeof(next[ID]));
    typ[ID] = fail[ID] = mask[ID] = 0;
    return ID++;
}

int ins(char *s, int id) {
    int p(0);
    for(; *s; ++s) {
        if(!next[p][mp[*s]])  next[p][mp[*s]] = newnode();
        p = next[p][mp[*s]];
    }
    return p;
}
void bfs() {

    int p(0);
    queue<int> q;
    q.push(0);
    while(!q.empty()) {
        int p = q.front();
        q.pop();
        typ[p] |= typ[fail[p]];
        mask[p] |= mask[fail[p]];

        for(int i = 0; i < D; ++i) {
            if(next[p][i]) {
                int t = next[p][i];
                q.push(t);
                if(p)  fail[t] = next[fail[p]][i];
                else  fail[t] = 0;
            } else  next[p][i] = next[fail[p]][i];
        }
    }
}
void init() {
    ID = 0;
    newnode();
    mp['0'] = 0;
    mp['1'] = 1;
}
构造相应矩阵
void make() {
    for(int i=0; i<ID; i++) {
        if(type[i]==0)
            for(int j=0; j<D; j++) {
                if(type[next[i][j]]==0)
                    mu[i][next[i][j]]++;
            }
    }
}
后缀自动机：  while(pre[last]) 初始化
    int n, ID, last;
int next[MAXN][D], pre[MAXN],  step[MAXN], mp[300];

int newnode() {
    memset(next[ID], 0, sizeof(next[ID]));
    return ID++;
}
void init() {

    for(int i = 'a'; i <= 'z'; ++i)  mp[i] = i - 'a';

    ID = last = 0;
    newnode();
    step[0] = 0;
    pre[0] = -1;
}
void ins(char ch) {

    int np = newnode(), p(last);
    last = np;
    step[np] = step[p] + 1;
    for(; ~p && !next[p][ch]; p = pre[p])  next[p][ch] = np;
    if(p == -1)  pre[np] = 0;
    else {
        int q = next[p][ch];
        if(step[q] == step[p] + 1) {
            pre[np] = q;   // np
        } else {
            int nq = newnode();
            memcpy(next[nq], next[q], sizeof(next[nq]));
            pre[nq] = pre[q];
            step[nq] = step[p] + 1;   // nq  step
            pre[np] = pre[q] = nq;
            for(; ~p && next[p][ch] == q; p = pre[p])  next[p][ch] = nq;  // nq
        }
    }
}
杭州网络赛G题：
#include <stdio.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <queue>

using namespace std;

typedef long long LL;

const int MAXN = 300000;
const int INF = 0;
const int D = 26;
LL num[MAXN], total;
int Max[MAXN];

int ID, last;
int next[MAXN][D], pre[MAXN], step[MAXN], mp[300];
bool vis[MAXN];

inline int newnode() {
    memset(next[ID], 0, sizeof(next[ID]));
    pre[ID] = num[ID] = Max[ID] = 0;
    return ID++;
}

void init() {
    for(int i = 'a'; i <= 'z'; ++i)  mp[i] = i - 'a';
    ID = 0;
    newnode();
    step[0] = last = 0;
    pre[0] = -1;
}

void ins(char ch) {

    int np = newnode(), p(last);
    last = np;
    step[np] = step[p] + 1;
    for(; ~p && !next[p][ch]; p = pre[p])  next[p][ch] = np;
    if(p == -1)  pre[np] = 0;
    else {
        int q = next[p][ch];
        if(step[q] == step[p] + 1) {
            pre[np] = q;   // np
        } else {
            int nq = newnode();
            memcpy(next[nq], next[q], sizeof(next[nq]));
            pre[nq] = pre[q];
            step[nq] = step[p] + 1;   // nq  step
            pre[np] = pre[q] = nq;
            for(; ~p && next[p][ch] == q; p = pre[p])  next[p][ch] = nq;  // nq
        }
    }
}
int st[MAXN], sz;

void getOrder() {
    int l(0), r(0);
    st[0] = 0;
    while(l <= r) {
        int u = st[l++];
        for(int i = 0; i < D; ++i) {
            int v = next[u][i];
            if(v && step[v] == step[u] + 1) {
                st[++r] = v;
            }
        }
    }
    sz = r;
}

LL getTotal() {

    LL res(0);
    num[0] = 1;
    for(int i = 0; i <= sz; ++i) {
        int u = st[i];     // u  not i
        for(int j = 0; j < D; ++j) {
            if(next[u][j]) {
                num[next[u][j]] += num[u];
            }
        }
        if(u) res += num[u];
        Max[u] = 0;
    }
    return res;
}

void process(char *s, int len) {

    int p(0), now(0);
    for(int i = 0; i < len; ++i, ++s) {
        while(p && !next[p][ mp[*s]])  {   // p  not !p
            now = step[p = pre[p]];
        }
        p = next[p][mp[*s]];
        if(p) ++now;
        Max[p] = max(Max[p], now);
    }
}

char s1[MAXN], s2[MAXN], s[MAXN], len[MAXN];

void readin() {

    init();
    int n;
    scanf("%d", &n);
    scanf("%s", s1);
    for(int i = 0; s1[i]; ++i) {
        ins(mp[ s1[i] ]);
//        printf("%c %d\n", s1[i], ID);
    }
    getOrder();
    total = getTotal();
//    printf("ID=%d\n", ID);
    for(int i = 0; i < n; ++i) {
        scanf("%s", s2);
        process(s2, strlen(s2));
    }
}

LL solve() {
//    printf("total=%I64d\n", total);
    LL res(0);
    for(int i = sz; i > 0; --i) {
        int u = st[i];
        int v = pre[u];
        Max[v] = max(Max[v], Max[u]);
        if(Max[u]) {
            if(step[ u ] > Max[u])  res += (Max[u] - step[v]);
            else  res += num[u];
        }
    }
    return total - res;
}
int main() {

//    freopen("std.in", "r", stdin);
    int test, cas(1);
    scanf("%d", &test);
    while(test--) {
        readin();
        printf("Case %d: %I64d\n", cas++, solve());
    }
    return 0;
}

每个位置的最长回文串:
void turn(int &len) {
    len = len * 2 + 1;
    for(int i = len; i >= 0; --i) {
        if(i & 1)  s[i] = '#';
        else  s[i] = s[i / 2];
    }
    s[0] = '$';
    s[len + 1] = 0;
}
void make(int len) {
    int cnt(0);
    int mx = 0, id = 0;
    p[0] = 0;
    for(int i = 1, k; i < len; ++i) {
        if(mx > i) {
            k = min( mx - i, p[2 * id  - i]);
        } else {
            k = 1;
        }
        for(; s[i - k] == s[i + k] && i + k <= len && i - k > 0; ++k, ++cnt);
        p[i] = k - 1;
        if(p[i] + i > mx) {
            mx = p[i] + i;
            id = i;
        }
    }
}
拓展KMP：
void ext_kmp(int lenT) {
    int i(0), j(0), k(1);
    for(; S[j+ 1] && S[j] == S[j+ 1]; ++j)  ;
    A[0] = lenT;
    A[1] = j;
    for(i= 2; i< lenT; ++i) {
        int Len= k+ A[k] - 1;
        int L= A[i- k];
        if(L< Len- i+ 1)
            A[i] = L;
        else {
            j = max(0, Len- i+ 1);
            for(; i+ j< lenT&& S[i+ j] == S[j]; ++j) ;
            A[i] = j;
            k= i;
        }
    }//    printf("%s\n", S);
//    for(int i = 0; i < lenT; ++i)
//        printf("%d %d\n", i, A[i]);
}
void ext_kmp(int lenT) {

    int i(0), j(0), k(0);
    S[lenT] = 0;
    for(i= 1, A[0] = 0; i< lenT; ++i) {
        int Len = k + A[k] - 1;
        int L = A[i - k];
        if(L < Len- i + 1)
            A[i] = L;
        else {
            j = max(0, Len - i + 1);
            for(; i+ j < lenT && S[i+ j] == S[j]; ++j) ;
            A[i] = j;
            k= i;
        }
    }
    A[0] = lenT;
}
拓展gcd:
LL ex_gcd(LL a, LL b, LL&x, LL&y) {
    if(b== 0) {
        x= 1;
        y= 0;
        return a;
    }
    LL tx, ty;
    LL g= ex_gcd(b, a% b, tx, ty);
    x= ty;
    y = (tx- a/ b* ty) % mod;
    return g;
}
卢卡斯大数的组合数模P：ret = C(n%p, m%p)*lucas(n/p, m/p);
n和n表示成p进制相应位置的组合数的积
int lucas(int n,int m)//lucas定理{
    int ret=1;
    while(n&&m&&ret){
        ret=ret*C(n%p,m%p)%p;
        n/=p;
        m/=p;
    }
    return ret;
}
异或方程组枚举自由变量：
constint MAXN= 110;
int a[MAXN][MAXN], b[MAXN];
int ans;
void dfs(int k,int var,int cnt) {
    if(k < 0) {
        ans= min(ans, cnt);
        return;
    }
    if(a[k][k] == 0) {
        a[k][var] = 1;
        dfs(k- 1, var, cnt+ 1);
        a[k][var] = 0;
        dfs(k- 1, var, cnt);
    } else {
        int b= a[k][var];
        for(int j= k+ 1; j<= var; ++j)
            if(a[k][j])   a[k][var] ^= a[j][var];
        dfs(k- 1, var, cnt+ a[k][var]);
        a[k][var] = b;
    }
}
int gauss(int equ,int var) {
    int i, j, k(0), col(0), max_r;
    for(; k< equ&& col< var; ++k, ++col) {
        max_r = k;
        for(i= k; i< equ; ++i)
            if(a[i][col] > a[max_r][col])  max_r= i;
        for(j= col; j<= var&& max_r!= k; ++j)
            swap(a[k][j], a[max_r][j]);
        if(a[k][col] == 0) {
            --k;
            continue;
        }
        for(i= k+ 1; i< equ; ++i) {
            if(a[i][col]) {
                for(j= col; j<= var; ++j)
                    a[i][j] ^= a[k][j];
            }
        }
    }
    for(i= k; i< equ; ++i)
        if(a[i][var]) return-1;
//  将矩阵调成一个严格的阶梯矩阵
    for(int i= 0; i< equ; ++i) {
        if(a[i][i]) continue;
        col= -1;       // 将第i行换到j行 i,j之间的行相应的向上平移
        for(j= 0; j< var&& col== -1; ++j)
            if(a[i][j])    col= j;
        if(col== -1) continue;
        for(int k1= equ- 1; k1>= i; --k1)
            for(j= 0; j<= var; ++j) {
                if(k1>= i+ col- i)    a[k1][j] = a[k1+ i- col][j];
                else  a[k1][j] = 0;
            }
    }
    dfs( equ- 1, var, 0);
    return 1;
}
----------------------------------------------------------------------------------
gauss整数消除无关变量  ZOJ3636
#include <stdio.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <map>
#include <queue>

using namespace std;

typedef long long LL;

const int MAXN = 21000;
const int INF = 0;
const int L = 30;

int n, len;
int id[ L + 10 ];
int a[ MAXN ];

int toInt(char *s)
{
    int res(0);
    for(int i = 0; i < len; ++i) if(s[i] == '1') res += (1<<(len - i - 1));
    return res;
}
char s[50];
void toString(int x){
    for(int i = 0; i < len; ++i)  s[i] = ((x >> (len - i - 1) ) & 1) + '0';
    s[len] = 0;
}

void readin(){

    for(int i = 0; i < n; ++i){
        scanf("%s", s);
        a[i] = toInt(s);
    }
    sort(a, a + n);
    reverse(a, a + n);
    for(int i = 0; i < n; ++i){

        if(a[i] == 0)  continue;

        int j(len - 1);
        for(; j >= 0; --j)  if(a[i] & (1<<j))  break;

        for(int k = 0; k < n; ++k){
            if(k != i && (a[k] & (1<<j)))  a[k] ^= a[i];
        }
    }
    memset(id, -1, sizeof(id));
    for(int i = 0; i < n; ++i){
        if( a[i] ){
//            toString(a[i]);
//            printf("%d %s\n", i, s);
            for(int j = len - 1; j >= 0; --j){
                if( a[i] & (1<<j) ) {
                    id[j] = i;
                    break;
                }
            }
        }
    }
}
int ans, want;
bool dfs(int k, int x, int lt){

    if(lt < 0)  return 0;
    if(k < 0) {
        ans = x;
        return 1;
    }

    int b1 = (x>>k) & 1;
    int b2 = (want>>k) & 1;

    if(b1 && id[k] != -1 && dfs(  k - 1, x ^ a[ id[k] ], lt - (b2 == 1) ) )  return 1;   // lexicographic
    if(!b1 && dfs(k - 1, x, lt - (b2 == 1)))  return 1;

    if(!b1 && id[k] != -1 && dfs( k - 1, x ^ a[ id[k] ], lt - (b2 == 0) ) ) return 1;
    if(b1 && dfs(k - 1, x, lt - (b2 == 0) ) )  return 1;

//    if(b1 == b2 && dfs(k - 1, x, lt ) ) return 1;
//    if(b1 == b2 && id[k] != - 1 && dfs(k - 1, x ^ a[ id[k] ], lt - 1) ) return 1;
//    if(b1 != b2 && id[k] != -1 &&  dfs(k - 1, x ^ a[ id[k] ], lt ))  return 1;
//    if(b1 != b2 && dfs(k - 1, x, lt - 1)) return 1;

    return 0;
}

int slove(){

//    toString(want);
//    printf("%s\n", s);
    if(want == 0)  return 0;
    int cnt(0);
    for(int i = 0; i < len; ++i){
        if(want & (1<<i)) ++cnt;
    }
    ans = -1;
    for(int i = 0; i <= 3; ++i){
        if(cnt == i)  return 0;
        int x = want & ( 1 << (len - 1) );
        if(x && id[len - 1] != -1)  {
            x = a[ id[ len - 1] ];
            if( dfs(len - 1, x, i ) )  return ans;

        }else  if(!x) {
            if( dfs(len - 1, 0, i ) )  return ans;
        }
    }

    return -1;
}

int main(){

//    freopen("std.in", "r", stdin);
    int q;
    while(~scanf("%d%d%d", &len, &n, &q)){
        readin();
        for(int i = 0; i < q; ++i){
            scanf("%s", s);
            want = toInt(s);
            int res = slove();
            if(res == -1)  puts("NA");
            else {
                toString(res);
                printf("%s\n", s);
            }
        }
    }



    return 0;
}
中国剩余定理：
#include <iostream>
using namespace std;

int Extended_Euclid(int a,int b,int &x,int &y) {  //扩展欧几里得算法
    int d;
    if(b==0) {
        x=1;
        y=0;
        return a;
    }
    d=Extended_Euclid(b,a%b,y,x);
    y-=a/b*x;
    return d;
}

int Chinese_Remainder(int a[],int w[],int len) {  //中国剩余定理  a[]存放余数  w[]存放两两互质的数
    int i,d,x,y,m,n,ret;
    ret=0;
    n=1;
    for (i=0; i<len; i++)
        n*=w[i];
    for (i=0; i<len; i++) {
        m=n/w[i];
        d=Extended_Euclid(w[i],m,x,y);
        ret=(ret+y*m*a[i])%n;
    }
    return (n+ret%n)%n;
}


int main() {
    int n,i;
    int w[15],b[15];
    while (scanf("%d",&n),n) {
        for (i=0; i<n; i++) {
            scanf("%d%d",&w[i],&b[i]);
        }
        printf("%d/n",Chinese_Remainder(b,w,n));
    }
    return 0;
}



线段树 延迟
LL sum[MAXN], ssum[MAXN], Max[MAXN], Min[MAXN];
LL add[MAXN], clr[MAXN], child[MAXN];
LL ele[MAXN];

inline LL getSsum(int id) {
    return ssum[id] + child[id] *  add[id] *  add[id] + 2 * add[id] * sum[id];
}

inline void up(int id, int lc, int rc) {
    sum[id] = sum[lc] + sum[rc] + add[lc] * child[lc] + add[rc] * child[rc];
    ssum[id] = getSsum(lc) + getSsum(rc);
    Max[id] = max(Max[lc] + add[lc], Max[rc] + add[rc]);
    Min[id] = min(Min[lc] + add[lc], Min[rc] + add[rc]);
}
inline void down(int id, int lc, int rc) {
    if(clr[id] != INF) {
        add[lc] = add[rc] = 0;
        sum[lc] = child[lc] * clr[id];
        sum[rc] = child[rc] * clr[id];
        clr[lc] = clr[rc] = clr[id];
        ssum[lc] = child[lc] * clr[id] * clr[id];
        ssum[rc] = child[rc] * clr[id] * clr[id];
        Max[lc] = Max[rc] = Min[lc] = Min[rc] = clr[id];

        clr[id] = INF;
    }
    add[lc] += add[id];
    add[rc] += add[id];
    add[id] = 0;
}
void build(int id, int lt, int rt) {
    child[id] = rt - lt + 1;
    clr[id] = INF;
    add[id] = 0;
    ssum[id] = sum[id] = 0;
    if(lt == rt) {
        Min[id] = Max[id] = sum[id] = ele[lt];
        ssum[id] = ele[lt] * ele[lt];
        add[id] = 0;
        clr[id] = ele[lt];
//        printf(">>%d %d %I64d %I64d %I64d\n", lt, rt, ele[lt], Max[id], Min[id]);
        return;
    }
    build(id<<1, lt, (lt+rt)>>1);
    build((id<<1) + 1, (lt + rt) / 2 + 1, rt);
    up(id, id<<1, (id<<1) + 1);
//    printf("%d  %d  %lld %lld\n", lt, rt, ssum[id], sum[id]);
}
void ins(int id, int lt, int rt, int l, int r, int typ, LL val) {
    if(l > rt || r < lt) return;
    if(l <= lt && r >= rt) {
        if(typ == 0) {
            clr[id] = val;
            sum[id] = child[id] * val;
            ssum[id] = child[id] * val * val;
            Max[id] = Min[id] = val;
            add[id] = 0;
        } else {
            add[id] += val;
        }
        return;
    }
    int mid = (lt+rt)>>1, lc(id<<1), rc = (id<<1) + 1;
    down(id, lc, rc);
    ins(lc, lt, mid, l, r, typ, val);
    ins(rc, mid + 1, rt, l, r, typ, val);
    up(id, lc, rc);
}
LL qMx, qMn;
void query(int id, int lt, int rt, int l, int r, LL &s1, LL &s2) {
    if(l > rt || r < lt) {
        s1 = s2 = 0;
        return;
    }
    if(l <= lt && r >= rt) {
        qMx = max(qMx, Max[id] + add[id]);
        qMn = min(qMn, Min[id] + add[id]);
        s1 = getSsum(id);
        s2 = sum[id] + child[id] * add[id];
//        printf(">>%d %d %d %d %lld  %lld\n", lt, rt, l, r, s1, s2);
        return;
    }
    int mid = (lt + rt) / 2, lc = (id<<1), rc = (id<<1) + 1;
    down(id, lc, rc);
    LL ts1, ts2, tts1, tts2;
    query(lc, lt, mid, l, r, ts1, ts2);
    query(rc, mid + 1, rt, l, r, tts1, tts2);
    s1 = ts1 + tts1;
    s2 = ts2 + tts2;
    up(id, lc, rc);
//    printf("%")
}

优先队列：
struct cmp {
    bool operator()（const int &a,const int &b） {
        return a>b;//最大堆
        return a<b;//最小堆
    }
};
priority_queue< int, vector<int>, cmp >
//priority_queue比较函数
struct cmp {
    bool operator()(const T &a,const T&b) {
        return a>b;//最大堆
        return a<b;//最小堆
    }
}；

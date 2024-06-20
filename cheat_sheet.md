二分查找

```python
import bisect
bisect.bisect_left(lst,x)
# 使用bisect_left查找插入点，若x∈lst，返回最左侧x的索引；否则返回最左侧的使x若插入后能位于其左侧的元素的当前索引。
bisect.bisect_right(lst,x)
# 使用bisect_right查找插入点，若x∈lst，返回最右侧x的索引；否则返回最右侧的使x若插入后能位于其右侧的元素的当前索引。
bisect.insort(lst,x)
# 使用insort插入元素，返回插入后的lst
```

排序

```python
def insertion_sort(arr):#向前比对，若不符合要求则交换，一直交换直到位置正确。在列表较低的一端维护一个有序的子列表
    for i in range(1, len(arr)):
        j = i																				  
        # Insert arr[j] into the sorted sequence arry[0..j-1]
        while arr[j - 1] > arr[j] and j > 0:
            arr[j - 1], arr[j] = arr[j], arr[j - 1] 
            j -= 1
```

```python
def bubbleSort(alist):
  for passnum in range(len(alist)-1, 0, -1):
    for i in range(passnum):
     if alist[i] > alist[i+1]:
      temp = alist[i]
      alist[i] = alist[i+1]
      alist[i+1] = temp
```

```python
def selection_sort(A):
# Traverse through all array elements
 for i in range(len(A)):
    # Find the minimum element in remaining unsorted array
    min_idx = i
    for j in range(i + 1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
        # Swap the found minimum element with the first element
    A[i], A[min_idx] = A[min_idx], A[i]
```

```python
def quicksort(arr, left, right):
  if left < right:
    partition_pos = partition(arr, left, right)
  quicksort(arr, left, partition_pos - 1)
  quicksort(arr, partition_pos + 1, right)
def partition(arr, left, right):
  i = left
  j = right - 1
  pivot = arr[right]#以最右侧数据为pivot
  while i <= j:
    while i <= right and arr[i] < pivot:
      i += 1
    while j >= left and arr[j] >= pivot:
      j -= 1
    if i < j:
      arr[i], arr[j] = arr[j], arr[i]#双指针
    if arr[i] > pivot:
      arr[i], arr[right] = arr[right], arr[i]
  return i
```

```python
def mergeSort(arr):#分治算法，分别排两半。先递归地排完左再排右
	if len(arr) > 1:
		mid = len(arr)//2
		L = arr[:mid]	# Dividing the array elements
		R = arr[mid:] # Into 2 halves
		mergeSort(L) # Sorting the first half
		mergeSort(R) # Sorting the second half
		i = j = k = 0
		# Copy data to temp arrays L[] and R[]
		while i < len(L) and j < len(R):#把数据按顺序放回arr
			if L[i] <= R[j]:
				arr[k] = L[i]
				i += 1
			else:
				arr[k] = R[j]
				j += 1
			k += 1
		# Checking if any element was left
		while i < len(L):
			arr[k] = L[i]
			i += 1
			k += 1
		while j < len(R):
			arr[k] = R[j]
			j += 1
			k += 1
```

```python
def shellSort(arr, n):
    gap = n // 2
#将列表分成数个子列表，并对每一个子列表应用插入排序。并不是连续切分，而是使用增量i（有时称作步长）选取所有间隔为i 的元素组成子列表。插入排序每次只往前移一个位置，希尔排序每次相当于往前移i个。
    while gap > 0:
        j = gap
        # Check the array in from left to right Till the last possible index of j
        while j < n:
            i = j - gap  # This will keep help in maintain gap value
            while i >= 0:
                # If value on right side is already greater than left side value
                # We don't do swap else we swap
                if arr[i + gap] > arr[i]:
                    break
                else:
                    arr[i + gap], arr[i] = arr[i], arr[i + gap]

                i = i - gap  # To check left side also
            # If the element present is greater than current element
            j += 1
        gap = gap // 2
```

实现堆结构

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]#初始有一个0
        self.currentSize = 0
    def push(self,nums):
        self.heapList.append(nums)
        self.currentSize+=1
        i=self.currentSize#所以这里比之前自己手搓的多一
        while i//2>0:
            if nums<self.heapList[i//2]:
                self.heapList[i//2],self.heapList[i]=self.heapList[i],self.heapList[i//2]
            i=i//2

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                self.heapList[i],self.heapList[mc]=self.heapList[mc],self.heapList[i]
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.percDown(1)
        return retval
```

找质数：欧拉筛

```python
def euler(r):
    prime = [0 for i in range(r+1)]
    common = []
    for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
            if i % j == 0:
                break
    return prime 

s = euler(1000000)
#print(s)

input()
for i in map(int,input().split()):
    if i<4:
        print('NO')
        continue
    elif int(i**0.5)**2 != i:
        print('NO')
        continue
    if s[int(i**0.5)]==0:
        print('YES')
    else:
        print('NO')
```

最大公因数

```python
def gcd(m,n):
    while m%n != 0:
        oldm,oldn = m,n
        m,n = oldn,oldm%oldn
    return n
```

### 并查集

```python
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n#这是啥
    def find(self,x):#一般只写find和union两个函数即可，不必建类
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])#路径压缩
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.h[rootx]<self.h[rooty]:
                self.p[rootx]=rooty
            elif self.h[rootx]>self.h[rooty]:
                self.p[rooty]=rootx
            else:
                self.p[rooty]=rootx
                self.h[rootx]+=1
```

### 栈

shunting yard

从左到右遍历中缀表达式的每个符号。
如果是操作数（数字），则将其添加到输出栈。
如果是左括号，则将其推入运算符栈。
如果是运算符：如果运算符的优先级⼤于运算符栈顶的运算符，或者运算符栈顶是左括号，则将当前运算符推入运算符栈。否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满⾜上述条件（或者运算符栈为空）。
将当前运算符推入运算符栈。
如果是右括号，则将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号。将左括号弹出但不添加到输出栈中。
遍历完成后，如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中。
输出栈中的元素就是转换后的后缀表达式。

```python
pre={'+':1,'-':1,'*':2,'/':2}
for _ in range(int(input())):
    expr=input()
    ans=[]; ops=[]
    for char in expr:
        if char.isdigit() or char=='.':
            ans.append(char)
        elif char=='(':
            ops.append(char)
        elif char==')':
            while ops and ops[-1]!='(':
                ans.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1]!='(' and pre[ops[-1]]>=pre[char]:
                ans.append(ops.pop())
            ops.append(char)
    while ops:
        ans.append(ops.pop())
    print(''.join(ans))
```

单调栈

```python
n = int(input())
a = list(map(int, input().split()))
stack = []

#f = [0]*n
for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        #f[stack.pop()] = i + 1
        a[stack.pop()] = i + 1


    stack.append(i)

while stack:
    a[stack[-1]] = 0
    stack.pop()

print(*a)
```

### 树

前中序得后序

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
```

中后序得前序

```python
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

多叉树--二叉树

```python
class B_node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
  
class T_node:
    def __init__(self, value):
        self.value = value
        self.children = []
  
def to_b_tree(t_node):
    if t_node is None: return None
    b_node = B_node(t_node.value)
    if len(t_node.children) > 0: b_node.left = to_b_tree(t_node.children[0])
    current_node = b_node.left
    for child in t_node.children[1:]: current_node.right = to_b_tree(child); current_node = current_node.right
    return b_node

def to_tree(b_node):
    if b_node is None: return None
    t_node, child = T_node(b_node.value), b_node.left
    while child is not None: t_node.children += to_tree(child); child = child.right
    return t_node
```

字典树

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1
```

后序还是先左后右！！！

### 图

bfs

```python
from collections import deque
def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    visited.add(start_node)
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

dijkstra

```python
def dijkstra(start,end):
    heap=[(0,start,[start])]
    vis=set()
    while heap:
        (cost,u,path)=heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==end: return (cost,path)
        for v in graph[u]:
            if v not in vis:
                heappush(heap,(cost+graph[u][v],v,path+[v]))
```

floyd_warshall 多源 or 负权

```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif j in graph[i]:
                dist[i][j] = graph[i][j]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])###

    return dist
```

kruskal

```python
n,m=map(int,input().split())
edges=[]
parents=[i for i in range(n)]
def find(x):
    if parents[x]!=x:
        parent[x]=find(parents[x])
    return parent[x]
def union(x,y):
    xp=find(x)
    yp=find(y)
    if xp!=yp:
        parents[yp]=xp
for _ in range(m):
    s,e,w=input().split()
    edges.append((float(w),int(s),int(e)))
edges.sort()
result=[]
cost=0
for edge in edges:
    w,s,e=edge
    if find(s)!=find(e):
        union(s,e)
        result+=[(min(s,e),max(s,e))]
        cost+=w

t=set()
for i in range(n):
    t.add(find(i))
if len(t)>1:
    print('NOT CONNECTED')
else:
    print('%.2f'%cost)
    result.sort()
    for i in result:
        print(*i)
```

prim

```python
vis=[0]*n
q=[(0,0)]
ans=0
while q:
    w,u=heappop(q)
    if vis[u]:
        continue
    ans+=w
    vis[u]=1
    for v in range(n):
        if not vis[v] and graph[u][v]!=-1:
            heappush(q,(graph[u][v],v))
print(ans)
```

拓扑排序

```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in graph}#计算入度
#用列表亦可
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return []  
    return topo_order
```

### dp

```python
#0-1背包的memory简化
f[i][l]=max(f[i-1][l],f[i-1][l-w[i]]+v[i])#这要二维数组i为进行到第i个物品，l为最大容量
for i in range(1, n + 1):#这时只需要一维，l为最大容量，通过反复更新维护
    for l in range(W, w[i] - 1, -1):#必须这样逆序，要让每个f只被更新一次
        f[l] = max(f[l], f[l - w[i]] + v[i])
```

```python
#完全背包（每件物品可以选择任意次）
f[i][l]=max(f[i-1][l],f[i][l-w[i]]+v[i])#这要二维数组i为进行到第i个物品，l为最大容量
for i in range(1, n + 1):#这时只需要一维，l为最大容量，通过反复更新维护
    for l in range(0, W - w[i] + 1):#此时要正序，根本原因是可以多次选择
        f[l + w[i]] = max(f[l] + v[i], f[l + w[i]])
```

```python
#多重背包（物品选择指定次）
#朴素想法转化为0-1背包，可能超时，因此考察二进制拆分（先尽力拆为1，2，4，8...)
import math
k=int(math.log(x,2))
    for i in range(k+2):
        if x>=2**i:
            x-=2**i
            coi.append(y*(2**i))
        else:
            coi.append(x*y)
            break
```

扩栈

```python
import sys
sys.setrecursionlimit(1<<30)
```

集合

```python
a - b                              # 集合a中包含而集合b中不包含的元素
a | b                              # 集合a或b中包含的所有元素
a & b                              # 集合a和b中都包含了的元素
a ^ b                              # 不同时包含于a和b的元素
```

调试技巧

1. **对于TLE**:
   - 检查是否有冗余或低效的操作。
   - 如果没有明显的效率问题，可能需要改进算法。
2. **对于RE**:
   - 检查数组越界、空指针访问、栈溢出等常见错误。
   - `min()` 和 `max()` 函数不能对空列表进行操作
3. **对于WA**:
   - 仔细检查代码逻辑，特别是循环和条件判断。
   - 确保所有初始化正确，边界条件得到处理。

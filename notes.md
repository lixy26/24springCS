## 背经典算法

概念、树的表示、类的写法

#### 时间复杂度

![Rate of Growth of Algorithms](https://raw.githubusercontent.com/GMyhf/img/main/img/202402232027181.png)

### 排序

#### Insertion sort 插入排序

1. 向前比对，若不符合要求则交换，一直交换直到位置正确。在列表较低的一端维护一个有序的子列表
2. 时间复杂度n**2
3. 最好情况：数据正序，n；最坏情况：数据逆序

   ```python
   def insertion_sort(arr):#向前比对，若不符合要求则交换，一直交换直到位置正确。在列表较低的一端维护一个有序的子列表
       for i in range(1, len(arr)):
           j = i																				  
           # Insert arr[j] into the sorted sequence arry[0..j-1]
           while arr[j - 1] > arr[j] and j > 0:
               arr[j - 1], arr[j] = arr[j], arr[j - 1] 
               j -= 1
   ```

#### **Bubble sort 冒泡排序**

1. 原位排序，不使用额外空间。两个两个遍历，不符合要求则交换
2. 时间复杂度O（n**2）
3. 最好情况：数据正序，n；最坏情况：数据逆序

```python
def bubbleSort(alist):
  for passnum in range(len(alist)-1, 0, -1):
    for i in range(passnum):
     if alist[i] > alist[i+1]:
      temp = alist[i]
      alist[i] = alist[i+1]
      alist[i+1] = temp
```

#### **Selection sort 选择排序**

每次选择最大值（或最小值）并将其置于未排序列表最后（或最前，和未排序列表的第一个/最后一个元素交换）

最坏情况：倒序，每次选择最小（或最大）元素时，需要遍历未排序部分的所有元素，导致时间复杂度为O(n^2)

时间复杂度O（N2），空间复杂度1，不稳定

#### Shell sort  希尔排序

将列表分成数个子列表，并对每一个子列表应用插入排序。并不是连续切分，而是使用增量i（有时称作步长）选取所有间隔为i 的元素组成子列表。插入排序每次只往前移一个位置，希尔排序每次相当于往前移i个。The idea of ShellSort is to allow the exchange of far items.不断减小i直到它变成1

当数据量大，插入排序爆栈时改用希尔排序

时间复杂度n2，空间复杂度1，不稳定

希尔排序的性能不是简单地由数据的初始顺序决定的

```python
def shellSort(arr, n):
    gap = n // 2
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

#### Quick sort 快速排序

分治算法

选择一个pivot，遍历列表，将小于pivot的元素置于左侧，大于的元素置于右侧；对两侧的列表重复该操作

时间复杂度nlogn，最坏n2（The worst-case Scenario for Quicksort occur when the pivot at each step consistently results in highly  **unbalanced ** partitions. When the array is already sorted and the pivot is always chosen as the smallest or largest element.->choose the right pivot）

空间复杂度logn

不稳定

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

#### Merge sort 归并排序

分治算法，分别排两半。先递归地排完左再排右

外部排序算法，适合主存储器空间有限的情况，不需要频繁的磁盘访问

时间复杂度nlogn

空间复杂度n

建堆logn？

### 总结


|        Name        | Best | Average | Worst | Memory | Stable |       Method       |                                                              Other notes                                                              |
| :-----------------: | :---: | :-----: | :----: | :----: | :----: | :-----------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| In-place merge sort |  —  |   —   | nlog2n |   1   |  Yes  |       Merging       |                                 Can be implemented as a stable sort based on stable in-place merging.                                 |
|      Heapsort      | nlogn |  nlogn  | nlogn |   1   |   No   |      Selection      |                                                                                                                                      |
|     Merge sort     | nlogn |  nlogn  | nlogn |  *n*  |  Yes  |       Merging       |                            Highly parallelizable (up to*O*(log *n*) using the Three Hungarian's Algorithm)                            |
|       Timsort       |  *n*  |  nlogn  | nlogn |  *n*  |  Yes  | Insertion & Merging |                               Makes*n-1* comparisons when the data is already sorted or reverse sorted.                               |
|      Quicksort      | nlogn |  nlogn  |   n2   |  logn  |   No   |    Partitioning    |                                   Quicksort is usually done in-place with*O*(log *n*) stack space.                                   |
|      Shellsort      | nlogn |  n4/3  |  n3/2  |   1   |   No   |      Insertion      |                                                           Small code size.                                                           |
|   Insertion sort   |  *n*  |   n2   |   n2   |   1   |  Yes  |      Insertion      |                                *O*(n + d), in the worst case over sequences that have *d* inversions.                                |
|     Bubble sort     |  *n*  |   n2   |   n2   |   1   |  Yes  |     Exchanging     |                                                            Tiny code size.                                                            |
|   Selection sort   |  n2  |   n2   |   n2   |   1   |   No   |      Selection      | Stable with O(n) extra space, when using linked lists, or when made as a variant of Insertion Sort instead of swapping the two items. |

在排序算法中，稳定性是指相等元素的相对顺序是否在排序后保持不变。换句话说，当有两个相等的元素A和B，且在排序前A出现在B的前面，在稳定算法排序后A仍然会出现在B的前面。

对于判断一个排序算法是否稳定，一种常见的方法是观察交换操作。挨着交换（相邻元素交换）是稳定的，而隔着交换（跳跃式交换）可能会导致不稳定性。

### 找质数：经典算法

求解素数的三种方法，包括：试除法（trial division）、埃氏筛（Sieve of Eratosthenes）、欧拉筛（Sieve of Euler，线性法），[https://blog.dotcpp.com/a/69737](https://blog.dotcpp.com/a/69737)

数据类型时间复杂度，[https://wiki.python.org/moin/TimeComplexity](https://wiki.python.org/moin/TimeComplexity)

埃氏筛法，时间复杂度为：O(n\*logn)。Python3, Accepted, 1154ms

```python
# http://codeforces.com/problemset/problem/230/B

# https://www.geeksforgeeks.org/python-program-for-sieve-of-eratosthenes/
# Python program to print all primes smaller than or equal to 
# n using Sieve of Eratosthenes 

def SieveOfEratosthenes(n, prime): 
    p = 2
    while (p * p <= n): 
  
    	# If prime[p] is not changed, then it is a prime 
    	if (prime[p] == True): 
  
        	# Update all multiples of p 
        	for i in range(p * 2, n+1, p): 
            	prime[i] = False
    	p += 1

n = int(input())
x = [int(i) for i in input().split()]

s = [True]*(10**6+1)

SieveOfEratosthenes(10**6, s)

for i in x:
    if i<4:
        print('NO')
        continue
    elif int(i**0.5)**2 != i:
        print('NO')
        continue
    print(['NO','YES'][s[int(i**0.5)]])
    #if s[int(i**0.5)]:
    #    print('YES')
    #else:
    #    print('NO')
```

小优化（原因如下，下面用到集合实现），第15行可以写成 **for** i **in** range(p \* p, n+1, p): 则998ms可以AC.

埃氏筛法，时间复杂度为：O(n\*loglogn)。Python3, Accepted, 1558ms

这里有一个小优化，j 从 i \* i 而不是从 i + i开始，因为 i\*(2\~ i-1)在 2\~i-1时都已经被筛去，所以从i \* i开始。

According to [Python wiki: Time complexity](https://wiki.python.org/moin/TimeComplexity), **set** is implemented as a [hash table](https://en.wikipedia.org/wiki/Hash_table). So you can expect to lookup/insert/delete in **O(1)** average. [https://stackoverflow.com/questions/7351459/time-complexity-of-python-set-operations](https://stackoverflow.com/questions/7351459/time-complexity-of-python-set-operations)

```python
n = 1000000
a = [1] * n
s = set() 

#directly add the square of prime into a set, then check if num_input is in set.
for i in range(2,n):
    if a[i]:
        s.add(i*i)
        for j in range(i*i,n,i):
            a[j] = 0

input()
for x in map(int,input().split()):
    print(["NO","YES"][x in s])
```

埃氏筛法，by 2020fall-cs101, 汪元正, Python3, Accepted, 1340ms

```python
a = [1]*(10**6)
a[0] = 0
for i in range(1,10**3,1):
    if a[i]==1:
        for j in range(2*i+1,10**6,i+1):
            a[j]=0

n = int(input())
l = [int(x) for x in input().split()]
for i in range(n):
    m = l[i]
    if m**0.5%1==0:
        r = int(m**0.5)
        if a[r-1]==1:
            print('YES')
        else:
            print('NO')
    else:
        print('NO')
```

线性筛（欧拉筛），时间复杂度为：O(n)。Python3, Accepted, 1808ms。

```python
# https://blog.dotcpp.com/a/69737
# https://blog.csdn.net/xuechen_gemgirl/article/details/79555123
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

### shunting yard调度场算法：将中序表达式转换为后序表达式

从左到右遍历中缀表达式的每个符号。
如果是操作数（数字），则将其添加到输出栈。
如果是左括号，则将其推入运算符栈。
如果是运算符：如果运算符的优先级⼤于运算符栈顶的运算符，或者运算符栈顶是左括号，则将当前运算符推入运算符栈。否则，将运算符栈顶的运算符弹出并添加到输出栈中，直到满⾜上述条件（或者运算符栈为空）。
将当前运算符推入运算符栈。
如果是右括号，则将运算符栈顶的运算符弹出并添加到输出栈中，直到遇到左括号。将左括号弹出但不添加到输出栈中。
遍历完成后，如果还有剩余的运算符在运算符栈中，将它们依次弹出并添加到输出栈中。
输出栈中的元素就是转换后的后缀表达式。

```
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}##优先级判断！！
    stack = []
    postfix = []
    number = ''
    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)#区分小数和整数
                number = ''
            if char in '+-*/':
                 while stack and stack[-1] in '+-*/' and precedence[char] <=precedence[stack[-1]]:
                    postfix.append(stack.pop())
                 stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()
    if number:
            num = float(number)
            postfix.append(int(num) if num.is_integer() else num)
    while stack:
            postfix.append(stack.pop())
    return ' '.join(str(x) for x in postfix)
```

DFS:与stack有关

```python
import sys

def ret_graph():
    return {
        'A': {'B':5.5, 'C':2, 'D':6},
        'B': {'A':5.5, 'E':3},
        'C': {'A':2, 'F':2.5},
        'D': {'A':6, 'F':1.5},
        'E': {'B':3, 'J':7},
        'F': {'C':2.5, 'D':1.5, 'K':1.5, 'G':3.5},
        'G': {'F':3.5, 'I':4},
        'H': {'J':2},
        'I': {'G':4, 'J':4},
        'J': {'H':2, 'I':4},
        'K': {'F':1.5}
    }

start = 'A'   
dest = 'J'  
visited = []  
stack = []  
graph = ret_graph()
path = []


stack.append(start)  
visited.append(start)  
while stack:   
    curr = stack.pop()  
    path.append(curr)
    for neigh in graph[curr]:  
        if neigh not in visited:   
            visited.append(neigh)   
            stack.append(neigh)   
            if neigh == dest :  
                print("FOUND:", neigh)
                print(path)
                sys.exit(0)
print("Not found")
print(path)
```

BFS：与queue有关,即按层次遍历

```python
def print_tree(node):#层次遍历
    Q = []
    s = []
    while node:
        if node.value != '$':
            Q.append(node)
        node = node.children[1] if len(node.children) > 1 else None
    while Q:
        node = Q.pop(0)
        print(node.value, end=' ')
        if node.children:
            node = node.children[0]
            while node:
                if node.value != '$':
                    Q.append(node)
                node = node.children[1] if len(node.children) > 1 else None

```

```python

def bfs(start):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()#deque：popleft时间复杂度O（1），比list pop快
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()  # 取队首作为当前顶点
        for neighbor in current.get_neighbors():   # 遍历当前顶点的邻接顶点
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black" # 当前顶点已经处理完毕，设黑色
```

### 二叉堆的实现

结构性：完全二叉树，可以用一个列表来表示它，而不需要采用“列表之列表”或“节点与引用”表示法。由于树是完全的，因此对于在列表中处于位置 p 的节点来说，它的左子节点正好处于位置 2p；同理，右子节点处于位置 2p+1。若要找到树中任意节点的父节点，只需使用 Python 的整数除法即可。给定列表中位置 n 处的节点，其父节点的位置就是 n/2。图 2 展示了一棵完全二叉树，并给出了列表表示。树的列表表示——加上这个“完全”的结构性质——让我们得以通过一些简单的数学运算遍历完全二叉树。我们会看到，这也有助于高效地实现二叉堆。

有序性：最小堆

时间复杂度nlogn，最优n，不稳定（有相同元素时）

应用：贪心，Huffman编码

实现：

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]#列表 heapList 的第一个元素是 0，为了使后续的方法可以使用整数除法
        self.currentSize = 0
    def percUp(self,i):#如果新元素小于其父元素，就将二者交换
       while i // 2 > 0:
        if self.heapList[i] < self.heapList[i // 2]:
         tmp = self.heapList[i // 2]
         self.heapList[i // 2] = self.heapList[i]
         self.heapList[i] = tmp
        i = i // 2
     def insert(self,k):
       self.heapList.append(k)
       self.currentSize = self.currentSize + 1
       self.percUp(self.currentSize)
     def percDown(self,i):
       while (i * 2) <= self.currentSize:
        mc = self.minChild(i)
        if self.heapList[i] > self.heapList[mc]:
            tmp = self.heapList[i]
            self.heapList[i] = self.heapList[mc]
            self.heapList[mc] = tmp
        i = mc
     def minChild(self,i):
       if i * 2 + 1 > self.currentSize:
        return i * 2
       else:
        if self.heapList[i*2] < self.heapList[i*2+1]:
            return i * 2
        else:
            return i * 2 + 1
#第一步，取出列表中的最后一个元素，将其移到根节点的位置，保证堆的结构性质
#第二步，将新的根节点沿着树推到正确的位置，以重获堆的有序性
     def delMin(self):
       retval = self.heapList[1]
       self.heapList[1] = self.heapList[self.currentSize]
       self.currentSize = self.currentSize - 1
       self.heapList.pop()
       self.percDown(1)
       return retval
     def buildHeap(self,alist):
       i = len(alist) // 2  # 超过中点的节点都是叶子节点
       self.currentSize = len(alist)
       self.heapList = [0] + alist[:]
       while (i > 0):
        self.percDown(i)
        i = i - 1
```

### Huffman Coding 哈夫曼编码

Huffman算法是一种用于数据压缩的算法，它通过构建最优的可变长度前缀编码来实现文本压缩。

该算法使用短的码词字符串来编码高频字符和长的码词字符串来编码低频字符，从而节省了固定长度编码的空间。

该算法基于构建表示该编码的二叉树T。T中的每条边表示码词中的一个位，指向左子节点的边表示“0”，指向右子节点的边表示“1”。每个叶节点v与特定字符相关联，该字符的码词由从T的根到v的路径上与边相关联的位序列定义。（见图13.9）每个叶节点v都有一个频率f(v)，它简单地是与v相关联的字符在X中的频率。此外，我们为T中的每个内部节点v赋予一个频率f(v)，该频率是v所在子树中所有叶节点频率的总和。

Huffman算法的实现步骤如下：

1. 统计字符频率：遍历待压缩的文本数据，统计每个字符出现的频率。
2. 构建字符节点：为每个字符创建一个叶节点，并将频率作为节点权重。
3. 构建Huffman树：重复以下步骤，直到只剩下一个节点：
   * 选择权重最小的两个节点作为左右子节点。（堆）
   * 创建一个新节点作为它们的父节点，权重为左右子节点的权重之和。
   * 将新节点加入到节点集合中。
4. 生成编码：从根节点开始，遍历Huffman树的路径，为每个字符生成对应的编码。向左子节点移动时，添加一个"0"位；向右子节点移动时，添加一个"1"位。
5. 压缩文本：使用生成的编码，将原始文本中的每个字符替换为对应的编码，生成压缩后的二进制字符串。

```python
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq) # note: 合并之后 char 字典是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))

def main():
    char_freq = {'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 8, 'f': 9, 'g': 11, 'h': 12}
    huffman_tree = huffman_encoding(char_freq)
    external_length = external_path_length(huffman_tree)
    print("The weighted external path length of the Huffman tree is:", external_length)

if __name__ == "__main__":
    main()

# Output:
# The weighted external path length of the Huffman tree is: 169 
```

### AVL

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
    def delete(self, value):
        self.root = self._delete(value, self.root)

    def _delete(self, value, node):
        if not node:
            return node

        if value < node.value:
            node.left = self._delete(value, node.left)
        elif value > node.value:
            node.right = self._delete(value, node.right)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp

            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete(temp.value, node.right)

        if not node:
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        # Rebalance the tree
        if balance > 1:
            if self._get_balance(node.left) >= 0:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if self._get_balance(node.right) <= 0:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```

### 并查集

主要用于解决一些**元素分组**的问题。它管理一系列**不相交的集合**，并支持两种操作：

* **合并**（Union）：把两个不相交的集合合并为一个集合。
* **查询**（Find）：查询两个元素是否在同一个集合中。

```python
#原题：冰阔落1
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

while True:
    try:
        n, m = map(int, input().split())
        parent = list(range(n + 1))

        for _ in range(m):
            a, b = map(int, input().split())
            if find(a) == find(b):
                print('Yes')
            else:
                print('No')
                union(a, b)

        unique_parents = set(find(x) for x in range(1, n + 1))  # 获取不同集合的根节点
        ans = sorted(unique_parents)  # 输出有冰阔落的杯子编号
        print(len(ans))
        print(*ans)

    except EOFError:
        break
```

### 图

```python
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self,f,t,weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())
```

#### 算法

1. BFS
   Dijkstra 算法是一个基于「贪心」、「广度优先搜索」、「动态规划」求一个图中一个点到其他所有点的最短路径的算法，时间复杂度 O(n2)

   ```python
   import heapq

   def dijkstra(graph, start):
       distances = {node: float('infinity') for node in graph}
       distances[start] = 0
       pq = [(0, start)]

       while pq:
           current_distance, current_node = heapq.heappop(pq)

           if current_distance > distances[current_node]:
               continue

           for neighbor, weight in graph[current_node].items():
               distance = current_distance + weight

               if distance < distances[neighbor]:
                   distances[neighbor] = distance
                   heapq.heappush(pq, (distance, neighbor))

       return distances
   ```

   最短路径问题：有权图DIJKSTRA,无权图BFS

   ***Dijkstra算法不能处理负权值的边：需要使用以下算法***

   Floyd-Warshall 算法

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
                   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

       return dist
   ```
2. DFS
3. 拓扑排序

   拓扑排序（Topological Sorting）是对有向无环图（DAG）进行排序的一种算法。它将图中的顶点按照一种线性顺序进行排列，使得对于任意的有向边 (u, v)，顶点 u 在排序中出现在顶点 v 的前面。

   Kahn算法：

   Kahn算法的基本思想是通过不断地移除图中的入度为0的顶点，并将其添加到拓扑排序的结果中，直到图中所有的顶点都被移除。具体步骤如下：

   1. 初始化一个队列，用于存储当前入度为0的顶点。
   2. 遍历图中的所有顶点，计算每个顶点的入度，并将入度为0的顶点加入到队列中。
   3. 不断地从队列中弹出顶点，并将其加入到拓扑排序的结果中。同时，遍历该顶点的邻居，并将其入度减1。如果某个邻居的入度减为0，则将其加入到队列中。
   4. 重复步骤3，直到队列为空。

      ```python
      from collections import deque, defaultdict

      def topological_sort(graph):
          indegree = defaultdict(int)
          result = []
          queue = deque()

          # 计算每个顶点的入度
          for u in graph:
              for v in graph[u]:
                  indegree[v] += 1

          # 将入度为 0 的顶点加入队列
          for u in graph:
              if indegree[u] == 0:
                  queue.append(u)

          # 执行拓扑排序
          while queue:
              u = queue.popleft()
              result.append(u)

              for v in graph[u]:
                  indegree[v] -= 1
                  if indegree[v] == 0:
                      queue.append(v)

          # 检查是否存在环
          if len(result) == len(graph):
              return result
          else:
              return None

      # 示例调用代码
      graph = {
          'A': ['B', 'C'],
          'B': ['C', 'D'],
          'C': ['E'],
          'D': ['F'],
          'E': ['F'],
          'F': []
      }

      sorted_vertices = topological_sort(graph)
      if sorted_vertices:
          print("Topological sort order:", sorted_vertices)
      else:
          print("The graph contains a cycle.")

      # Output:
      # Topological sort order: ['A', 'B', 'C', 'D', 'E', 'F']
      ```
4. 强连通单元（SCC）
5. 最小生成树 MSTs

   有权图Prim 无权图BFS

Prim：

* Prim算法是一种用于解决最小生成树（MST）问题的贪心算法，它会逐步构建一个包含所有顶点的树，并且使得树的边权重之和最小。
* BFS是一种用于无权图的遍历算法，它按照层次遍历的方式访问图的所有节点，并找到从起始顶点到其他所有顶点的最短路径。
* Prim算法通过选择具有最小权重的边来扩展生成树，并且只考虑与当前生成树相邻的顶点。
* BFS通过队列来保存待访问的顶点，并按照顺序进行遍历，不考虑边的权重。

```python
import sys
import heapq

def prim(graph, start):
    pq = []
    start.distance = 0
    heapq.heappush(pq, (0, start))
    visited = set()

    while pq:
        currentDist, currentVert = heapq.heappop(pq)
        if currentVert in visited:
            continue
        visited.add(currentVert)

        for nextVert in currentVert.getConnections():
            weight = currentVert.getWeight(nextVert)
            if nextVert not in visited and weight < nextVert.distance:
                nextVert.distance = weight
                nextVert.pred = currentVert
                heapq.heappush(pq, (weight, nextVert))

```

# 其他

#### 最大公约数 GCD

```python
def gcd(m,n):
    while m%n != 0:
        oldm = m
        oldn = n

        m = oldn
        n = oldm%oldn
    return n

print(gcd(20,10))
```

#### 递归是用栈实现的，递归栈容量即最大递归深度/次数

#### 回溯使用递归实现而不是使用队列保存路径。

回溯算法的特点是在搜索过程中具有跳跃性，即可以根据问题的特点进行剪枝或跳过某些无效的搜索路径，以提高效率。

#### @lru_cache(max_size=None)缓存系统

### 单调栈

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

# 数据结构

**数据结构是相互之间存在一种或多种特定关系的数据元素的集合** 。**“结构”就是指数据元素之间存在的关系，分为逻辑结构和存储结构。**

逻辑结构分为：

* 线性结构：数组、队列、链表、栈
* 非线性结构：多维数组、广义表、树结构和图结构，堆

存储结构分为：

* 顺序存储，在内存中用一组地址连续的存储单元依次存储线性表的各个数据元素
* 链式存储，元素节点存放数据元素以及通过指针指向相邻元素的地址信息
* 索引存储，建立附加的索引表来标识节点的地址
* 散列存储（Hash存储），由节点的关键码值决定节点的存储地址

### 线性表

线性表是一种逻辑结构，描述了元素按线性顺序排列的规则。该序列有**唯一的头元素和尾元素**，除了头元素外，每个元素都有唯一的前驱元素，除了尾元素外，每个元素都有唯一的后继元素

线性表中的元素属于相同的数据类型

顺序表（线性表的顺序存储，也叫数组？）和链表是指存储结构，两者属于不同层面的概念，线性表分为顺序表和链表

表中元素具有逻辑上的顺序性，表中元素有其先后次序

#### 顺序表

顺序表最主要的特点是**随机访问**，即通过首地址和元素序号可在时间O（1）内找到指定的元素。

顺序表的存储密度高，每个结点只存储数据元素。
顺序表逻辑上相邻的元素物理上也相邻，所以插入和删除操作需要移动大量元素。

#### 单链表

线性表的链式存储又称单链表，它是指通过一组任意的存储单元来存储线性表中的数据元素。

为了建立数据元素之间的线性关系，对每个链表结点，除存放元素自身的信息外，还需要存放一个指向其后继的指针。

#### 栈

常见应用：括号匹配，shunting yard

#### 队列

##### 双端队列

##### 环形队列

环形队列也是队列的一种数据结构, 也是在队头出队, 队尾入队，只是环形队列的大小是确定的, 不能进行一个长度的增加; 环形队列在物理上是一个定长的数组

指针走到队尾后会重置到0：rear=(rear+1) %m

#### 散列表

## 树

#### 二叉树

#### 多叉树

#### 完全二叉树--堆

#### 二叉搜索树 BST

二叉搜索性：小于父节点的键都在左子树中，大于父节点的键则都在右子树中

#### 平衡二叉搜索树 AVL

当二叉搜索树不平衡时，get和put等操作的性能可能降到O(n)。AVL能自动维持平衡

实现：平衡因子，左右旋

本质上，左旋包括以下步骤。

1. 将右子节点提升为子树的根节点
2. 将旧根节点作为新根节点的左子节点
3. 如果新根节点（节点B）已经有一个左子节点，将其作为新左子节点（节点A）的右子节点。注意，因为节点B之前是节点A的右子节点，所以此时节点A必然没有右子节点。因此，可以为它添加新的右子节点，而无须过多考虑。

![image-20240322192203776](https://raw.githubusercontent.com/GMyhf/img/main/img/202403221922941.png)

LL：右旋一次

LR：以C为root左旋一次转化为LL型

## 图

### 邻接表

邻接表是图的一种最主要存储结构,用来描述图上的每一个点，包含点的所有邻接顶点

入度出度

（其余定义、算法等见上）

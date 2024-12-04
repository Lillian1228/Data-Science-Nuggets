
### 1. Foundamental Ideas

- All data structures are transformations of arrays (sequential storage) and linked lists (linked storage).

- The key aspects of data structures are traversal and access, which include basic operations like addition, deletion, search, and modification.

- All algorithms are based on exhaustive search.

- The key to exhaustive search is no omissions and no redundancies. Mastering the algorithm framework ensures no omissions; effectively utilizing information ensures no redundancies.

#### 1.1 Storage Methods of Data Structures

There are only two storage methods for data structures: arrays (sequential storage) and linked lists (linked storage).

**Queues and stacks** can be implemented using either linked lists or arrays. Using arrays requires handling resizing issues; using linked lists avoids this but requires more memory space to store node pointers.

The two storage methods for **graph structures** are adjacency lists and adjacency matrices. An adjacency list is essentially a linked list, and an adjacency matrix is a two-dimensional array. 

A **hash table** maps keys into a large array using a hash function. To resolve hash collisions, the 
chaining method employs linked list characteristics, making operations simple but requiring additional space for pointers. The linear probing method utilizes array characteristics for continuous addressing, eliminating the need for pointer storage but making operations slightly more complex.

In tree structures, an array-based implementation is a "heap", as a heap is a complete binary tree. Using arrays for storage eliminates the need for node pointers, and operations are relatively simple, as seen in applications like binary heaps. More common tree structures (incomplete trees) use a linked list-based implementation.

#### 1.2 Basic Operations of Data Structures

For any data structure, its basic operations are simply traversal + access, which can be further detailed as: add, delete, search, and update.

From a high-level perspective, there are only two forms of traversal + access for various data structures: linear and non-linear. Linear traversal is typically represented by for/while iteration, while non-linear traversal is represented by recursion. 

- Array traversal

```Python
def traverse(arr: List[int]):
    for i in range(len(arr)):
        # iterate over arr[i]
```

- Linked list traversal

```Python
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def traverse(head: ListNode):
  # initialize the pointer for loop below
  p = head
  while p is not None:
    # iteratively access p.val
    p = p.next

def traverse(head: ListNode):
  # recursively access head.val
  traverse(head.next)
```

- Binary tree traversal

```Python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def traverse(root: TreeNode):
  traverse(root.left)
  traverse(root.right)
```
- N-ary tree traversal / Graph traversal

```Python
class TreeNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = List[TreeNode]

def traverse(root: TreeNode):
  for child in root.children:
    traverse(child)
```

### Recursion

"Loops may achieve a performance gain for your program. Recursion may achieve a performance gain for your programmer." There's no performance benefit to using recursion. Loops are sometimes better for performance.

Recursion is when a function calls itself. Every recursive function has two cases: the base case and the recursive case. The base case is when the function doesn't call itself  so it doesn't go into an infinite loop.

A stack is a simple data structure with only two actions: push and pop. Push is to add a new item to the top, and pop is to remove the topmost item and read it. 

In the context of calling a function, computers use a stack internally called the **call stack** to save variables for nested function calls. When you call a function from another function, the 2nd function call is placed on top of the call stack and the 1st function is paused in a partially completed state. All the varibles for the 1st function are still stored in memory. You return back to the 1st function after you're done with the 2nd function call (it is poped out of the call stack).

The call stack can get very large and takes up too much memory. At that point, you can rewrite your code to use a loop instead. Or consider advanced techniques like tail recursion.

A resursive example:
```Python
def fact(x):
    if x ==1:
        return 1
    else:
        return x*fact(x-1)

fact(3)
```
![eg](src/call_stack1.png)
![eg](src/call_stack.png)




### 4. Divide and Conquer (D&C)

D&C works by breaking down a problem into smaller and saller pieces. There are two steps: 1) Figure out the base case. 2) Divide or decrease your problem until it becomes the base case. If you are using D&C on a list, the base case is probably an empty array or an array with one element.

#### 4.1 Quicksort: an efficient sorting algorithm based on D&C



#### 4.2 Why O($n*log(n)$) for average case?

1. Each level of recursion requries scanning through all elements to place them in the correct subarray. Therefore partitioning step requires linear time O(n).
2. each function call split the array approximately in half, creating a tree structure of recursive calls with about log(n) levels. In other words, **the recursion depth (the height of the call stack) is O($log(n)$)**.
3. In total, time complexity is operations per level * number of levels.

| Variation |	Time Complexity | Space Complexity |
| -------- | ------- | ------- |
| Best Case |	O(n log n) |	O(log n) |
| Average Case |	O(n log n) |	O(log n) |
| Worst Case |	O(n^2) |	O(n) |

Space complexity is measured by the height of the call stack.

#### 4.3 Revisiting Big 0 notation

In practical applications constants in Big O could matter sometimes, i.e. when input sizes are relatively small or when algorithms in comparison have similar complexity. For instance, both merge sort and quicksort have O(n*log n) time complexity, but quicksort often outperforms due to a smaller constant factor (no need for extra memory allocation in place).



### Union-Find


### 7. Weighted Graph and Dijkstra's algorithm

Breadth-first search is used to calculate the shortest path for an unweighted graph. Dijkstra's algorithm is used to calculate the **shortest path for a weighted graph**.

#### 7.1 What is Dijkstra's algorithm?

Dijkstra's algorithm solves the shortest-path problem for any weighted, directed graph with non-negative weights.

- Four steps:
  1) Find the cheapest node you can get to at the least amount of cost
  2) check whether there's a cheaper path to the neighbors of that node. If so, update their costs (costs for all nodes are initialized as positive infinity).
  3) Repeat until you've done this for every node in the graph.
  4) Calculate the final path.


- Dijkstra's works only when all the weights are non-negative. This is to ensure that once a node has been visited, its optimal distance cannot be improved. This property is especially important to enable Dijkstra's algorithm to act in a greedy manner by always selecting the next most promising node.
  
- If you have negative weights, use the Bellman-Ford algorithm.

- depending on how it is implemented and what data structures are used, the time complexity is typically O(E*log(V)) which is competitive against other shortest path algorithms.

#### 7.2 Implementation





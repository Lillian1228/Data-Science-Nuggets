
### 1. Key Metrics of Sorting Algorithms

- time and space complexity
- sorting stability
    - A sorting algorithm is considered "stable" if identical elements maintain their relative positions after sorting; otherwise, it is "unstable."
    - For instance, consider you have several order records already sorted by transaction date, and you now want to further sort them by user ID. If you use a stable sorting algorithm, the orders with the same user ID will remain sorted by transaction date after sorting; an unstable sorting algorithm will result in a loss of order by transaction date.
- in-place sorting
    - does not require additional auxiliary space, only a constant amount of extra space, and directly sorts the original array.
    - The key here is whether extra space is needed, not whether a new array is returned. 

### 2. Selection Sort

Selection sort is the simplest and most straightforward sorting algorithm, but it has a high time complexity and is not a stable sort. Other basic sorting algorithms are optimizations based on selection sort.

How it works: First, go through the array to find the smallest value and swap it with the first element of the array. Then, iterate through the array again to find the second smallest element and swap it with the second element of the array. Continue this process until the entire array is sorted.


```Python
def sort(nums: List[int]) -> None:
    n = len(nums)
    # sortedIndex is a boundary
    # elements with index < sortedIndex are sorted
    # elements with index >= sortedIndex are unsorted
    # initialized to 0, meaning the whole array is unsorted
    sortedIndex = 0
    while sortedIndex < n:
        # find the minimum in the unsorted part [sortedIndex, n)
        minIndex = sortedIndex
        for i in range(sortedIndex + 1, n):
            if nums[i] < nums[minIndex]:
                minIndex = i
        # swap the minimum value with the element at sortedIndex
        nums[sortedIndex], nums[minIndex] = nums[minIndex], nums[sortedIndex]

        # increment sortedIndex by one
        sortedIndex += 1
```

- Selection sort can be an in-place sort. It does not use additional array space for assistance, only a few variables, making the space complexity O(1).

- Time conplexity: O(n^2). It is a nested for loop. The total number of iterations is (n - 1) + (n - 2) + (n - 3) +... + 1, which is the sum of an arithmetic series. The result is approximately n^2 / 2.

- The time complexity of selection sort is independent of the initial order of the data. Even if the input is an already sorted array, the time complexity remains O(n^2).

- It is not a stable sorting, because it swaps the smallest element with the current element each time, which can change the relative order of identical elements.


### 3. Bubble Sort

The bubble sort algorithm is an optimization of Selection Sort. Instead of simply swapping nums[sortedIndex] with nums[minIndex] for convenience, it sorts by iteratively swapping the adjacent inverted pairs, w/o touching telements with the same value, making it a stable sorting algorithm.

```Python
def bubble_sort(nums: List[int]) -> None:
    n = len(nums)
    sortedIndex = 0

    # until sorted_index reaches the end of the array
    while sortedIndex < n:
        # comparing adjacent elements and swapping them if needed, effectively "bubbling" the smallest value toward the front.
        for i in range(n-1, sortedIndex, -1):
            if nums[i] < nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]

        # increment sortedIndex by one
        sortedIndex += 1
```

- The time complexity of this algorithm remains O(n^2). The actual number of operations is similar to selection sort, forming an arithmetic series sum, approximately n^2 / 2 times.

- A further enhancement can be done to terminate the sort early when the array has been sorted:

```Python
def bubble_sort(nums: List[int]) -> None:
    n = len(nums)
    sortedIndex = 0

    # until sorted_index reaches the end of the array
    while sortedIndex < n:
        # record whether a swap operation has been performed.
        swapped = False
        # comparing adjacent elements and swapping them if needed, effectively "bubbling" the smallest value toward the front.
        for i in range(n-1, sortedIndex, -1):
            if nums[i] < nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]
                swapped = True
        
        # if no swap operation is performed, it indicates that the array is already sorted, and the algorithm can terminate early
        if swapped=True:
            break
        # increment sortedIndex by one
        sortedIndex += 1
```

### 4. Insersion Sort

Insertion Sort is an optimization based on Selection Sort, where nums[sortedIndex] is inserted into the sorted portion on the left. It works similarly to how you might sort a deck of playing cards in your hands. The algorithm divides the array into a "sorted" and an "unsorted" part. It iterates over the unsorted part and inserts each element into its correct position in the sorted part. 

For arrays with a high degree of order, Insertion Sort is quite efficient.

```Python
# further optimize selection sort by inserting elements into the left sorted array this algorithm has another name, called insertion sort
def insertion_sort(nums: List[int]) -> None:
    n = len(nums)
    sortedIndex = 1

    # until sorted_index reaches the end of the array
    while sortedIndex < n:
        # Insert nums[sorted_index] into the sorted array [0, sorted_index)
        for i in range(sortedIndex, 0, -1):
            if nums[i] < nums[i-1]:
                nums[i], nums[i-1] = nums[i-1], nums[i]
            # ensures that the loop stops early if no more swaps are needed, optimizing the insertion process.
            else:
                break
        # increment sortedIndex by one
        sortedIndex += 1
```

- Insertion sort has a space complexity of O(1), making it an in-place sorting algorithm. Its time complexity is O(n^2), similar to selection sort, as it involves an arithmetic series summation, approximately n^2 / 2 operations.

- Insertion sort is a stable sorting algorithm because elements are only swapped when nums[i] < nums[i - 1], so the relative order of identical elements remains unchanged.

- The efficiency of insertion sort is closely related to the initial order of the input array. If the input array is already sorted or only a few elements are out of order, the inner for loop of insertion sort hardly needs to perform any element swaps, so the time complexity approaches O(n).

- Comparing insertion sort with bubble sort, the overall performance of insertion sort should be better than bubble sort.

### 5. Shell Sort

Insertion sort is slow for large unordered arrays because the only exchanges it does involve adjacent entries, so items can move through the array only one place at a time. For example, if the item with the smallest key happens to be at the end of the array, N 1 exchanges are needed to get that one item where it belongs. 

Shellsort is a simple extension of insertion sort that gains speed by allowing exchanges of array entries that are far apart, to produce partially sorted arrays that can be efficiently sorted, eventually by insertion sort, breaking through the O(n^2) time complexity of insertion sort.

- **h-sorted array**: taking every hth entry (starting anywhere) yields a sorted subsequence. By h-sorting for some large values of h, we can move items in the array long distances and thus make it easier to h-sort for smaller values of h. For example, for h=3:

```
nums:
[1, 2, 4, 3, 5, 7, 8, 6, 10, 9, 12, 11]
 ^--------^--------^---------^
    ^--------^--------^---------^
       ^--------^--------^----------^

 1--------3--------8---------9
    2--------5--------6---------12
        4--------7--------10---------11
```

- increment sequence (gap sequence): defines the distance between the elements to be compared and sorted during each pass. It starts with a large gap and gradually decreases to 1, enabling the algorithm to efficiently sort elements far apart before focusing on closer ones. The choice of the increment sequence greatly influences the performance.
    - original Shell's sequence: $2^{k-1}$. for n=16, the sequence is [8,4,2,1]. simple but inefficient.
    - Knuth's sequence: $(3^k-1)/2$, starting at the largest increment less than N/3 and decreasing to 1. i.e. for n = 100, the sequence is [1,4,13]. Better performance compared to the original.

```Python
def shell_sort(nums):
    n = len(nums)
    # Generate gap sequence (3^k - 1) / 2, i.e., h = 1, 4, 13, 40, 121, 364...
    h = 1

    # The initial gap h is calculated using the increment sequence (3^k - 1) / 2.
    while h < n / 3:
        h = 3 * h + 1

    # Perform h-sorting of the array
    while h >= 1:
        # until iteration reaches the end
        for i in range(h, n):
            # initialize the inner loop var j from i. Insert arr[i] among arr[i-h], arr[i-2*h], ...
            j = i
            # performs the comparison and swaps elements that are h distance apart while traversing backward.
            while j>=h and nums[j] < nums[j-h]:
                nums[j], nums[j-h] = nums[j-h], nums[j]
                j-=h

        # Reduce h according to the sequence
        h //= 3 # h=h//3

```

- Shellsort is unstable sorting. When h>1 swapping elements can change the relative position of the same elements.
- time complexity is less than O(N^2), but the actual complexity depends on the increment sequence.

### 6. Quick Sort

Quicksort is a divide-and-conquer method for sorting. It works by partitioning an array into two subarrays, then sorting the subarrays independently.

- Approach: Partitioning around a pivot. 
  - Quicksort selects a "pivot" element and partitions the array into 2 subarrays - one with elements less than the pivot and one with elements greater than the pivot. Notice that after sorting around pivot, the pivot position will be final.
  - It recursively applies this partitioning process to each subarray.
- Time complexity: O($n*log(n)$) on average case, but can become O($n^2$) in the worst-case.
- Pivot selection (partitioning) matters. Choosing a random pivot reduces the likelihood of unbalanced splits, and helps quicksort achieve the average runtime O(n*log(n)). 

```Python
# Pseudo code

def quicksort(nums:List[int], lo: int, hi: int):
    # base case: when the sort list has <=1 element
    if lo>=hi:
        return
    
    # traverse through nums[lo,hi] to place the pivot at the correct position and return index p.
    p = partition(nums, lo, hi)

    quicksort(nums, lo, p-1)
    quicksort(nums, p+1, hi)

# 二叉树的遍历框架
def traverse(root):
    if root is None:
        return
    # 前序位置
    traverse(root.left)
    traverse(root.right)
```
This code framework is in essence the **pre-order traversal framework** in the basics of binary tree traversal: sorting `nums[p]` at the pre-order position and then recursively sorting the left and right elements.
- The `sort` function in the Quick Sort code framework is equivalent to the `traverse` function in the binary tree traversal framework. It is used to traverse all the nodes (array elements), and for each node (array element), it calls the `partition` function to place that node (array element) in the correct position.

- After the traversal is completed, all the array elements are placed in the correct positions, and the entire array is sorted.

- As for the idea behind the `partition` function, it will be easier to understand after you have studied the **Linked List Two-pointer Technique Summary** and the **Array Two-pointer Technique Summary**.

#### 6.2 Time and space complexity analysis

For example, for each pivot (binary tree node) we need to traverse through the entire list O(n). The ideal number of partitions (depth of the tree) is O(log(n)). Therefore the overall time complextiy is o(n*log(n))
```
        [4, 1, 7, 2, 5, 3, 6]
         /                 \
    [2, 1, 3]    [4]     [7, 5, 6]
     /     \              /     \
  [1]  [2]  [3]        [5]  [6]  [7]
```

Quick Sort does not require additional auxiliary space, as it is an in-place sorting algorithm. When recursively traversing a binary tree, the recursion function's stack depth is equal to the height of the tree, so the space complexity is O(log n).

Quick Sort is an unstable sorting algorithm because in the `partition` function, the relative positions of identical elements are not considered, so the relative positions of identical elements may change.

### 7. Merge Sort

The approach: to sort an array, divide it into two halves, sort the two halves (recursively), and
then merge the results. As you will see, one of mergesort’s most attractive properties is
that it guarantees to sort any array of N items in time proportional to N log N. Its prime
disadvantage is that it uses extra space proportional to N.

```Python
# Pseudo code

def mergesort(nums:List[int], lo: int, hi: int):
    # base case: when the sort list has <=1 element
    if lo==hi:
        return
    mid = (lo+hi)//2

    # sort two unordered arrays
    mergesort(nums, lo, mid)
    mergesort(nums, mid+1, hi)

    # merge the two ordered subarrays
    merge(nums, lo, mid, hi)

# 二叉树的遍历框架
def traverse(root):
    if root is None:
        return
    
    traverse(root.left)
    traverse(root.right)

    # 后序位置
```
- whether the mergesort is stable or not depends on the implementation of merge(), which requires two-pointer techniques that will be discussed separately.
- time complexity is O(n*log n): merge function will traverse the entire array, and the ideal number of merges is log n.
- not an in-place sort as it requires extra array spaces to merge two ordered arrays. Space complexity is O(n).

### 8. Heap Sort based on binary heap

The core concept of the heap sort involves constructing a heap from our input (heapify) and repeatedly removing the minimum/maximum element to sort the array. A naive approach to heapsort would start with creating a new array and adding elements one by one into the new array. As with previous sorting algorithms, this sorting algorithm can also be performed in place, so no extra memory is used in terms of space complexity.

The key idea for in-place heapsort involves a balance of two central ideas:
    (a) Building a heap from an unsorted array through a “bottom-up heapification” process, and
    (b) Using the heap to sort the input array.

Heapsort traditionally uses a max heap to accomplish sorting `nums` from small to large, because the elements deleted from the heap top are filled into the `nums` array from back to front, ultimately resulting in the elements in `nums` being sorted from small to large.

If you use a min heap instead, the elements in `nums` would be sorted from large to small, and you would need to additionally reverse the array, which is not as efficient as using a max heap.

```Python
def swap(heap, i, j):
    # 交换数组中两个元素的位置
    heap[i], heap[j] = heap[j], heap[i]

def max_heap_swim(heap, node):
    # 大顶堆的上浮操作
    while node > 0 and heap[parent(node)] < heap[node]:
        swap(heap, parent(node), node)
        node = parent(node)

def max_heap_sink(heap, node, size):
    # 大顶堆的下沉操作
    while left(node) < size or right(node) < size:
        # 小顶堆和大顶堆的唯一区别就在这里，比较逻辑相反
        # 比较自己和左右子节点，看看谁最大
        max_index = node
        if left(node) < size and heap[left(node)] > heap[max_index]:
            max_index = left(node)
        if right(node) < size and heap[right(node)] > heap[max_index]:
            max_index = right(node)
        if max_index == node:
            break
        swap(heap, node, max_index)
        node = max_index

def sort(nums):
    # 第一步，原地建堆，注意这里创建的是大顶堆
    # 从最后一个非叶子节点开始，依次下沉，合并二叉堆
    n = len(nums)
    # by processing the nodes from the bottom-up, once we are at a specific node in our heap, it is guaranteed that all child nodes are also heaps.
    for i in range(n // 2 - 1, -1, -1):
        max_heap_sink(nums, i, n)

    # 第二步，排序
    # 现在整个数组已经是一个大顶了，直接模拟删除堆顶元素的过程即可
    heap_size = len(nums)
    while heap_size > 0:
        # 从堆顶删除元素，放到堆的后面
        swap(nums, 0, heap_size - 1)
        # omit the sorted element from the heap while keeping it in the array
        heap_size -= 1
        # 恢复堆的性质
        max_heap_sink(nums, 0, heap_size)
        # 现在 nums[0..heap_size) 是一个大顶堆，nums[heap_size..) 是有序元素
```

Why bottom-up + sink instead of top-down swim?

Each leaf node already satisfies the heap property, so the above code starts from the last non-leaf node (index at n // 2 - 1) and sequentially calls the `maxHeapSink` method, merging all sub-heaps, ultimately transforming the entire array into a max heap. This is more efficient during heapification because it only needs to call the sink method on half of the elements, with the total number of operations being approximately 1/2 * N * log(N).

Although the Big O notation remains O(N * log(N)), the actual number of operations performed will definitely be fewer compared to calling the swim method on every element.

- Time Complexity: O(N log N). Space Complexity: O(1)
- It's not a stable sort. 


### 9. Counting Sort

The idea of counting sort is simple: count how many times each element appears, then figure out the index of each element in the sorted array, and finally finish sorting.

It is a non-comparison-based sorting algorithm that operates by **counting the occurrences of each distinct element** in the input array. It is particularly **efficient for sorting collections of objects with small positive integer keys** or when the range of key values is not significantly larger than the number of items.

The time and space complexity of counting sort is **$O(n+max−min)$**, where 
n is the length of the array, and max−min is the range of the numbers in the array.

```Python
from collections import Counter
def CountingSort(self, nums: List[int]) -> None:
        # count the frequency of distinct numbers
        count = Counter(nums)
        # fill the original array according to the count array
        index = 0
        for element in range(len(count)): # note that we assumed the distinct numbers range from 0 to n.
            for _ in range(count[element]):
                nums[index] = element
                index += 1
```

### 10. Bucket Sort

### 11. Radix Sort
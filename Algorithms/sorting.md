
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

Shell Sort is a simple improvement of Insertion Sort that increases the local orderliness of an array through preprocessing, breaking through the O(n^2) time complexity of insertion sort.


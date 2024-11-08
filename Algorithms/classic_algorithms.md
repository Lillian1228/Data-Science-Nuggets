### 3. Recursion

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

- Approach: Partitioning around a pivot. 
  - Quicksort selects a "pivot" element and partitions the array into 2 subarrays - one with elements less than the pivot and one with elements greater than the pivot. 
  - It recursively applies this partitioning process to each subarray.
- Time complexity: O($n*log(n)$) on average case, but can become O($n^2$) in the worst-case.
- Pivot selection matters. Choosing a random pivot reduces the likelihood of unbalanced splits, and helps quicksort achieve the average runtime O(n*log(n)). 

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

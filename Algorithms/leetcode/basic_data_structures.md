## Circular Array

A circular array (also called a ring buffer) is a data structure that **uses fixed sized array** as if it were connected end-to-end in a circle. When you reach the end of the array, you "wrap around" to the beginning instead of stopping. 

- Core idea: 
  
  This is achieved by using the modulo operator (%) for index calculations. For an array of size N, the index of the next element after i is ```i = (i + 1) % N```, which handles the wrap-around automatically.
  
- Common Use

    1. Implementing a queue (especially circular queue): when removing elements from the front and adding to the rear, you you maintain and move 2 indices instead of shiting elements: 

        ```Python
        # front points to where the next element will be removed
        front = (front + 1) % N # deletion
        
        # rear points to where the next element will be inserted
        rear - (rear + 1) % N # insertion at the end or wrap around to the beginning if array is full
        ```
    2. Streaming or Real-time Data Buffers: 

        When data, such as audio or video, is streamed, a circular buffer is used to store incoming data temporarily.

        Once the buffer is full, new data overwrites the oldest data, ensuring the buffer always contains the most recent information.
    
    3. Scheduling Problems: useful when tasks repeat in cycles, i.e. a round-robin scheduler.
   
- Advantages
    - **Constant time O(1) complexity for insertions and deletions**
    - Efficient use of space - no need to shift elements

- Implementation

    1. Circular traversal with the modulo operator

    ```Python  
    class CircularArray:
        def __init__(self, size):
            self.size = size
            self.array = [None] * size
        def set(self, index, value):
            self.array[index % self.size] = value
        def get(self, index):
            return self.array[index % self.size]

    # Example usage
    carr = CircularArray(5)
    for i in range(7):
        carr.set(i, chr(65 + i))  # set A, B, C, D, E, F, G
    print(carr)  # ['F', 'G', 'C', 'D', 'E'] — wraps around!
    ```

    2. ```collections.deque``` can be treated as a circular array
    ```Python
    from collections import deque

    # A deque can be treated as a circular array
    circular_queue = deque(maxlen=5)

    # Enqueue (add elements to the right)
    circular_queue.append(1)
    circular_queue.append(2)
    circular_queue.append(3)

    print("After initial appends:", circular_queue)
    # Output: After initial appends: deque([1, 2, 3], maxlen=5)

    # Enqueue more elements; '1' will be overwritten
    circular_queue.append(4)
    circular_queue.append(5)
    circular_queue.append(6)

    print("After overwriting oldest element:", circular_queue)
    # Output: After overwriting oldest element: deque([2, 3, 4, 5, 6], maxlen=5)

    # Dequeue (remove element from the left)
    removed_element = circular_queue.popleft()

    print("Removed element:", removed_element)
    # Output: Removed element: 2

    print("After popleft:", circular_queue)
    # Output: After popleft: deque([3, 4, 5, 6], maxlen=5)
    ```


## Stack & Queue

**Stack (LIFO)** and **Queue (FIFO)** are two fundamental and important data structures, featuring "restricted operations" compared to basic arrays and linked lists. They are not only widely used on their own but can also be extended to advanced data structures like monotonic stacks and monotonic queues to efficiently solve complex problems.

A queue allows elements to be **added at the back and removed from the front**, while a stack allows elements to be **added and removed from the top**. 

In Python, they can be easily implemented using the doubly linked list ```from collections import deque``` or arrays.

[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/description/)

[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/description/)



## Hash table

The hash table is a core data structure for fast key-value access. Mastering its basic principles and understanding how it combines with arrays, linked lists, etc., to form more powerful data structures like LinkedHashMap and ArrayHashMap is crucial for solving algorithm problems.

[128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)


## OOD

Data structure design problems test the comprehensive understanding and application of data structures, requiring the ability to combine basic data structures to design efficient custom data structures for specific needs, such as LRU cache, LFU cache, and calculators.
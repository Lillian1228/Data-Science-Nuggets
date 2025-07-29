
### Bit Manipulation

- ```int(a, 2)``` converts binary strings to an integer in base 2
- ```bin(a)``` converts an integer to binary type prefixed with "0b"
- Binary Operators:

| Operator | Symbol | Effect | Example |
|------|---------|-------|---|
|XOR |```a ^ b``` | 1 if bits are different, else 0, used to sum without taking carry into account | "11"^"1" &rarr; "10"|
|AND|```a & b```| 1 if both bits are 1, used to find the current carry | "11" & "1" &rarr; "1"|
| Left Shift | ```a << 1```  | Shift bits left (by 1), used to multiply by 2 | "10" << 1 &rarr; "100" |
| Right Shift | ```a >> 1```  | Shift bits right (by 1), used to divide by 2 | "10" << 1 &rarr; "1" |
 
### Math Operator

- Exponentiation operator: ```x**2``` (square), ```x**0.5``` (square root)
- Division operators 
  - Truncation to zero: discard the fractional part and move the number closer to zero. ```int(7 / 3)``` is 2; ```int(-7 / 3)``` is -2
  - Floor division: round the result down the nearest whole number. ```7 // 3``` is 2; ```-7 // 3``` is -3.

### Built-in Functions

-  Quick counter to get the frequency of elements in a Python list: 
  
      ```Python
      from collections import Counter
      frequency = Counter(my_list)
      ```
- In Python loops, continue and pass statements serve different purposes:
  - continue statement skips the rest of the current iteration
  - pass is a placeholder that does nothing
- ```all()``` function takes an iterable (e.g., a list, tuple, set, dictionary) as its argument and returns True if all elements in the iterable are true. i.e. all(x==1 for x in list)
- using the built-in memo decorator for DP problem:
  - from functools import cache, lru_cache
  - @cache or @lru_cache(None)
- Convert a single-character string to the index position in the Unicode character set: ```ord('a')``` &rarr; 97; Inversely, convert a Unicode index to the corresponding character: ```chr(65)``` &rarr; 'A'

### Standard Libraries

- Array binary search: locate the insertion index for x in array a to maintain sorted order.
  - bisect_left: if x is already present in a, the index will be before any existing entries.
  - bisect_right: if x is already present in a, the index will be after (to the right of) any existing entries of x in a.

  ```Python
  from bisect import bisect_right, bisect_left
  idx = bisect_left(a, x, lo=0, hi=len(a)) # index of the leftmost element that's <x
  idx = bisect_right(a, x, lo=0, hi=len(a)) # index of the rightmost element that's <=x +1
  ```
- Sorted Containers
  - ```SortedList()```:  sorted mutable sequence
  ```Python
  from sortedcontainers import SortedDict, SortedList

  l = SortedList()
  l.add(x)
  l.update(iterable) # Update sorted list by adding all values from iterable
  l.remove(x) # Remove value from sorted list
  l.pop(index=-1) # Remove and return value at index in sorted list
  l.bisect_right(x) # look up values
  ```
  - ```SortedDict()```: A sorted map keeps the stored key-value pairs sorted based on the key.
  ```Python
  d = SortedDict()
  d[key]=value # add
  value = d.pop(key) # Remove and return value for item identified by key.
  kv_pair = d.popitem(index=-1) # Remove and return (key, value) pair at index from sorted dict.
  value = d.get(key, default=None) # Return the value for key if key is in the dictionary, else default.
  kv_pair = d.peekitem(index=-1) # Return (key, value) pair at index in sorted dict
  d.bisect_right(k) # index of the rightmost key that's <=k +1
  ```

### Data Strucures

- Double-ended queue (essentially doubly linked list) in Python: 

  ```from collections import deque``` 
  - Compared to a standard queue (FIFO - First In First Out queue), it offers additional operations: insertion and removal of elements at both the head and the tail with O(1) time complexity: ```popleft()```, ```appendleft()```
  - Used to implement stack: ```pop()``` (remove from top), ```append()```(append to top)
  - Used to implement level order traversal

- Min Heap Implemetation:  ```import heapq``` 
  - heaps can store not only values, but also **tuples**, i.e. (value to be sorted on, object or index associated with the value). heapq will primarily sort based on the first element of the tuple, and then use the second element (the index) as a tie-breaker if values are equal.
  - ```heapq.nlargest(n, iterable, key=None)```: Returns a list with the n largest elements from the iterable.
  - ```heapq.nsmallest(n, iterable, key=None)```: Returns a list with the n smallest elements from the iterable.
  - ```heapq.heapify(iterable)```: heapify a list of values or tuples
  - ```heapq.heappop(iterable)```: pop the smallest element (minheap)
- Python List: 
  - An empty list evaluates to False in a boolean context:  ```if not my_list``` means my_list is not empty
  - ```list.sort()``` will sort the list ascendingly in place and return None. ```list.sort(reverse=True)``` will do descendingly. 
  - ```list.sort(key=lambda x: x[0])``` sort by the first element in a list of lists.
  - ```sorted(list)``` returns a new sorted list without modifying the original list. 
  - Remove elements by value (return None): ```list.remove(value)```
  - Remove elements by index (return the element): ```list.pop(index)```
- Tuple ```t = ("apple",)```: a number of values separated by commas. 
  - Tuples are immutable. Once being created, its values cannot be updated, appended, sorted or removed. However, you can concatenate tuples into a new one.
  - Heterogeneous in nature: elements can be multiple data types. Great for storing immutable key-value pairs. Iterating through a 'tuple' is faster than in a 'list'. 
  - We generally use 'tuples' for heterogeneous (different) data types and 'lists' for homogeneous (similar) data types.
  - Unpack values in a list of tuples: ```for k, v in list_of_tuples```

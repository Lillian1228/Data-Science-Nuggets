
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

- Creating a copy of a list
  - Shallow Copy: ```list.copy()``` or ```list[:]```creates a new list, but the elements within the new list are still references to the original elements.
  - Deep Copy: creates a new list and recursively copies all nested objects within it. This ensures that the copied list is completely independent of the original list, and changes to nested objects in the copied list will not affect the original. 

    ```Python
      import copy
      original_list = [1, 2, [3, 4]]
      copied_list = copy.deepcopy(original_list)
      ```
  - To update the elements in a list without affecting the original elements, we can also shallow copy each of the elements directly. 
-  Quick counter to get the frequency of elements in a Python list: 
  
      ```Python
      from collections import Counter
      frequency = Counter(my_list)
      ```
- In Python loops, continue and pass statements serve different purposes:
  - ```continue``` statement skips the rest of the current iteration
  - ```pass``` is a placeholder that does nothing

- ```all()``` function takes an iterable (e.g., a list, tuple, set, dictionary) as its argument and returns True if all elements in the iterable are true. i.e. all(x==1 for x in list)
- Built-in memo decorator for DP problem or recursive function memorization:
  - ```from functools import cache, lru_cache```
  - Add the decorator right above the recursive function: ```@cache``` for python 3.9+ or ```@lru_cache(maxsize=None)``` for earlier version

- Convert a single-character string to the index position in the Unicode character set: ```ord('a')``` &rarr; 97; Inversely, convert a Unicode index to the corresponding character: ```chr(65)``` &rarr; 'A'

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

### Common Error Checking

1. Index out of range

    - In while loop, make sure to prevent the pointer from moving out of range 
    
    ```Python
    while i<n and iterable[i] is True: 
      i+=1
    ```
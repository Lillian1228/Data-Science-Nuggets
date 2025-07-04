
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
 


### Built-in Functions

- In Python loops, continue and pass statements serve different purposes:
  - continue statement skips the rest of the current iteration
  - pass is a placeholder that does nothing
- all() function takes an iterable (e.g., a list, tuple, set, dictionary) as its argument and returns True if all elements in the iterable are true. i.e. all(x==1 for x in list)
- using the built-in memo decorator for DP problem:
  - from functools import cache, lru_cache
  - @cache or @lru_cache(None)
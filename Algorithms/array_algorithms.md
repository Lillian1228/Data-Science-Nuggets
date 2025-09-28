### Roadmap
![eg](src/1.png)

### Prefix Sum

Computational method used to speed up queries about the sum of elements within any range of an array. The core idea is a space-for-time tradeoff: by spending linear time to pre-process an array, you can answer subsequent range sum queries in constant time.

#### Key Considersations

1. Initialize the prefix sum array with length n+1 and pad 0 at index 0
2. Develop the transition equation (simplified form of DP)

- ID Array

```Python
class NumArray:
    # 前缀和数组
    def __init__(self, nums: List[int]):
        # 输入一个数组，构造前缀和
        # preSum[0] = 0，便于计算累加和
        self.preSum = [0] * (len(nums) + 1)
        # 计算 nums 的累加和
        for i in range(1, len(self.preSum)):
            self.preSum[i] = self.preSum[i - 1] + nums[i - 1]

    # 查询闭区间 [left, right] 的累加和
    def sumRange(self, left: int, right: int) -> int:
        return self.preSum[right + 1] - self.preSum[left]

```

- 2D Array

```Python
class NumMatrix:
    # preSum[i][j] 记录矩阵 [0, 0, i-1, j-1] 的元素和
    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        if m == 0 or n == 0:
            return
        # 构造前缀和矩阵
        self.preSum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # 计算每个矩阵 [0, 0, i, j] 的元素和
                self.preSum[i][j] = (self.preSum[i - 1][j] + self.preSum[i][j - 1] +
                                     matrix[i - 1][j - 1] - self.preSum[i - 1][j - 1])

    # 计算子矩阵 [x1, y1, x2, y2] 的元素和
    def sumRegion(self, x1: int, y1: int, x2: int, y2: int) -> int:
        # 目标矩阵之和由四个相邻矩阵运算获得
        return (self.preSum[x2 + 1][y2 + 1] - self.preSum[x1][y2 + 1] -
                self.preSum[x2 + 1][y1] + self.preSum[x1][y1])

```

#### Limitations of Prefix Sum

1. The prefix sum technique assumes the original array nums is **immutable**.

    If an element in the original array changes, the values in the preSum array that come after it become invalid. You would need to re-calculate the preSum array, which takes \(O(n)\) time and offers no significant advantage over a standard brute-force solution.

2. The prefix sum technique **only applies to scenarios with an inverse operation**. i.e. from x+6=10 we know x=10-6; or from x*6=12 then x=12/6. In some scenarios i.e. max(x,8)=8 we cannot derive x.

To solve both of these problems, you need a more advanced data structure. The most common solution is a **segment tree**.

Leetcode Problems:

[238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)

[304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/description/)

[1352. Product of the Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/description/)
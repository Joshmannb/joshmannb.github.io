---
layout: post
title:  "[LeetCode] Biweekly Contest 111"
date:   2023-08-20 13:16:49 +0800
categories: LeetCode Contest
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

{% include toc %}

---

## Count Pairs Whose Sum is Less than Target (2824)

#### 难度

- **Easy**

#### 问题描述

Given a **0-indexed** integer array `nums` of length `n` and an integer `target`, return _the number of pairs_ `(i, j)` _where_ `0 <= i < j < n` _and_ `nums[i] + nums[j] < target`.

**Example 1:**

**Input:** nums = [-1,1,2,3,1], target = 2  
**Output:** 3  
**Explanation:** There are 3 pairs of indices that satisfy the conditions in the statement:  
- (0, 1) since 0 < 1 and nums[0] + nums[1] = 0 < target  
- (0, 2) since 0 < 2 and nums[0] + nums[2] = 1 < target 
- (0, 4) since 0 < 4 and nums[0] + nums[4] = 0 < target
Note that (0, 3) is not counted since nums[0] + nums[3] is not strictly less than the target.

**Example 2:**

**Input:** nums = [-6,2,5,-2,-7,-1,3], target = -2  
**Output:** 10  
**Explanation:** There are 10 pairs of indices that satisfy the conditions in the statement:  
- (0, 1) since 0 < 1 and nums[0] + nums[1] = -4 < target
- (0, 3) since 0 < 3 and nums[0] + nums[3] = -8 < target
- (0, 4) since 0 < 4 and nums[0] + nums[4] = -13 < target
- (0, 5) since 0 < 5 and nums[0] + nums[5] = -7 < target
- (0, 6) since 0 < 6 and nums[0] + nums[6] = -3 < target
- (1, 4) since 1 < 4 and nums[1] + nums[4] = -5 < target
- (3, 4) since 3 < 4 and nums[3] + nums[4] = -9 < target
- (3, 5) since 3 < 5 and nums[3] + nums[5] = -3 < target
- (4, 5) since 4 < 5 and nums[4] + nums[5] = -8 < target
- (4, 6) since 4 < 6 and nums[4] + nums[6] = -4 < target

**Constraints:**

- `1 <= nums.length == n <= 50`
- `-50 <= nums[i], target <= 50`

#### 解题思路

- 双循环暴力解。

#### 复杂度

- 时间复杂度：$$O(n^2)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
from typing import List

class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        n = len(nums)
        res = 0
        
        for i in range(n):
            for j in range(i):
                if nums[i] + nums[j] < target:
                    res += 1
        
        return res
```

---

## Make String a Subsequence Using Cyclic Increments (2825)

#### 难度

- **Medium**

#### 问题描述

You are given two **0-indexed** strings `str1` and `str2`.

In an operation, you select a **set** of indices in `str1`, and for each index `i` in the set, increment `str1[i]` to the next character **cyclically**. That is `'a'` becomes `'b'`, `'b'` becomes `'c'`, and so on, and `'z'` becomes `'a'`.

Return `true` _if it is possible to make_ `str2` _a subsequence of_ `str1` _by performing the operation **at most once**_, _and_ `false` _otherwise_.

**Note:** A subsequence of a string is a new string that is formed from the original string by deleting some (possibly none) of the characters without disturbing the relative positions of the remaining characters.

**Example 1:**

**Input:** str1 = "abc", str2 = "ad"  
**Output:** true  
**Explanation:** Select index 2 in str1.  
Increment str1[2] to become 'd'.   
Hence, str1 becomes "abd" and str2 is now a subsequence. Therefore, true is returned.  

**Example 2:**

**Input:** str1 = "zc", str2 = "ad"  
**Output:** true  
**Explanation:** Select indices 0 and 1 in str1.   
Increment str1[0] to become 'a'.   
Increment str1[1] to become 'd'.   
Hence, str1 becomes "ad" and str2 is now a subsequence. Therefore, true is returned.

**Example 3:**

**Input:** str1 = "ab", str2 = "d"  
**Output:** false  
**Explanation:** In this example, it can be shown that it is impossible to make str2 a subsequence of str1 using the operation at most once.   
Therefore, false is returned.  

**Constraints:**

- `1 <= str1.length <= 105`
- `1 <= str2.length <= 105`
- `str1` and `str2` consist of only lowercase English letters.

#### 解题思路

- **双指针**  
遍历`str1`，若当前元素在经过转换或没有经过转换等于`str2`指针所指的元素，则右指针+1，若右指针可以指完整个`str2`，则返回`true`。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        
        def transform(letter):
            return chr((ord(letter) - ord('a') + 1) % 26 + ord('a'))
        
        right = 0
        n = len(str2)
        
        for letter in str1:
            if letter == str2[right] or transform(letter) == str2[right]:
                right += 1
            if right == n:
                break
        
        return right == n
```

---

## Sorting Three Groups (2826)

#### 难度

- **Medium**

#### 问题描述

You are given a **0-indexed** integer array `nums` of length `n`.  
  
The numbers from `0` to `n - 1` are divided into three groups numbered from `1` to `3`, where number `i` belongs to group `nums[i]`. Notice that some groups may be **empty**.  
  
You are allowed to perform this operation any number of times:

- Pick number `x` and change its group. More formally, change `nums[x]` to any number from `1` to `3`.

A new array `res` is constructed using the following procedure:

1. Sort the numbers in each group independently.
2. Append the elements of groups `1`, `2`, and `3` to `res` **in this order**.

Array `nums` is called a **beautiful array** if the constructed array `res` is sorted in **non-decreasing** order.

Return _the **minimum** number of operations to make_ `nums` _a **beautiful array**_.

**Example 1:**

**Input:** nums = [2,1,3,2,1]  
**Output:** 3  
**Explanation:** It's optimal to perform three operations:  
1. change nums[0] to 1.  
2. change nums[2] to 1.  
3. change nums[3] to 1.  
After performing the operations and sorting the numbers in each group, group 1 becomes equal to [0,1,2,3,4] and group 2 and group 3 become empty. Hence, res is equal to [0,1,2,3,4] which is sorted in non-decreasing order.  
It can be proven that there is no valid sequence of less than three operations.  

**Example 2:**

**Input:** nums = [1,3,2,1,3,3]  
**Output:** 2  
**Explanation:** It's optimal to perform two operations:  
1. change nums[1] to 1.  
2. change nums[2] to 1.  
After performing the operations and sorting the numbers in each group, group 1 becomes equal to [0,1,2,3], group 2 becomes empty, and group 3 becomes equal to [4,5]. Hence, res is equal to [0,1,2,3,4,5] which is sorted in non-decreasing order.  
It can be proven that there is no valid sequence of less than two operations.  

**Example 3:**

**Input:** nums = [2,2,2,2,3,3]  
**Output:** 0  
**Explanation:** It's optimal to not perform operations.  
After sorting the numbers in each group, group 1 becomes empty, group 2 becomes equal to [0,1,2,3] and group 3 becomes equal to [4,5]. Hence, res is equal to [0,1,2,3,4,5] which is sorted in non-decreasing order.  

**Constraints:**

- `1 <= nums.length <= 100`
- `1 <= nums[i] <= 3`

#### 解题思路

- **线性规划**  
DP的状态为`[idx, boundry]`，代表当看到`idx`元素时若采取下界为`boundry`的操作数。

#### 复杂度

- 时间复杂度：$$O(3n) = O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
from typing import List
from functools import cache

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        
        @cache
        def dfs(idx, boundry) -> int:
            if idx == n:
                return 0
            res = 10**10
            if nums[idx] < boundry:
                res = min(res, dfs(idx + 1, boundry) + 1)
            else:
                res = min(res, dfs(idx + 1, nums[idx]))
                res = min(res, dfs(idx + 1, boundry) + 1)
            return res

        return dfs(0, 1)
```

---

## Number of Beautiful Integers in the Range (2827)

#### 难度

- **Hard**

#### 问题描述

You are given positive integers `low`, `high`, and `k`.

A number is **beautiful** if it meets both of the following conditions:

- The count of even digits in the number is equal to the count of odd digits.
- The number is divisible by `k`.

Return _the number of beautiful integers in the range_ `[low, high]`.

**Example 1:**

**Input:** low = 10, high = 20, k = 3  
**Output:** 2  
**Explanation:** There are 2 beautiful integers in the given range: [12,18].   
- 12 is beautiful because it contains 1 odd digit and 1 even digit, and is divisible by k = 3.  
- 18 is beautiful because it contains 1 odd digit and 1 even digit, and is divisible by k = 3.  
Additionally we can see that:  
- 16 is not beautiful because it is not divisible by k = 3.  
- 15 is not beautiful because it does not contain equal counts even and odd digits.  
It can be shown that there are only 2 beautiful integers in the given range.  

**Example 2:**

**Input:** low = 1, high = 10, k = 1  
**Output:** 1  
**Explanation:** There is 1 beautiful integer in the given range: [10].  
- 10 is beautiful because it contains 1 odd digit and 1 even digit, and is divisible by k = 1.  
It can be shown that there is only 1 beautiful integer in the given range.  

**Example 3:**

**Input:** low = 5, high = 5, k = 2  
**Output:** 0  
**Explanation:** There are 0 beautiful integers in the given range.  
- 5 is not beautiful because it is not divisible by k = 2 and it does not contain equal even and odd digits.  

**Constraints:**

- `0 < low <= high <= 109`
- `0 < k <= 20`

#### 解题思路

- 经典**数位DP**题目。  
DP的状态为`[target, idx, val, greater, smaller, count, remainder]`，其中`target`为上界，`idx`为当前所看为第几个数位，`val`为该数位的数值，`greater`为当数位长度与上界相同时，该数是否大于上界，`smaller`类似`greater`，`count`为记录奇数数位与偶数数位的个数是否相同，`remainder`为记录当前代表的数是否可以被`k`整除。

#### 复杂度

- 时间复杂度：$$O(3\times{10}\times n^2 k) = O(n^2k)$$
- 空间复杂度：$$O(n^2k)$$

#### 代码

```python
from functools import cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:

        @cache
        def dfs(target, idx, val, greater, smaller, count, remainder):
            if idx == len(target) - 1:
                if greater:
                    return 0
                if not greater and not smaller:
                    if val > int(target[idx]):
                        return 0
                return 1 if count == 0 and remainder % k == 0 else 0
            if not greater and not smaller:
                if val > int(target[idx]):
                    greater = True
                elif val < int(target[idx]):
                    smaller = True
            res = 0
            if count == 0 and remainder % k == 0:
                res += 1
            for i in range(10):
                res += dfs(
                    target,
                    idx + 1,
                    i,
                    greater,
                    smaller,
                    count + 1 if i % 2 == 0 else count - 1,
                    (remainder * 10 + i) % k
                )
            return res

        def dp(target):
            res = 0
            for i in range(1, 10):
                res += dfs(target=str(target), idx=0, val=i, greater=i > int(str(target)[0]), smaller=i < int(str(target)[0]), count=1 if i % 2 == 0 else -1, remainder=i % k)
            return res
            
        return dp(high) - dp(low - 1)
```
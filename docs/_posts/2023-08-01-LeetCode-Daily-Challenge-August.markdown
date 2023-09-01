---
layout: post
title:  "LeetCode Daily Challenge August"
date:   2023-08-01 13:16:49 +0800
categories: LeetCode Algorithm
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

This post is for LeetCode Daily Challenge August.

{% include toc %}

## 1. Combinations (77)

#### 难度

- **Medium**

#### 问题描述

Given two integers `n` and `k`, return _all possible combinations of_ `k` _numbers chosen from the range_ `[1, n]`.

You may return the answer in **any order**.

**Example 1:**

**Input:** n = 4, k = 2
**Output:** [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
**Explanation:** There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.

**Example 2:**

**Input:** n = 1, k = 1
**Output:** [[1]]
**Explanation:** There is 1 choose 1 = 1 total combination.

**Constraints:**

- `1 <= n <= 20`
- `1 <= k <= n`

#### 解题思路

- **回溯法**，**递归**：
	- Base case:
		- 若求所含元素数量为**1**的情况，则返回**[1, 2, 3, ..., n]**
	- Recurrence function:
		- 若求所含元素数量为**n**的情况，则为所含元素数量为**n - 1**的情况加上任一大于末尾元素且不曾在该组合中出现过的元素，
		- 对于任一组合，若末尾元素 + 剩余元素长度 > n，则直接剪枝并直接返回。

#### 复杂度

- 时间复杂度：$O(\frac{n!}{k!(n-k)!})$
- 空间复杂度：$O(\frac{n!}{k!(n-k)!})$

#### 代码

```python
from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        def dfs(length):
            if length == 1:
                return [[i] for i in range(1, n + 1)]
            base = dfs(length - 1)
            res = []
            dist = k - length
            for val in base:
                if val[-1] + dist > n:
                    continue
                for i in range(val[-1] + 1, n + 1):
                    res.append(val + [i])
            return res

        return dfs(k)
```

---

## 2. Permutations (46)

#### 难度

- **Medium**

#### 问题描述

Given an array `nums` of distinct integers, return _all the possible permutations_. You can return the answer in **any order**.

**Example 1:**

**Input:** nums = [1,2,3]
**Output:** [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

**Example 2:**

**Input:** nums = [0,1]
**Output:** [[0,1],[1,0]]

**Example 3:**

**Input:** nums = [1]
**Output:** [[1]]

**Constraints:**

- `1 <= nums.length <= 6`
- `-10 <= nums[i] <= 10`
- All the integers of `nums` are **unique**.

#### 解题思路

- **回溯法**，**递归**：
	- Base case:
		- 若求所含元素数量为**1**的情况，则返回**[1, 2, 3, ..., n]**
	- Recurrence function:
		- 若求所含元素数量为**n**的情况，则为所含元素数量为**n - 1**的情况加上任一大于末尾元素且不曾在该组合中出现过的元素，


#### 复杂度

- 时间复杂度：$O(n\times{n!})$
- 空间复杂度：$O(n!)$

#### 代码

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        def dfs(length):
            if length == 1:
                return [[nums[i]] for i in range(len(nums))]
            res = []
            for val in dfs(length - 1):
                for num in nums:
                    if num in val:
                        continue
                    res.append(val + [num])
            return res

        return dfs(len(nums))
```

---

## Letter Combinations of a Phone Number (17)

#### 难度

- **Medium**

#### 问题描述

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![](https://assets.leetcode.com/uploads/2022/03/15/1200px-telephone-keypad2svg.png)

**Example 1:**

**Input:** digits = "23"
**Output:** ["ad","ae","af","bd","be","bf","cd","ce","cf"]

**Example 2:**

**Input:** digits = ""
**Output:** []

**Example 3:**

**Input:** digits = "2"
**Output:** ["a","b","c"]

**Constraints:**

- `0 <= digits.length <= 4`
- `digits[i]` is a digit in the range `['2', '9']`.

#### 解题思路

- 利用**递归**将所有的组合找到。

#### 复杂度

- 时间复杂度：$O(4^n)$
- 空间复杂度：$O(n)$

#### 代码

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        mapping = ['', '', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
        
        def dfs(length):
            if length == 0:
                return []
            if length == 1:
                return list(mapping[int(digits[0])])
            res = []
            for base in dfs(length - 1):
                for val in list(mapping[int(digits[length - 1])]):
                    res.append(base + val)
            return res
        
        return dfs(len(digits))
```

---

## Word Break (139)

#### 难度

- **Medium**

#### 问题描述

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**

**Input:** s = "leetcode", wordDict = ["leet","code"]
**Output:** true
**Explanation:** Return true because "leetcode" can be segmented as "leet code".

**Example 2:**

**Input:** s = "applepenapple", wordDict = ["apple","pen"]
**Output:** true
**Explanation:** Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.

**Example 3:**

**Input:** s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
**Output:** false

**Constraints:**

- `1 <= s.length <= 300`
- `1 <= wordDict.length <= 1000`
- `1 <= wordDict[i].length <= 20`
- `s` and `wordDict[i]` consist of only lowercase English letters.
- All the strings of `wordDict` are **unique**.

#### 解题思路

- **DP**，**记忆法**
状态转移函数：`dp[i] = dp[i + word.length]`

#### 复杂度

- 时间复杂度：$O(mnk)$
- 空间复杂度：$O(n)$

#### 代码

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordSet = set(wordDict)
        
        @cache
        def dfs(left):
            if left == len(s):
                return True
            res = False
            for right in range(left, len(s)):
                if s[left:right + 1] in wordSet:
                    res = True and dfs(right + 1)
                if res:
                    return res
            return res
        
        return dfs(0)
```

---

## Unique Binary Search Trees II (95)

#### 难度

- **Medium**

#### 问题描述

Given an integer `n`, return _all the structurally unique **BST'**s (binary search trees), which has exactly_ `n` _nodes of unique values from_ `1` _to_ `n`. Return the answer in **any order**.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

**Input:** n = 3
**Output:** [[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]

**Example 2:**

**Input:** n = 1
**Output:** [[1]]

**Constraints:**

- `1 <= n <= 8`

#### 解题思路

- **递归**
题目求结构上独特的BST的数量，假设节点为`n`，则在左子树的节点为`[1, ..., n]`，可能的左子树为`dfs([1, ..., n])`，右子树也相似。则该情况下的BST数量为左子树的数量乘右子树的数量。枚举所有可能的情况。

#### 复杂度

- 时间复杂度：$O(3^n)$
- 空间复杂度：$O(3^n)$

#### 代码

```python
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:

		@cache
        def dfs(left, right):
            if right < left:
                return [None]
            res = []
            for i in range(left, right + 1):
                for lst in dfs(left, i - 1):
                    for rst in dfs(i + 1, right):
                        node = TreeNode(i, lst, rst)
                        res.append(node)
            return res
        
        return dfs(1, n)  # type: ignore
```

---

## Number of Music Playlists (920)

#### 难度

- **Hard**

#### 问题描述

Your music player contains `n` different songs. You want to listen to `goal` songs (not necessarily different) during your trip. To avoid boredom, you will create a playlist so that:

- Every song is played **at least once**.
- A song can only be played again only if `k` other songs have been played.

Given `n`, `goal`, and `k`, return _the number of possible playlists that you can create_. Since the answer can be very large, return it **modulo** `109 + 7`.

**Example 1:**

**Input:** n = 3, goal = 3, k = 1
**Output:** 6
**Explanation:** There are 6 possible playlists: [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], and [3, 2, 1].

**Example 2:**

**Input:** n = 2, goal = 3, k = 0
**Output:** 6
**Explanation:** There are 6 possible playlists: [1, 1, 2], [1, 2, 1], [2, 1, 1], [2, 2, 1], [2, 1, 2], and [1, 2, 2].

**Example 3:**

**Input:** n = 2, goal = 3, k = 1
**Output:** 2
**Explanation:** There are 2 possible playlists: [1, 2, 1] and [2, 1, 2].

#### 解题思路

- **线性规划**
- DP的状态为`[song, listLength]`，转移函数有两种：
	- 一种是加入一首新的歌曲，则可能性乘上剩余可选择的歌曲数。
	- 另一种为已经有`k`首歌曲被播放过了，则可能行为可不重复播放的歌曲数。
	- 将两种可能相加即为当前状态的可能数目。

#### 复杂度

- 时间复杂度：$O(n\times goal)$
- 空间复杂度：$O(n\times goal)$

#### 代码

```python
from functools import cache

class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        MOD = 10**9 + 7
        
        @cache
        def dfs(songs, length):
            if songs > length or songs < 0:
                return 0
            if songs == 0 and length == 0:
                return 1
            res = dfs(songs - 1, length - 1) * (n - songs + 1)
            res %= MOD
            if songs > k:
                res += dfs(songs, length - 1) * (songs - k)
                res %= MOD
            return res % MOD
        
        return dfs(n, goal)
```

---

## Search a 2D Matrix (74)

#### 难度

- **Medium**

#### 问题描述

You are given an `m x n` integer matrix `matrix` with the following two properties:

- Each row is sorted in non-decreasing order.
- The first integer of each row is greater than the last integer of the previous row.

Given an integer `target`, return `true` _if_ `target` _is in_ `matrix` _or_ `false` _otherwise_.

You must write a solution in `O(log(m * n))` time complexity.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/05/mat.jpg)

**Input:** matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
**Output:** true

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/05/mat2.jpg)

**Input:** matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
**Output:** false

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 100`
- `-104 <= matrix[i][j], target <= 104`

#### 解题思路

- **二分查找法**
分别对行和列做二分查找。

#### 复杂度

- 时间复杂度：$O(\log m + \log n)$
- 空间复杂度：$O(m)$

#### 代码

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        firstCol = list(zip(*matrix))[0]
        
        row = bisect.bisect_right(firstCol, target)
        
        if row == 0:
            return False
        
        col = bisect.bisect_left(matrix[row - 1], target)

        return False if col == n else matrix[row - 1][col] == target
```

---

## Search in Rotated Sorted Array (33)

#### 难度

- **Medium**

#### 问题描述

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is **possibly rotated** at an unknown pivot index `k` (`1 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,5,6,7]` might be rotated at pivot index `3` and become `[4,5,6,7,0,1,2]`.

Given the array `nums` **after** the possible rotation and an integer `target`, return _the index of_ `target` _if it is in_ `nums`_, or_ `-1` _if it is not in_ `nums`.

You must write an algorithm with `O(log n)` runtime complexity.

**Example 1:**

**Input:** nums = [4,5,6,7,0,1,2], target = 0
**Output:** 4

**Example 2:**

**Input:** nums = [4,5,6,7,0,1,2], target = 3
**Output:** -1

**Example 3:**

**Input:** nums = [1], target = 0
**Output:** -1

**Constraints:**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- All values of `nums` are **unique**.
- `nums` is an ascending array that is possibly rotated.
- `-104 <= target <= 104`

#### 解题思路

- **二分搜索法**
先利用二分搜索法找到旋转的分割点在哪，然后分别对`分割点`两侧的子数组做二分查找。

#### 复杂度

- 时间复杂度：$O(\log n)$
- 空间复杂度：$O(1)$

#### 代码

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        pivot = -1
        
        if not nums[left] <= nums[right]:
            while left < right:
                mid = (left + right) // 2
                if nums[mid] > nums[right]:
                    left = mid + 1
                else:
                    right = mid
            pivot = left

        if pivot == -1:
            idx = bisect.bisect_left(nums, target)
            if idx == len(nums):
                return -1
            return idx if nums[idx] == target else - 1
        else:
            idx1 = bisect.bisect_left(nums[:pivot], target)
            idx2 = bisect.bisect_left(nums[pivot:], target) + pivot
            for idx in (idx1, idx2):
                if idx == len(nums):
                    continue
                if nums[idx] == target:
                    return idx
            return -1
```

---

## Minimize the Maximum Difference of Pairs (2616)

#### 难度

- **Medium**

#### 问题描述

You are given a **0-indexed** integer array `nums` and an integer `p`. Find `p` pairs of indices of `nums` such that the **maximum** difference amongst all the pairs is **minimized**. Also, ensure no index appears more than once amongst the `p` pairs.

Note that for a pair of elements at the index `i` and `j`, the difference of this pair is `|nums[i] - nums[j]|`, where `|x|` represents the **absolute** **value** of `x`.

Return _the **minimum** **maximum** difference among all_ `p` _pairs._ We define the maximum of an empty set to be zero.

**Example 1:**

**Input:** nums = [10,1,2,7,1,3], p = 2
**Output:** 1
**Explanation:** The first pair is formed from the indices 1 and 4, and the second pair is formed from the indices 2 and 5. 
The maximum difference is max(|nums[1] - nums[4]|, |nums[2] - nums[5]|) = max(0, 1) = 1. Therefore, we return 1.

**Example 2:**

**Input:** nums = [4,2,1,2], p = 1
**Output:** 0
**Explanation:** Let the indices 1 and 3 form a pair. The difference of that pair is |2 - 2| = 0, which is the minimum we can attain.

**Constraints:**

- `1 <= nums.length <= 105`
- `0 <= nums[i] <= 109`
- `0 <= p <= (nums.length)/2`

#### 解题思路

- **Greedy**, **binary search**
	- 对答案进行二分查找，答案的lower limit为0，upper limit为`max(nums)`，中间值为`(low + high) // 2`
	- 检查当相邻数最大差距为中间值是是否可以找到`p`对数值对，利用贪心法进行检查。

#### 复杂度

- 时间复杂度：$O(n\log{n} + n\log{\max{(nums)}})$
- 空间复杂度：$O(1)$

#### 代码

```python
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        nums.sort()
        n = len(nums)
        left, right = 0, max(nums)
        
        def check(diff):
            count = 0
            idx = 0
            while True:
                if idx + 1 >= n:
                    break
                if abs(nums[idx] - nums[idx + 1]) <= diff:
                    count += 1
                    idx += 2
                else:
                    idx += 1
                if count >= p:
                    return True
            return count >= p
        
        while left < right:
            mid = (left + right) >> 1
            if check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
```

---

## Search in Rotated Sorted Array II (81)

#### 难度

- **Medium**

#### 问题描述

There is an integer array `nums` sorted in non-decreasing order (not necessarily with **distinct** values).

Before being passed to your function, `nums` is **rotated** at an unknown pivot index `k` (`0 <= k < nums.length`) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]` (**0-indexed**). For example, `[0,1,2,4,4,4,5,6,6,7]` might be rotated at pivot index `5` and become `[4,5,6,6,7,0,1,2,4,4]`.

Given the array `nums` **after** the rotation and an integer `target`, return `true` _if_ `target` _is in_ `nums`_, or_ `false` _if it is not in_ `nums`_._

You must decrease the overall operation steps as much as possible.

**Example 1:**

**Input:** nums = [2,5,6,0,0,1,2], target = 0
**Output:** true

**Example 2:**

**Input:** nums = [2,5,6,0,0,1,2], target = 3
**Output:** false

**Constraints:**

- `1 <= nums.length <= 5000`
- `-104 <= nums[i] <= 104`
- `nums` is guaranteed to be rotated at some pivot.
- `-104 <= target <= 104`

**Follow up:** This problem is similar to [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/), but `nums` may contain **duplicates**. Would this affect the runtime complexity? How and why?

#### 解题思路

-  **Binary search**
- 做两次二分查找，第一次找`pivot`在何处，第二次在两个子数组中找`target`。
	- 若二分查找时上界与下界的值一样，有两种情况：
		1. `pivot`在上下界之间，此时将下界+1。
		2. `pivot`在上界之前，此时将上界-1。

#### 复杂度

- 时间复杂度：$O(n)$, $\Omega(\log{n})$
- 空间复杂度：$O(1)$

#### 代码

```python
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums) - 1
        
        while left < right:
            if nums[left] == nums[right]:
                if left > 0 and nums[left] < nums[left - 1]:
                    right -= 1
                else:
                    left += 1
                continue
            mid = (left + right) // 2
            if nums[mid] <= nums[right]:
                right = mid
            else:
                left = mid + 1
        
        pivot = left
        
        idxLeft = bisect.bisect_left(nums[:pivot], target)
        hasLeft = True if nums[idxLeft] == target else False
        idxRight = bisect.bisect_left(nums[pivot:], target) + pivot
        hasRight = True if idxRight < len(nums) and nums[idxRight] == target else False

        return hasLeft or hasRight
```

---

## Coin Change II (518)

#### 难度

- **Medium**

#### 问题描述

You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money.

Return _the number of combinations that make up that amount_. If that amount of money cannot be made up by any combination of the coins, return `0`.

You may assume that you have an infinite number of each kind of coin.

The answer is **guaranteed** to fit into a signed **32-bit** integer.

**Example 1:**

**Input:** amount = 5, coins = [1,2,5]
**Output:** 4
**Explanation:** there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

**Example 2:**

**Input:** amount = 3, coins = [2]
**Output:** 0
**Explanation:** the amount of 3 cannot be made up just with coins of 2.

**Example 3:**

**Input:** amount = 10, coins = [10]
**Output:** 1

**Constraints:**

- `1 <= coins.length <= 300`
- `1 <= coins[i] <= 5000`
- All the values of `coins` are **unique**.
- `0 <= amount <= 5000`

#### 解题思路

- 经典**完全背包**问题。

#### 复杂度

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(mn)$

#### 代码

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        coins.sort(reverse=True)
        
        @cache
        def dfs(idx, target):
            if target < 0 or idx >= n:
                return 0
            if target == 0:
                return 1
            res = 0
            res += dfs(idx, target - coins[idx])
            res += dfs(idx + 1, target)
            return res

        return dfs(0, amount)
```

---

## Unique Paths II (63)

#### 难度

- **Medium**

#### 问题描述

You are given an `m x n` integer array `grid`. There is a robot initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

An obstacle and space are marked as `1` or `0` respectively in `grid`. A path that the robot takes cannot include **any** square that is an obstacle.

Return _the number of possible unique paths that the robot can take to reach the bottom-right corner_.

The testcases are generated so that the answer will be less than or equal to `2 * 109`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/11/04/robot1.jpg)

**Input:** obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
**Output:** 2
**Explanation:** There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/11/04/robot2.jpg)

**Input:** obstacleGrid = [[0,1],[0,0]]
**Output:** 1

**Constraints:**

- `m == obstacleGrid.length`
- `n == obstacleGrid[i].length`
- `1 <= m, n <= 100`
- `obstacleGrid[i][j]` is `0` or `1`.

#### 解题思路

- **动态规划**
状态转移函数：$dp[i][j] = dp[i - 1][j] + dp[i][j - 1] + dp[i][j]$

#### 复杂度

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(mn)$

#### 代码

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        if m == 1 and n == 1:
            return 1 if obstacleGrid[0][0] == 0 else 0
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1] == 1:
            return 0
        
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1 - obstacleGrid[0][0]
        
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    continue
                fromAbove = 0 if i - 1 < 0 else dp[i - 1][j]
                fromLeft = 0 if j - 1 < 0 else dp[i][j - 1]
                dp[i][j] += fromAbove + fromLeft

        return dp[m - 1][n - 1]
```

---

## Check if There is a Valid Partition For The Array (2369)

#### 难度

- **Medium**

#### 问题描述

You are given a **0-indexed** integer array `nums`. You have to partition the array into one or more **contiguous** subarrays.

We call a partition of the array **valid** if each of the obtained subarrays satisfies **one** of the following conditions:

1. The subarray consists of **exactly** `2` equal elements. For example, the subarray `[2,2]` is good.
2. The subarray consists of **exactly** `3` equal elements. For example, the subarray `[4,4,4]` is good.
3. The subarray consists of **exactly** `3` consecutive increasing elements, that is, the difference between adjacent elements is `1`. For example, the subarray `[3,4,5]` is good, but the subarray `[1,3,5]` is not.

Return `true` _if the array has **at least** one valid partition_. Otherwise, return `false`.

**Example 1:**

**Input:** nums = [4,4,4,5,6]
**Output:** true
**Explanation:** The array can be partitioned into the subarrays [4,4] and [4,5,6].
This partition is valid, so we return true.

**Example 2:**

**Input:** nums = [1,1,1,2]
**Output:** false
**Explanation:** There is no valid partition for this array.

**Constraints:**

- `2 <= nums.length <= 105`
- `1 <= nums[i] <= 106`

#### 解题思路

- **Dynamic programming**
- 状态转移函数：
	- 若满足条件1，则为`dp[i]  |= dp[i - 2]`
	- 若满足条件2或3，则为`dp[i] |= dp[i - 3]`

#### 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$

#### 代码

```java
class Solution {
    public boolean validPartition(int[] nums) {
        int n = nums.length;
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for (int i = 1; i < dp.length; i++) {
            if (i > 1 && nums[i - 1] == nums[i - 2]) {
                dp[i] |= dp[i - 2];
            }
            if (i > 2 && nums[i - 1] == nums[i - 2] && nums[i - 1] == nums[i - 3]) {
                dp[i] |= dp[i - 3];
            }
            if (i > 2 && nums[i - 1] - nums[i - 2] == 1 && nums[i - 1] - nums[i - 3] == 2) {
                dp[i] |= dp[i - 3];
            }
        }
        
        return dp[n];
    }
}
```

---

## Kth Largest Element in an Array (215)

#### 难度

- **Medium**

#### 问题描述

Given an integer array `nums` and an integer `k`, return _the_ `kth` _largest element in the array_.

Note that it is the `kth` largest element in the sorted order, not the `kth` distinct element.

Can you solve it without sorting?

**Example 1:**

**Input:** nums = [3,2,1,5,6,4], k = 2
**Output:** 5

**Example 2:**

**Input:** nums = [3,2,3,1,2,4,5,5,6], k = 4
**Output:** 4

**Constraints:**

- `1 <= k <= nums.length <= 105`
- `-104 <= nums[i] <= 104`

#### 解题思路

- 经典**Quickselect**题目。

#### 复杂度

- 时间复杂度：
	- Average case: $O(n)$
	- Worst case: $O(n^2)$
- 空间复杂度：$O(n)$

#### 代码

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        List<Integer> vals = new ArrayList<>();

        for (int val : nums) {
            vals.add(val);
        }
        
        return dfs(vals, k);
    }

    private int dfs(List<Integer> vals, int k) {
        int pivotIdx = new Random().nextInt(vals.size());
        int pivot = vals.get(pivotIdx);

        List<Integer> left = new ArrayList<>();
        List<Integer> mid = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        
        for (int val : vals) {
            if (val > pivot) {
                left.add(val);
            } else if (val == pivot) {
                mid.add(val);
            } else {
                right.add(val);
            }
        }
        
        if (left.size() >= k) {
            return dfs(left, k);
        } else if (left.size() + mid.size() >= k) {
            return pivot;
        } else {
            return dfs(right, k - left.size() - mid.size());
        }
    }
}
```

---

## Partition List (86)

## 难度

- **Medium**

## 问题描述

Given the `head` of a linked list and a value `x`, partition it such that all nodes **less than** `x` come before nodes **greater than or equal** to `x`.

You should **preserve** the original relative order of the nodes in each of the two partitions.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/01/04/partition.jpg)

**Input:** head = [1,4,3,2,5,2], x = 3
**Output:** [1,2,2,4,3,5]

**Example 2:**

**Input:** head = [2,1], x = 2
**Output:** [1,2]

**Constraints:**

- The number of nodes in the list is in the range `[0, 200]`.
- `-100 <= Node.val <= 100`
- `-200 <= x <= 200`

## 解题思路

- **双指针**
新建两个空的链表，遍历原来链表的同时依照条件将节点加入两个链表中，最后将两个链表连接起来并返回。

## 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$

## 代码

```java
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode smaller = new ListNode(0);
        ListNode bigger = new ListNode(0);
        ListNode smallerNow = smaller;
        ListNode biggerNow = bigger;
        ListNode now = head;

        while (now != null) {
            if (now.val < x) {
                smallerNow.next = new ListNode(now.val);
                smallerNow = smallerNow.next;
            } else {
                biggerNow.next = new ListNode(now.val);
                biggerNow = biggerNow.next;
            }

            now = now.next;
        }
        
        smallerNow.next = bigger.next;

        return smaller.next;
    }
}
```

---

## Sliding Window Maximum (239)

## 难度

- **Hard**

## 问题描述

You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return _the max sliding window_.

**Example 1:**

**Input:** nums = [1,3,-1,-3,5,3,6,7], k = 3
**Output:** [3,3,5,5,6,7]
**Explanation:** 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       **3**
 1 [3  -1  -3] 5  3  6  7       **3**
 1  3 [-1  -3  5] 3  6  7       **5**
 1  3  -1 [-3  5  3] 6  7       **5**
 1  3  -1  -3 [5  3  6] 7       **6**
 1  3  -1  -3  5 [3  6  7]      **7**

**Example 2:**

**Input:** nums = [1], k = 1
**Output:** [1]

**Constraints:**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`
- `1 <= k <= nums.length`

## 解题思路

- 维护一个**单调队列**来找到滑动窗口大小为`k`时的最大值。

## 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(k)$

## 代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        LinkedList<Integer> queue = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];

        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && nums[queue.getLast()] < nums[i]) { queue.removeLast(); }
            queue.add(i);
            if (i - queue.getFirst() >= k) { queue.removeFirst(); }
            if (i >= k - 1) { res[i - k + 1] = nums[queue.getFirst()]; }
        }
        
        return res;
    }
}
```

---

## 01 Matrix (542)

## 难度

- **Medium**

## 问题描述

Given an `m x n` binary matrix `mat`, return _the distance of the nearest_ `0` _for each cell_.

The distance between two adjacent cells is `1`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/04/24/01-1-grid.jpg)

**Input:** mat = [[0,0,0],[0,1,0],[0,0,0]]
**Output:** [[0,0,0],[0,1,0],[0,0,0]]

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/04/24/01-2-grid.jpg)

**Input:** mat = [[0,0,0],[0,1,0],[1,1,1]]
**Output:** [[0,0,0],[0,1,0],[1,2,1]]

**Constraints:**

- `m == mat.length`
- `n == mat[i].length`
- `1 <= m, n <= 104`
- `1 <= m * n <= 104`
- `mat[i][j]` is either `0` or `1`.
- There is at least one `0` in `mat`.

## 解题思路

- **多源BFS**
将所有元素为0的位置作为起始点做BFS。

## 复杂度

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(mn)$

## 代码

```java
class Solution {
    public int[][] updateMatrix(int[][] mat) {
        int[][] res = new int[mat.length][mat[0].length];
        Deque<int[]> deque = new LinkedList<>();
        boolean[][] vis = new boolean[mat.length][mat[0].length];

        for (int i = 0; i < res.length; i++) {
            for (int j = 0; j < res[0].length; j++) {
                if (mat[i][j] == 0) {
                    vis[i][j] = true;
                    deque.add(new int[] {i, j});
                } else {
                    res[i][j] = Integer.MAX_VALUE;
                }
            }
        }
        
        
        while (!deque.isEmpty()) {
            int[] location = deque.pop();

            for (int[] dir : DIRS) {
                int i = location[0] + dir[0];
                int j = location[1] + dir[1];
                if (i < 0 || j < 0 || i >= res.length || j >= res[0].length) { continue; }
                if (vis[i][j]) { continue; }
                res[i][j] = res[location[0]][location[1]] + 1;
                vis[i][j] = true;
                deque.add(new int[] {i, j});
            }
        }
        
        return res;
    }
}
```

---

## Maximal Network Rank (1615)

## 难度

- **Medium**

## 问题描述

There is an infrastructure of `n` cities with some number of `roads` connecting these cities. Each `roads[i] = [ai, bi]` indicates that there is a bidirectional road between cities `ai` and `bi`.

The **network rank** of **two different cities** is defined as the total number of **directly** connected roads to **either** city. If a road is directly connected to both cities, it is only counted **once**.

The **maximal network rank** of the infrastructure is the **maximum network rank** of all pairs of different cities.

Given the integer `n` and the array `roads`, return _the **maximal network rank** of the entire infrastructure_.

**Example 1:**

**![](https://assets.leetcode.com/uploads/2020/09/21/ex1.png)**

**Input:** n = 4, roads = [[0,1],[0,3],[1,2],[1,3]]
**Output:** 4
**Explanation:** The network rank of cities 0 and 1 is 4 as there are 4 roads that are connected to either 0 or 1. The road between 0 and 1 is only counted once.

**Example 2:**

**![](https://assets.leetcode.com/uploads/2020/09/21/ex2.png)**

**Input:** n = 5, roads = [[0,1],[0,3],[1,2],[1,3],[2,3],[2,4]]
**Output:** 5
**Explanation:** There are 5 roads that are connected to cities 1 or 2.

**Example 3:**

**Input:** n = 8, roads = [[0,1],[1,2],[2,3],[2,4],[5,6],[5,7]]
**Output:** 5
**Explanation:** The network rank of 2 and 5 is 5. Notice that all the cities do not have to be connected.

**Constraints:**

- `2 <= n <= 100`
- `0 <= roads.length <= n * (n - 1) / 2`
- `roads[i].length == 2`
- `0 <= ai, bi <= n-1`
- `ai != bi`
- Each pair of cities has **at most one** road connecting them.

## 解题思路

- **枚举法**
枚举所有的城市组合并返回最大值。

## 复杂度

- 时间复杂度：$O(E+V^2)$
- 空间复杂度：$O(V)$

## 代码

```java
class Solution {
    public int maximalNetworkRank(int n, int[][] roads) {
        HashMap<Integer, Integer> connectedRoads = new HashMap<>();
        HashMap<Integer, HashSet<Integer>> connectedCities = new HashMap<>();

        for (int i = 0; i < roads.length; i++) {
            int cityA = roads[i][0], cityB = roads[i][1];
            connectedRoads.compute(cityA, (x, y) -> y == null ? 1 : y + 1);
            connectedRoads.compute(cityB, (x, y) -> y == null ? 1 : y + 1);
            connectedCities.computeIfAbsent(cityA, (x) -> new HashSet<>()).add(cityB);
            connectedCities.computeIfAbsent(cityB, (x) -> new HashSet<>()).add(cityA);
        }
        
        int res = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                boolean connected = connectedCities.getOrDefault(i, new HashSet<>()).contains(j);
                int tempRes = connectedRoads.getOrDefault(i, 0) + connectedRoads.getOrDefault(j, 0);
                if (connected) { tempRes--; }
                res = Math.max(res, tempRes);
            }
        }
        
        return res;
    }
}
```

---

## Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree (1489)

## 难度

- **Hard**

## 问题描述

Given a weighted undirected connected graph with `n` vertices numbered from `0` to `n - 1`, and an array `edges` where `edges[i] = [ai, bi, weighti]` represents a bidirectional and weighted edge between nodes `ai` and `bi`. A minimum spanning tree (MST) is a subset of the graph's edges that connects all vertices without cycles and with the minimum possible total edge weight.

Find _all the critical and pseudo-critical edges in the given graph's minimum spanning tree (MST)_. An MST edge whose deletion from the graph would cause the MST weight to increase is called a _critical edge_. On the other hand, a pseudo-critical edge is that which can appear in some MSTs but not all.

Note that you can return the indices of the edges in any order.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/06/04/ex1.png)

**Input:** n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
**Output:** [[0,1],[2,3,4,5]]
**Explanation:** The figure above describes the graph.
The following figure shows all the possible MSTs:
![](https://assets.leetcode.com/uploads/2020/06/04/msts.png)
Notice that the two edges 0 and 1 appear in all MSTs, therefore they are critical edges, so we return them in the first list of the output.
The edges 2, 3, 4, and 5 are only part of some MSTs, therefore they are considered pseudo-critical edges. We add them to the second list of the output.

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/06/04/ex2.png)

**Input:** n = 4, edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]]
**Output:** [[],[0,1,2,3]]
**Explanation:** We can observe that since all 4 edges have equal weight, choosing any 3 edges from the given 4 will yield an MST. Therefore all 4 edges are pseudo-critical.

**Constraints:**

- `2 <= n <= 100`
- `1 <= edges.length <= min(200, n * (n - 1) / 2)`
- `edges[i].length == 3`
- `0 <= ai < bi < n`
- `1 <= weighti <= 1000`
- All pairs `(ai, bi)` are **distinct**.

## 解题思路

- **最小生成树**，**并查集**
- 利用**Kruskal's Algorithm**生成最小生成树。
- 遍历每条边，若将该边移除后生成的最小生成树的权重变大，则该边**critical**；若权重不变，则将该边强制加入到最小生成树中，若权重还是相同，则该边**pseudo-critical**，否则该边不会存在在任何一个最小生成树中。

## 复杂度

- 时间复杂度：$O(E^2\times{\alpha({V})})$
- 空间复杂度：$O(m + n)$

## 代码

```python
class UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = [x for x in range(size)]
    
    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parents[px] = py

class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        edges = [x + [idx] for idx, x in enumerate(edges)] # type: ignore
        edges.sort(key=lambda x: x[2])

        def constructBST(edges: List[List[int]], mustEdge: List[int]):
            uf = UnionFind(n)
            uf.union(mustEdge[0], mustEdge[1])
            count = 0
            
            totalWeight = mustEdge[2]
            
            for i in range(len(edges)):
                a, b, w, _ = edges[i]
                if uf.find(a) == uf.find(b):
                    continue
                totalWeight += w
                count += 1
                if count == n - 1:
                    break
                uf.union(a, b)
            
            return totalWeight
        
        minWeight = constructBST(edges, [0, 0, 0])
        critical, noncritical = [], []
        
        for i in range(len(edges)):
            weight = constructBST(edges[:i] + edges[i + 1:], [0, 0, 0])
            if weight == minWeight:
                weight = constructBST(edges[:i] + edges[i + 1:], edges[i])
                if weight == minWeight:
                    noncritical.append(edges[i][-1])
            else:
                critical.append(edges[i][-1])
        
        return [critical, noncritical]
```

---

## Sort Items by Groups Respecting Dependencies (1203)

## 难度

- **Hard**

## 问题描述

There are `n` items each belonging to zero or one of `m` groups where `group[i]` is the group that the `i`-th item belongs to and it's equal to `-1` if the `i`-th item belongs to no group. The items and the groups are zero indexed. A group can have no item belonging to it.

Return a sorted list of the items such that:

- The items that belong to the same group are next to each other in the sorted list.
- There are some relations between these items where `beforeItems[i]` is a list containing all the items that should come before the `i`-th item in the sorted array (to the left of the `i`-th item).

Return any solution if there is more than one solution and return an **empty list** if there is no solution.

**Example 1:**

**![](https://assets.leetcode.com/uploads/2019/09/11/1359_ex1.png)**

**Input:** n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], beforeItems = [[],[6],[5],[6],[3,6],[],[],[]]
**Output:** [6,3,4,1,5,2,0,7]

**Example 2:**

**Input:** n = 8, m = 2, group = [-1,-1,1,0,0,1,0,-1], beforeItems = [[],[6],[5],[6],[3],[],[4],[]]
**Output:** []
**Explanation:** This is the same as example 1 except that 4 needs to be before 6 in the sorted list.

**Constraints:**

- `1 <= m <= n <= 3 * 104`
- `group.length == beforeItems.length == n`
- `-1 <= group[i] <= m - 1`
- `0 <= beforeItems[i].length <= n - 1`
- `0 <= beforeItems[i][j] <= n - 1`
- `i != beforeItems[i][j]`
- `beforeItems[i]` does not contain duplicates elements.

## 解题思路

- **拓扑排序**
分别从`物体`和`群组`的角度进行拓扑排序，`群组`的拓扑排序决定了不同群组间的先后顺序，`物体`的拓扑排序决定了群组内的先后顺序，若任一排序为空，则无解。

## 复杂度

- 时间复杂度：$O(V + E)$
- 空间复杂度：$O(V + E)$

## 代码

```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        maxGroup = max(group)
        newGroup = maxGroup + 1
        itemToGroup = dict()
        
        for i in range(len(group)):
            if group[i] == -1:
                group[i] = newGroup
                newGroup += 1
            itemToGroup[i] = group[i]

        graphItems = defaultdict(set)
        graphGroups = defaultdict(set)
        indegreesItems = [0] * len(group)
        indegreesGroups = [0] * (newGroup - 1)
        
        for idx, valList in enumerate(beforeItems):
            for val in valList:
                indegreesItems[idx] += 1
                graphItems[val].add(idx)
                if itemToGroup[idx] not in graphGroups[itemToGroup[val]]:
                    indegreesGroups[itemToGroup[idx]] += 1
                    graphGroups[itemToGroup[val]].add(itemToGroup[idx])
        
        def topoSort(indegrees, dependencies):
            q = deque([x for x in range(len(indegrees)) if indegrees[x] == 0])
            res = []
            while q:
                node = q.popleft()
                res.append(node)
                for adjNode in dependencies[node]:
                    indegrees[adjNode] -= 1
                    if indegrees[adjNode] == 0:
                        q.append(adjNode)
            return res if len(res) == len(indegrees) else []
        
        itemsOrder = topoSort(indegreesItems, graphItems)
        groupsOrder = topoSort(indegreesGroups, graphGroups)
        
        if not itemsOrder or not groupsOrder:
            return []
        
        res = defaultdict(list)
        
        for item in itemsOrder:
            res[itemToGroup[item]].append(item)
        
        temp = []
        
        for group in groupsOrder:
            temp.extend(res[group])
        
        return temp
```

---

## Repeated Substring Pattern (459)

## 难度

- **Easy**

## 问题描述

Given a string `s`, check if it can be constructed by taking a substring of it and appending multiple copies of the substring together.

**Example 1:**

**Input:** s = "abab"
**Output:** true
**Explanation:** It is the substring "ab" twice.

**Example 2:**

**Input:** s = "aba"
**Output:** false

**Example 3:**

**Input:** s = "abcabcabcabc"
**Output:** true
**Explanation:** It is the substring "abc" four times or the substring "abcabc" twice.

**Constraints:**

- `1 <= s.length <= 104`
- `s` consists of lowercase English letters.

## 解题思路

- **暴力解**

## 复杂度

- 时间复杂度：$O(n\times \sqrt n)$
- 空间复杂度：$O(n)$

## 代码

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        for i in range(len(s) // 2):
            if len(s) % (i + 1) != 0:
                continue
            substring = s[:i + 1]
            if substring * (len(s) // (i + 1)) == s:
                return True
        
        return False
```

---

## Excel Sheet Column Title (168)

## 难度

- **Easy**

## 问题描述

Given an integer `columnNumber`, return _its corresponding column title as it appears in an Excel sheet_.

For example:

A -> 1
B -> 2
C -> 3
...
Z -> 26
AA -> 27
AB -> 28 
...

**Example 1:**

**Input:** columnNumber = 1
**Output:** "A"

**Example 2:**

**Input:** columnNumber = 28
**Output:** "AB"

**Example 3:**

**Input:** columnNumber = 701
**Output:** "ZY"

**Constraints:**

- `1 <= columnNumber <= 231 - 1`

## 解题思路

- **模除**
本题类似将10进制的数转为26进制表达，tricky的点是这种26进制的表达中没有字母可以代表`0`，解决办法是只要当前的余数为0时，就将商减一，并将余数指向字母`z`。

## 复杂度

- 时间复杂度：$O(\log{n})$
- 空间复杂度：$O(1)$

## 代码

```java
class Solution {
    public String convertToTitle(int columnNumber) {
        char[] numToLetter = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ".toCharArray();
        int leftOver;
        List<Character> res = new LinkedList<>();
        
        while (columnNumber > 0) {
            leftOver = columnNumber % 26;
            columnNumber = columnNumber / 26;
            if (leftOver == 0) {
                columnNumber--;
                leftOver = 26;
            }
            res.add(0, numToLetter[leftOver]);
        }
        
        StringBuffer temp = new StringBuffer();

        for (var letter : res) {
            temp.append(letter);
        }
        
        return temp.toString();
    }
}
```

---

## Reorganize String (767)

## 难度

- **Medium**

## 问题描述

Given a string `s`, rearrange the characters of `s` so that any two adjacent characters are not the same.

Return _any possible rearrangement of_ `s` _or return_ `""` _if not possible_.

**Example 1:**

**Input:** s = "aab"
**Output:** "aba"

**Example 2:**

**Input:** s = "aaab"
**Output:** ""

**Constraints:**

- `1 <= s.length <= 500`
- `s` consists of lowercase English letters.

## 解题思路

- **优先队列**
每次将剩余个数`最多`和`次多`的字母从优先队列中弹出，加进答案并修改剩余次数后再加入优先队列，若最后优先队列只有一个元素并次数大于1则无答案。

## 复杂度

- 时间复杂度：$O(n\log k)$
- 空间复杂度：$O(k)$

## 代码

```python
class Solution:
    def reorganizeString(self, s: str) -> str:
        frequency = dict(Counter(s))
        heap = [(-y, x) for x, y in frequency.items()]
        heapq.heapify(heap)
        res = []
        
        while len(heap) > 1:
            freq, letter = heapq.heappop(heap)
            res.append(letter)
            freq2, letter2 = heapq.heappop(heap)
            res.append(letter2)
            if freq < -1:
                heapq.heappush(heap, (freq + 1, letter))
            if freq2 < -1:
                heapq.heappush(heap, (freq2 + 1, letter2))
        
        if heap:
            if heap[0][0] < -1:
                return ''
            res.append(heap[0][1])
        
        return ''.join(res)
```

---

## Text Justification (68)

## 难度

- **Hard**

## 问题描述

Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly `maxWidth` characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

**Note:**

- A word is defined as a character sequence consisting of non-space characters only.
- Each word's length is guaranteed to be greater than `0` and not exceed `maxWidth`.
- The input array `words` contains at least one word.

**Example 1:**

**Input:** words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
**Output:**
[
   "This    is    an",
   "example  of text",
   "justification.  "
]

**Example 2:**

**Input:** words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
**Output:**
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
**Explanation:** Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.
Note that the second line is also left-justified because it contains only one word.

**Example 3:**

**Input:** words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
**Output:**
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]

**Constraints:**

- `1 <= words.length <= 300`
- `1 <= words[i].length <= 20`
- `words[i]` consists of only English letters and symbols.
- `1 <= maxWidth <= 100`
- `words[i].length <= maxWidth`

## 解题思路

- **模拟**，**双指针**
当左右指针内的长度大于`maxWidth`时就新增一行，最后左右指针指的范围即为最后一行。

## 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$

## 代码

```python
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        lengths = [len(x) for x in words]
        res = []
        left, right = 0, 0
        tempLen = lengths[0]
        
        def justify(words: List[str], maxWidth: int):
            blanks = maxWidth - sum(len(x) for x in words)
            n = len(words) - 1
            res = []
            for word in words:
                res.append(word)
                if n:
                    res.append(' ' * math.ceil(blanks / n))
                    blanks -= math.ceil(blanks / n)
                    n -= 1
            if blanks:
                res.append(' ' * blanks)
            return ''.join(res)
        
        for right in range(1, len(lengths)):
            if tempLen + lengths[right] + 1 > maxWidth:
                res.append(justify(words[left:right], maxWidth))
                tempLen = 0
                left = right
                tempLen += lengths[right]
            else:
                tempLen += (lengths[right] + 1)
        
        temp = []
        tempLen = 0

        for i in range(left, right):
            temp.append(words[i] + ' ')
            tempLen += lengths[i] + 1
        
        temp.append(words[-1])
        tempLen += lengths[-1]
        temp.append(' ' * (maxWidth - tempLen))
        res.append(''.join(temp))
        
        return res
```

---

## Interleaving String (97)

## 难度

- **Medium**

## 问题描述

Given strings `s1`, `s2`, and `s3`, find whether `s3` is formed by an **interleaving** of `s1` and `s2`.

An **interleaving** of two strings `s` and `t` is a configuration where `s` and `t` are divided into `n` and `m`

substrings

respectively, such that:

- `s = s1 + s2 + ... + sn`
- `t = t1 + t2 + ... + tm`
- `|n - m| <= 1`
- The **interleaving** is `s1 + t1 + s2 + t2 + s3 + t3 + ...` or `t1 + s1 + t2 + s2 + t3 + s3 + ...`

**Note:** `a + b` is the concatenation of strings `a` and `b`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)

**Input:** s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
**Output:** true
**Explanation:** One way to obtain s3 is:
Split s1 into s1 = "aa" + "bc" + "c", and s2 into s2 = "dbbc" + "a".
Interleaving the two splits, we get "aa" + "dbbc" + "bc" + "a" + "c" = "aadbbcbcac".
Since s3 can be obtained by interleaving s1 and s2, we return true.

**Example 2:**

**Input:** s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"
**Output:** false
**Explanation:** Notice how it is impossible to interleave s2 with any other string to obtain s3.

**Example 3:**

**Input:** s1 = "", s2 = "", s3 = ""
**Output:** true

**Constraints:**

- `0 <= s1.length, s2.length <= 100`
- `0 <= s3.length <= 200`
- `s1`, `s2`, and `s3` consist of lowercase English letters.

## 解题思路

- **线性规划**
DP的状态为`[idx1, idx2]`,idx3可从另外两个idx推出。

## 复杂度

- 时间复杂度：$O(mn)$
- 空间复杂度：$O(mn)$

## 代码

```python
from functools import cache

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        len1, len2, len3 = len(s1), len(s2), len(s3)
        
        @cache
        def dfs(idx1, idx2, idx3):
            if idx3 == len3:
                return True if idx1 == len1 and idx2 == len2 else False
            res = False
            if idx1 < len1 and s1[idx1] == s3[idx3]:
                res = res or dfs(idx1 + 1, idx2, idx3 + 1)
            if idx2 < len2 and s2[idx2] == s3[idx3]:
                res = res or dfs(idx1, idx2 + 1, idx3 + 1)
            return res
        
        return dfs(0, 0, 0)
```

---

## Maximum Length of Pair Chain (646)

## 难度

- **Medium**

## 问题描述

You are given an array of `n` pairs `pairs` where `pairs[i] = [lefti, righti]` and `lefti < righti`.

A pair `p2 = [c, d]` **follows** a pair `p1 = [a, b]` if `b < c`. A **chain** of pairs can be formed in this fashion.

Return _the length longest chain which can be formed_.

You do not need to use up all the given intervals. You can select pairs in any order.

**Example 1:**

**Input:** pairs = [[1,2],[2,3],[3,4]]
**Output:** 2
**Explanation:** The longest chain is [1,2] -> [3,4].

**Example 2:**

**Input:** pairs = [[1,2],[7,8],[4,5]]
**Output:** 3
**Explanation:** The longest chain is [1,2] -> [4,5] -> [7,8].

**Constraints:**

- `n == pairs.length`
- `1 <= n <= 1000`
- `-1000 <= lefti < righti <= 1000`

## 解题思路

- **贪心**
将`pairs`按照末尾元素从小到大排序，可以通过贪心的方法选取末尾元素最小的可行`pair`进行更新，遍历`pairs`后即可得到答案

## 复杂度

- 时间复杂度：$O(n\log n)$
- 空间复杂度：$O(1)$

## 代码

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort(key=lambda x: x[1])
        cur = pairs[0][0] - 1
        res = 0
        
        for x, y in pairs:
            if x > cur:
                cur = y
                res += 1
        
        return res
```

---

## Frog Jump (403)

## 难度

- **Hard**

## 问题描述

A frog is crossing a river. The river is divided into some number of units, and at each unit, there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of `stones`' positions (in units) in sorted **ascending order**, determine if the frog can cross the river by landing on the last stone. Initially, the frog is on the first stone and assumes the first jump must be `2` unit.

If the frog's last jump was `k` units, its next jump must be either `k - 1`, `k`, or `k + 1` units. The frog can only jump in the forward direction.

**Example 1:**

**Input:** stones = [0,1,3,5,6,8,12,17]
**Output:** true
**Explanation:** The frog can jump to the last stone by jumping 1 unit to the 2nd stone, then 2 units to the 3rd stone, then 2 units to the 4th stone, then 3 units to the 6th stone, 4 units to the 7th stone, and 5 units to the 8th stone.

**Example 2:**

**Input:** stones = [0,1,2,3,4,8,9,11]
**Output:** false
**Explanation:** There is no way to jump to the last stone as the gap between the 5th and 6th stone is too large.

**Constraints:**

- `2 <= stones.length <= 2000`
- `0 <= stones[i] <= 231 - 1`
- `stones[0] == 0`
- `stones` is sorted in a strictly increasing order.

## 解题思路

- **线性规划**
DP的状态为`[stoneIdx, lastJump]`，若能跳到最后一块石头则返回`True`,否则返回`False`。

## 复杂度

- 时间复杂度：$O(n^2)$
- 空间复杂度：$O(n^2)$

## 代码

```python
from typing import List
from functools import cache

class Solution:
    def canCross(self, stones: List[int]) -> bool:
        n = len(stones)
        
        @cache
        def dfs(idx, lastJump):
            if idx == n - 1:
                return True
            res = False
            for i in range(idx + 1, n):
                dist = stones[i] - stones[idx]
                if dist < lastJump - 1:
                    continue
                if dist > lastJump + 1:
                    break
                if dist in range(lastJump - 1, lastJump + 2):
                    res |= dfs(i, dist)
            return res
        
        if stones[1] != 1:
            return False
        
        return dfs(1, 1)
```

---

## Implement Stack using Queues (225)

## 难度

- **Easy**

## 问题描述

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (`push`, `top`, `pop`, and `empty`).

Implement the `MyStack` class:

- `void push(int x)` Pushes element x to the top of the stack.
- `int pop()` Removes the element on the top of the stack and returns it.
- `int top()` Returns the element on the top of the stack.
- `boolean empty()` Returns `true` if the stack is empty, `false` otherwise.

**Notes:**

- You must use **only** standard operations of a queue, which means that only `push to back`, `peek/pop from front`, `size` and `is empty` operations are valid.
- Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.

**Example 1:**

**Input**
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
**Output**
[null, null, null, 2, 2, false]

**Explanation**
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False

**Constraints:**

- `1 <= x <= 9`
- At most `100` calls will be made to `push`, `pop`, `top`, and `empty`.
- All the calls to `pop` and `top` are valid.

**Follow-up:** Can you implement the stack using only one queue?

## 解题思路

- **模拟**
队列为FIFO，栈为FILO。在模拟队列时，要确保新加入的元素要在队列头，做法为加入新元素后，将其他元素弹出再加入到队列尾。

## 复杂度

- 时间复杂度：`pop`: $O(1)$, `push`: $O(n)$
- 空间复杂度：$O(1)$

## 代码

```python
from collections import deque

class MyStack:

    def __init__(self):
        self.stack = deque()

    def push(self, x: int) -> None:
        self.stack.append(x)
        for i in range(len(self.stack) - 1):
            self.stack.append(self.stack.popleft())

    def pop(self) -> int:
        return self.stack.popleft()

    def top(self) -> int:
        return self.stack[0]

    def empty(self) -> bool:
        return len(self.stack) == 0
```

---

## Minimum Penalty for a Shop (2483)

## 难度

- **Medium**

## 问题描述

You are given the customer visit log of a shop represented by a **0-indexed** string `customers` consisting only of characters `'N'` and `'Y'`:

- if the `ith` character is `'Y'`, it means that customers come at the `ith` hour
- whereas `'N'` indicates that no customers come at the `ith` hour.

If the shop closes at the `jth` hour (`0 <= j <= n`), the **penalty** is calculated as follows:

- For every hour when the shop is open and no customers come, the penalty increases by `1`.
- For every hour when the shop is closed and customers come, the penalty increases by `1`.

Return _the **earliest** hour at which the shop must be closed to incur a **minimum** penalty._

**Note** that if a shop closes at the `jth` hour, it means the shop is closed at the hour `j`.

**Example 1:**

**Input:** customers = "YYNY"
**Output:** 2
**Explanation:** 
- Closing the shop at the 0th hour incurs in 1+1+0+1 = 3 penalty.
- Closing the shop at the 1st hour incurs in 0+1+0+1 = 2 penalty.
- Closing the shop at the 2nd hour incurs in 0+0+0+1 = 1 penalty.
- Closing the shop at the 3rd hour incurs in 0+0+1+1 = 2 penalty.
- Closing the shop at the 4th hour incurs in 0+0+1+0 = 1 penalty.
Closing the shop at 2nd or 4th hour gives a minimum penalty. Since 2 is earlier, the optimal closing time is 2.

**Example 2:**

**Input:** customers = "NNNNN"
**Output:** 0
**Explanation:** It is best to close the shop at the 0th hour as no customers arrive.

**Example 3:**

**Input:** customers = "YYYY"
**Output:** 4
**Explanation:** It is best to close the shop at the 4th hour as customers arrive at each hour.

**Constraints:**

- `1 <= customers.length <= 105`
- `customers` consists only of characters `'Y'` and `'N'`.

## 解题思路

- **前缀和**
遍历数组并维护一个前缀和，若元素为`Y`则前缀和-1，反之+1，返回前缀和最小时的坐标。

## 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(1)$

## 代码

```python
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        minVal = 0
        idx = 0
        count = 0
        
        for i in range(len(customers)):
            if customers[i] == 'N':
                count += 1
            else:
                count -= 1
            if count < minVal:
                idx = i + 1
                minVal = count
        
        return idx
```

---

## Minimum Replacements to Sort the Array (2366)

## 难度

- **Hard**

## 问题描述

You are given a **0-indexed** integer array `nums`. In one operation you can replace any element of the array with **any two** elements that **sum** to it.

- For example, consider `nums = [5,6,7]`. In one operation, we can replace `nums[1]` with `2` and `4` and convert `nums` to `[5,2,4,7]`.

Return _the minimum number of operations to make an array that is sorted in **non-decreasing** order_.

**Example 1:**

**Input:** nums = [3,9,3]
**Output:** 2
**Explanation:** Here are the steps to sort the array in non-decreasing order:
- From [3,9,3], replace the 9 with 3 and 6 so the array becomes [3,3,6,3]
- From [3,3,6,3], replace the 6 with 3 and 3 so the array becomes [3,3,3,3,3]
There are 2 steps to sort the array in non-decreasing order. Therefore, we return 2.

**Example 2:**

**Input:** nums = [1,2,3,4,5]
**Output:** 0
**Explanation:** The array is already in non-decreasing order. Therefore, we return 0. 

**Constraints:**

- `1 <= nums.length <= 105`
- `1 <= nums[i] <= 109`

## 解题思路

- **贪心**
反向遍历数组，对于元素`i`，以`i + 1`为上界将元素`i`尽可能平均地做处理，产生的最小元素为新的上界。

## 复杂度

- 时间复杂度：$O(n)$
- 空间复杂度：$O(1)$

## 代码

```python
class Solution:
    def minimumReplacement(self, nums: List[int]) -> int:
        limit = nums[-1]
        n = len(nums)
        count = 0
        
        for i in range(n - 2, -1, -1):
            if nums[i] <= limit:
                limit = nums[i]
                continue
            ops = math.ceil(nums[i] / limit) - 1
            limit = nums[i] // (ops + 1)
            count += ops
        
        return count
```

---

## Minimum Number of Taps to Open to Water a Garden (1326)

## 难度

- **Hard**

## 问题描述

There is a one-dimensional garden on the x-axis. The garden starts at the point `0` and ends at the point `n`. (i.e The length of the garden is `n`).

There are `n + 1` taps located at points `[0, 1, ..., n]` in the garden.

Given an integer `n` and an integer array `ranges` of length `n + 1` where `ranges[i]` (0-indexed) means the `i-th` tap can water the area `[i - ranges[i], i + ranges[i]]` if it was open.

Return _the minimum number of taps_ that should be open to water the whole garden, If the garden cannot be watered return **-1**.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/01/16/1685_example_1.png)

**Input:** n = 5, ranges = [3,4,1,1,0,0]
**Output:** 1
**Explanation:** The tap at point 0 can cover the interval [-3,3]
The tap at point 1 can cover the interval [-3,5]
The tap at point 2 can cover the interval [1,3]
The tap at point 3 can cover the interval [2,4]
The tap at point 4 can cover the interval [4,4]
The tap at point 5 can cover the interval [5,5]
Opening Only the second tap will water the whole garden [0,5]

**Example 2:**

**Input:** n = 3, ranges = [0,0,0,0]
**Output:** -1
**Explanation:** Even if you activate all the four taps you cannot water the whole garden.

**Constraints:**

- `1 <= n <= 104`
- `ranges.length == n + 1`
- `0 <= ranges[i] <= 100`

## 解题思路

- **线性规划**
将所有的`interval`按照左侧排序并归并，之后利用线性规划求解，DP的状态为`(idx)`。

## 复杂度

假设n为元素个数，m为range
- 时间复杂度：$O(mn)$
- 空间复杂度：$O(n)$

## 代码

```python
class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        temp = [(max(0, i - ranges[i]), min(n, i + ranges[i])) for i in range(len(ranges))]
        temp.sort(key=lambda x: x[0])
        intervals = []
        
        for left, right in temp:
            if not intervals:
                intervals.append([left, right])
            else:
                if left == intervals[-1][0]:
                    intervals[-1][1] = max(intervals[-1][1], right)
                else:
                    intervals.append([left, right])
        
        @cache
        def dfs(idx):
            if intervals[idx][1] == n:
                return 1
            res = 10**10
            rightBound = intervals[idx][1]
            for i in range(idx + 1, len(intervals)):
                if intervals[i][0] <= rightBound:
                    res = min(res, 1 + dfs(i))
            return res

        return dfs(0) if dfs(0) < 10**10 else -1
```
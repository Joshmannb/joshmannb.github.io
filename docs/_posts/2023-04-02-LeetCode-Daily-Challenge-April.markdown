---
layout: post
title:  "LeetCode Daily Challenge April"
date:   2023-04-02 13:16:49 +0800
categories: LeetCode Algorithm
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
category_archive_path: "/categories/"
---

This post is for LeetCode Daily Challenge April

{% include toc %}

## Binary Search (704)

#### 问题描述

- Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with `O(log n)` runtime complexity.


##### Example 1:

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
```

##### Example 2:

```
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

##### Constraints:

- `1 <= nums.length <= 1e4`
- `-1e4 < nums[i], target < 1e4`
- All the integers in `nums` are unique.
- `nums` is sorted in ascending order.

#### 解题思路

- 该题考察基本二分查找算法的基本功，对于一个排序过的数组，只要每次都比对剩余范围内中间序号的值，即可将搜索范围减去 $$\frac{1}{2}$$。

#### 时间复杂度

- 二分法：$$O(\log n)$$

#### 代码

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if target < nums[mid]:
                right = mid
            elif target == nums[mid]:
                return mid
            else:
                left = mid + 1
        return left if nums[left] == target else -1
```

---

## Successful Pairs of Spells and Potions (2300)

#### 问题描述

- You are given two positive integer arrays `spells` and `potions`, of length `n` and `m` respectively, where `spells[i]` represents the strength of the $$i^{th}$$ spell and `potions[j]` represents the strength of the $$j^{th}$$ potion.

- You are also given an integer `success`. A spell and potion pair is considered successful if the product of their strengths is at least `success`.

- Return an integer array `pairs` of length `n` where `pairs[i]` is the number of potions that will form a successful pair with the $$i^{th}$$ spell.

##### Example 1:

```
Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
Output: [4,0,3]
Explanation:
- 0th spell: 5 * [1,2,3,4,5] = [5,10,15,20,25]. 4 pairs are successful.
- 1st spell: 1 * [1,2,3,4,5] = [1,2,3,4,5]. 0 pairs are successful.
- 2nd spell: 3 * [1,2,3,4,5] = [3,6,9,12,15]. 3 pairs are successful.
Thus, [4,0,3] is returned.
```

##### Example 2:

```
Input: spells = [3,1,2], potions = [8,5,8], success = 16
Output: [2,0,2]
Explanation:
- 0th spell: 3 * [8,5,8] = [24,15,24]. 2 pairs are successful.
- 1st spell: 1 * [8,5,8] = [8,5,8]. 0 pairs are successful. 
- 2nd spell: 2 * [8,5,8] = [16,10,16]. 2 pairs are successful. 
Thus, [2,0,2] is returned.
```

#### Constraints:

- `n == spells.length`
- `m == potions.length`
- `1 <= n, m <= 1e5`
- `1 <= spells[i], potions[i] <= 1e5`
- `1 <= success <= 1e10`

#### 解题思路

- 将potions按照降序排序，通过**二分查找**确定刚好满足每个`spells[i] * potions[j] >= success`的序号`j`。

#### 时间复杂度

- 排序`potions`: $$O(m\log{m})$$
- 找每个`spell`对应的序号：$$O(n\log m)$$

#### 代码

```python
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()

        res = []
        for spell in spells:
            left, right = 0, len(potions)
            while left < right:
                mid = (left + right) // 2
                if potions[mid] * spell < success:
                    left = mid + 1
                else:
                    right = mid
            res.append(len(potions) - left)
        return res
```

---

## Boats to Save People (881)

#### 问题描述

- You are given an array `people` where `people[i]` is the weight of the $$i^{th}$$ person, and an infinite number of boats where each boat can carry a maximum weight of `limit`. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most `limit`.

- Return the minimum number of boats to carry every given person.

 

##### Example 1:

```
Input: people = [1,2], limit = 3
Output: 1
Explanation: 1 boat (1, 2)
```

##### Example 2:

```
Input: people = [3,2,2,1], limit = 3
Output: 3
Explanation: 3 boats (1, 2), (2) and (3)
```

##### Example 3:

```
Input: people = [3,5,3,4], limit = 5
Output: 4
Explanation: 4 boats (3), (3), (4), (5)
```

##### Constraints:

- `1 <= people.length <= 5 * 1e4`
- `1 <= people[i] <= limit <= 3 * 1e4`

#### 解题思路

- **贪心算法**，**双指针**
  - 题目求解的是`最少所需的船`，换句话说则是对于每艘用于救援的船都尽可能`塞最多的人`（2人）
  - 将人按照`体重照降序排序`，同时初始化左右指针，左指针指向最重的人，右指针指向最轻的人。
  - 判断该船在`救援当前最重的人`的同时是否可以塞`当前最轻的人`上船，然后收缩左右指针，同时船数+1。

#### 时间复杂度

- 排序`people`：$$O(n\log n)$$
- 双指针：$$O(n)$$

#### 代码

```python
class Solution:
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort(reverse=True)
        left, right = 0, len(people) - 1

        res = 0
        while left <= right:
            res += 1
            if people[left] + people[right] <= limit:
                right -= 1
            left += 1
        return res
```

---

## Optimal Partition of String (2405)

#### 问题描述

- Given a string `s`, partition the string into one or more substrings such that the characters in each substring are unique. That is, no letter appears in a single substring more than once.

- Return the minimum number of substrings in such a partition.

- Note that each character should belong to exactly one substring in a partition. 

##### Example 1:

```
Input: s = "abacaba"
Output: 4
Explanation:
Two possible partitions are ("a","ba","cab","a") and ("ab","a","ca","ba").
It can be shown that 4 is the minimum number of substrings needed.
```

##### Example 2:

```
Input: s = "ssssss"
Output: 6
Explanation:
The only valid partition is ("s","s","s","s","s","s").
 ```

##### Constraints:

- `1 <= s.length <= 1e5`
- `s` consists of only English lowercase letters.

#### 解题思路

- 贪心算法
  - 题目求解的是`最少个数的不含重复字符的连续子字串`，这要求每个字子串都尽可能要长。
  - 初始化一个`set`来记录已经看过的字符，遍历输入的字符串，当遇到已经出现过的字符后将`set`清空并将结果+1。

#### 时间复杂度

- 遍历字符串: $$O(n)$$

#### 代码

```python
class Solution:
    def partitionString(self, s: str) -> int:
        vis = set()
        output = 1
        
        for idx, letter in enumerate(s):
            if letter in vis:
                output += 1
                vis.clear()
            vis.add(letter)
            
        return output
```

---

## Minimize Maximum of Array (2439)

#### 问题描述

- You are given a 0-indexed array `nums` comprising of `n` non-negative integers.

- In one operation, you must:
  - Choose an integer `i` such that `1 <= i < n` and `nums[i] > 0`.
  - Decrease `nums[i]` by 1.
  - Increase `nums[i - 1]` by 1.
- Return the minimum possible value of the maximum integer of `nums` after performing any number of operations.

##### Example 1:

```
Input: nums = [3,7,1,6]
Output: 5
Explanation:
One set of optimal operations is as follows:
1. Choose i = 1, and nums becomes [4,6,1,6].
2. Choose i = 3, and nums becomes [4,6,2,5].
3. Choose i = 1, and nums becomes [5,5,2,5].
The maximum integer of nums is 5. It can be shown that the maximum number cannot be less than 5.
Therefore, we return 5.
```

##### Example 2:

```
Input: nums = [10,1]
Output: 10
Explanation:
It is optimal to leave nums as is, and since 10 is the maximum value, we return 10.
```

##### Constraints:

- `n == nums.length`
- `2 <= n <= 1e5`
- `0 <= nums[i] <= 1e9`

#### 解题思路

- 贪心算法，前缀和
  - 对于比较简单的情况，即`nums[i]`可以跟两边的元素进行交换，那么题目的答案就是对整个数组取平均。
  - 而对于本题的情况，那么对于子数组`[nums[0], ..., nums[i]]`的平均则是`该子数组的下界` (不一定碰得到)。
  - 遍历子数组`[nums[0], ..., nums[i]]`，则答案是各子数组的下界中的`最大值`。
  - 通过前缀和加速子数组求和的计算。

#### 时间复杂度

- 遍历子数组：$$O(n)$$
- 前缀和：$$O(n)$$

#### 代码

```python
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        prefixSum = 0
        output = 0
        for i in range(len(nums)):
            prefixSum += nums[i]
            output = max(output, math.ceil(prefixSum / (i + 1)))
        return output
```

## Number of Closed Islands (1254)

#### 问题描述

- [LeetCode](https://leetcode.com/problems/number-of-closed-islands/description/)

#### 解题思路

- DFS
  - 循环遍历整个`grid`，且在当前为`land`时进行DFS
    - 将DFS触及到的`land`转为`water`，防止重复访问
    - 对`上下左右`的地方进行DFS，若碰到边界，则本次DFS所经过的land不是`island`
    - 若该次DFS结束也只能碰到water，则DFS过的地方为一块`island`，结果+1
    - 通过`cache`将已经访问过的状态作缓存，加速代码运行
- 注：在DFS进行过程中，要等DFS将所有路径都走过再return，否则会出错

#### 时间复杂度

- DFS: $$O(mn)$$

#### 代码

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        @cache
        def dfs(row, col):
            grid[row][col] = 1
            flag = False
            for i, j in dirs:
                newRow, newCol = row + i, col + j
                if not 0 <= newRow < m or not 0 <= newCol < n:
                    flag = True
                    continue
                if grid[newRow][newCol] == 1:
                    continue
                if dfs(newRow, newCol):
                    flag = True
            return flag

        output = 0
        for row in range(m):
            for col in range(n):
                if grid[row][col] == 0:
                    if dfs(row, col):
                        continue
                    else:
                        output += 1
        return output
```

## Number of Enclaves (1020)

#### 问题描述

- [LeetCode](https://leetcode.com/problems/number-of-enclaves/description/)

#### 解题思路

- DFS
  - 本题与`1254`极为相似，区别为`1254`求解的是不与边界接触的`island`数，而本题求解的是各个`island`面积之和。
  - 求解过程与`1254`形似，只要在DFS过程中记录走过的路径长度

#### 时间复杂度

- DFS: $$O(mn)$$

#### 代码

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        output = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        @cache
        def dfs(row, col):
            output = 1
            grid[row][col] = 0
            flag = False
            for i, j in dirs:
                newRow, newCol = row + i, col + j
                if not 0 <= newRow < m or not 0 <= newCol < n:
                    flag = True
                    continue
                if grid[newRow][newCol] == 0:
                    continue
                if (adj := dfs(newRow, newCol)) == 0:
                    flag = True
                output += adj
            return output if not flag else 0
        
        for row in range(m):
            for col in range(n):
                if grid[row][col] == 0:
                    continue
                output += dfs(row, col)
        return output
```
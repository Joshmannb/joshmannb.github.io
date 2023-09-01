---
layout: post
title:  "Template for posts"
date:   2023-04-03 13:16:49 +0800
categories: Template
hidden: true
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

This post is for LeetCode Daily Challenge April

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


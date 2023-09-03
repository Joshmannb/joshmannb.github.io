---
layout: post
title:  "[LeetCode] Biweekly Contest 112"
date:   2023-09-02 13:16:49 +0800
categories: LeetCode Contest
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

{% include toc %}

---

## Check if Strings Can be Made Equal With Operations I

#### 难度

- **Easy**

#### 问题描述

You are given two strings `s1` and `s2`, both of length `4`, consisting of **lowercase** English letters.

You can apply the following operation on any of the two strings **any** number of times:

- Choose any two indices `i` and `j` such that `j - i = 2`, then **swap** the two characters at those indices in the string.

Return `true` _if you can make the strings_ `s1` _and_ `s2` _equal, and_ `false` _otherwise_.

**Example 1:**

**Input:** s1 = "abcd", s2 = "cdab"  
**Output:** true  
**Explanation:** We can do the following operations on s1:  
- Choose the indices i = 0, j = 2. The resulting string is s1 = "cbad".
- Choose the indices i = 1, j = 3. The resulting string is s1 = "cdab" = s2.

**Example 2:**

**Input:** s1 = "abcd", s2 = "dacb"  
**Output:** false  
**Explanation:** It is not possible to make the two strings equal.  

**Constraints:**

- `s1.length == s2.length == 4`
- `s1` and `s2` consist only of lowercase English letters.

#### 解题思路

- **暴力解**  
检查所有的可能性。

#### 复杂度

- 时间复杂度：$$O(1)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        temp = [s1, s1[2] + s1[1] + s1[0] + s1[3], s1[0] + s1[3] + s1[2] + s1[1], s1[2] + s1[3] + s1[0] + s1[1]]
        
        for word in s1:
            if s2 == word:
                return True
        
        return False
```

---

## Check if Strings Can be Made Equal With Operations II

#### 难度

- **Medium**

#### 问题描述

You are given two strings `s1` and `s2`, both of length `n`, consisting of **lowercase** English letters.

You can apply the following operation on **any** of the two strings **any** number of times:

- Choose any two indices `i` and `j` such that `i < j` and the difference `j - i` is **even**, then **swap** the two characters at those indices in the string.

Return `true` _if you can make the strings_ `s1` _and_ `s2` _equal, and_ `false` _otherwise_.

**Example 1:**

**Input:** s1 = "abcdba", s2 = "cabdab"  
**Output:** true  
**Explanation:** We can apply the following operations on s1:  
- Choose the indices i = 0, j = 2. The resulting string is s1 = "cbadba".
- Choose the indices i = 2, j = 4. The resulting string is s1 = "cbbdaa".
- Choose the indices i = 1, j = 5. The resulting string is s1 = "cabdab" = s2.

**Example 2:**

**Input:** s1 = "abe", s2 = "bea"  
**Output:** false  
**Explanation:** It is not possible to make the two strings equal.  

**Constraints:**

- `n == s1.length == s2.length`
- `1 <= n <= 105`
- `s1` and `s2` consist only of lowercase English letters.

#### 解题思路

- **哈希表**  
对于`s1`与`s2`，分别检查奇数位数与偶数位数所构成的数组是否含有相同的元素集合。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def checkStrings(self, s1: str, s2: str) -> bool:
        odd1 = defaultdict(lambda: 0)
        even1 = defaultdict(lambda: 0)
        odd2 = defaultdict(lambda: 0)
        even2 = defaultdict(lambda: 0)
        
        for i in range(len(s1)):
            if i % 2 == 0:
                even1[s1[i]] += 1
                even2[s2[i]] += 1
            else:
                odd1[s1[i]] += 1
                odd2[s2[i]] += 1
        
        for key in odd2.keys():
            if odd2[key] != odd1[key]:
                return False
        
        for key in even2.keys():
            if even2[key] != even1[key]:
                return False
        
        return True
```

---

## Maximum Sum of Almost Unique Subarray

#### 难度

- **Medium**

#### 问题描述

You are given an integer array `nums` and two positive integers `m` and `k`.

Return _the **maximum sum** out of all **almost unique** subarrays of length_ `k` _of_ `nums`. If no such subarray exists, return `0`.

A subarray of `nums` is **almost unique** if it contains at least `m` distinct elements.

A subarray is a contiguous **non-empty** sequence of elements within an array.

**Example 1:**

**Input:** nums = [2,6,7,3,1,7], m = 3, k = 4  
**Output:** 18  
**Explanation:** There are 3 almost unique subarrays of size `k = 4`. These subarrays are [2, 6, 7, 3], [6, 7, 3, 1], and [7, 3, 1, 7]. Among these subarrays, the one with the maximum sum is [2, 6, 7, 3] which has a sum of 18.  

**Example 2:**

**Input:** nums = [5,9,9,2,4,5,4], m = 1, k = 3  
**Output:** 23  
**Explanation:** There are 5 almost unique subarrays of size k. These subarrays are [5, 9, 9], [9, 9, 2], [9, 2, 4], [2, 4, 5], and [4, 5, 4]. Among these subarrays, the one with the maximum sum is [5, 9, 9] which has a sum of 23.  

**Example 3:**

**Input:** nums = [1,2,1,2,1,2,1], m = 3, k = 3  
**Output:** 0  
**Explanation:** There are no subarrays of size `k = 3` that contain at least `m = 3` distinct elements in the given array [1,2,1,2,1,2,1]. Therefore, no almost unique subarrays exist, and the maximum sum is 0.  

**Constraints:**

- `1 <= nums.length <= 2 * 104`
- `1 <= m <= k <= nums.length`
- `1 <= nums[i] <= 109`

#### 解题思路

- **双指针**  
用双指针维护一个长度为`k`的区间，若区间的独特元素数大于或等于`m`，则尝试对答案进行更新。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        sumVal = 0
        track = defaultdict(lambda: 0)
        dq = deque()
        res = 0
        
        for right in range(len(nums)):
            dq.append(nums[right])
            sumVal += nums[right]
            track[nums[right]] += 1
            if len(dq) > k:
                val = dq.popleft()
                sumVal -= val
                track[val] -= 1
                if track[val] == 0:
                    del track[val]
            if len(dq) == k and len(track.keys()) >= m:
                res = max(res, sumVal)
        
        return res
```

---

## Count K-Subsequences of a String With Maximum Beauty

#### 难度

- **Hard**

#### 问题描述

You are given a string `s` and an integer `k`.

A **k-subsequence** is a **subsequence** of `s`, having length `k`, and all its characters are **unique**, **i.e**., every character occurs once.

Let `f(c)` denote the number of times the character `c` occurs in `s`.

The **beauty** of a **k-subsequence** is the **sum** of `f(c)` for every character `c` in the k-subsequence.

For example, consider `s = "abbbdd"` and `k = 2`:

- `f('a') = 1`, `f('b') = 3`, `f('d') = 2`
- Some k-subsequences of `s` are:
    - `"**ab**bbdd"` -> `"ab"` having a beauty of `f('a') + f('b') = 4`
    - `"**a**bbb**d**d"` -> `"ad"` having a beauty of `f('a') + f('d') = 3`
    - `"a**b**bb**d**d"` -> `"bd"` having a beauty of `f('b') + f('d') = 5`

Return _an integer denoting the number of k-subsequences_ _whose **beauty** is the **maximum** among all **k-subsequences**_. Since the answer may be too large, return it modulo `109 + 7`.

A subsequence of a string is a new string formed from the original string by deleting some (possibly none) of the characters without disturbing the relative positions of the remaining characters.

**Notes**

- `f(c)` is the number of times a character `c` occurs in `s`, not a k-subsequence.
- Two k-subsequences are considered different if one is formed by an index that is not present in the other. So, two k-subsequences may form the same string.

**Example 1:**

**Input:** s = "bcca", k = 2  
**Output:** 4  
**Explanation:** From s we have f('a') = 1, f('b') = 1, and f('c') = 2.  
The k-subsequences of s are:   
**bc**ca having a beauty of f('b') + f('c') = 3   
**b**c**c**a having a beauty of f('b') + f('c') = 3   
**b**cc**a** having a beauty of f('b') + f('a') = 2   
b**c**c**a** having a beauty of f('c') + f('a') = 3  
bc**ca** having a beauty of f('c') + f('a') = 3   
There are 4 k-subsequences that have the maximum beauty, 3.   
Hence, the answer is 4.   

**Example 2:**

**Input:** s = "abbcd", k = 4  
**Output:** 2  
**Explanation:** From s we have f('a') = 1, f('b') = 2, f('c') = 1, and f('d') = 1.   
The k-subsequences of s are:   
**ab**b**cd** having a beauty of f('a') + f('b') + f('c') + f('d') = 5  
**a**b**bcd** having a beauty of f('a') + f('b') + f('c') + f('d') = 5   
There are 2 k-subsequences that have the maximum beauty, 5.   
Hence, the answer is 2.   

**Constraints:**

- `1 <= s.length <= 2 * 105`
- `1 <= k <= s.length`
- `s` consists only of lowercase English letters.

#### 解题思路

- **数学**  
1. 计算各个字母出现的次数，并计算最大的`beauty`以及构成时所用的最小频率。
2. 对于每个字母，若频率大于最小频率则直接乘在答案里。
3. 若频率等于最小频率，则从中用组合的方法挑出可能的组合乘在答案里。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        counter = Counter(s)
        
        if len(counter.keys()) < k:
            return 0
        
        freq = sorted(counter.values(), reverse=True)
        minimum = freq[k - 1]
        must = 1
        canddidates = 0
        
        for letter, occurance in counter.items():
            if occurance < minimum:
                continue
            elif occurance > minimum:
                k -= 1
                must *= occurance
            elif occurance == minimum:
                canddidates += 1
        
        return must * math.comb(canddidates, k) * minimum**k % MOD
```
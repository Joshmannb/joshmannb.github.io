---
layout: post
title:  "[LeetCode] Weekly Contest 356"
date:   2023-07-30 13:16:49 +0800
categories: LeetCode Contest
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

{% include toc %}

---

## Number of Employees Who Met the Target (2798)

#### 难度

- **Easy**

#### 问题描述

There are `n` employees in a company, numbered from `0` to `n - 1`. Each employee `i` has worked for `hours[i]` hours in the company.

The company requires each employee to work for **at least** `target` hours.

You are given a **0-indexed** array of non-negative integers `hours` of length `n` and a non-negative integer `target`.

Return _the integer denoting the number of employees who worked at least_ `target` _hours_.

**Example 1:**

**Input:** hours = [0,1,2,3,4], target = 2  
**Output:** 3  
**Explanation:** The company wants each employee to work for at least 2 hours.  
- Employee 0 worked for 0 hours and didn't meet the target.  
- Employee 1 worked for 1 hours and didn't meet the target.  
- Employee 2 worked for 2 hours and met the target.  
- Employee 3 worked for 3 hours and met the target.
- Employee 4 worked for 4 hours and met the target.
There are 3 employees who met the target.

**Example 2:**

**Input:** hours = [5,1,4,2,2], target = 6  
**Output:** 0  
**Explanation:** The company wants each employee to work for at least 6 hours.  
There are 0 employees who met the target.

**Constraints:**

- `1 <= n == hours.length <= 50`
- `0 <= hours[i], target <= 105`

#### 解题思路

- 遍历数组，若元素符合条件则结果+1。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return sum(1 for i in range(len(hours)) if hours[i] >= target)
```

---

## Count Complete Subarrays in an Array (2799)

#### 难度

- **Medium**

#### 问题描述

You are given an array `nums` consisting of **positive** integers.

We call a subarray of an array **complete** if the following condition is satisfied:

- The number of **distinct** elements in the subarray is equal to the number of distinct elements in the whole array.

Return _the number of **complete** subarrays_.

A **subarray** is a contiguous non-empty part of an array.

**Example 1:**

**Input:** nums = [1,3,1,2,2]  
**Output:** 4  
**Explanation:** The complete subarrays are the following: [1,3,1,2], [1,3,1,2,2], [3,1,2] and [3,1,2,2].  

**Example 2:**

**Input:** nums = [5,5,5,5]  
**Output:** 10  
**Explanation:** The array consists only of the integer 5, so any subarray is complete. The number of subarrays that we can choose is 10.  

**Constraints:**

- `1 <= nums.length <= 1000`
- `1 <= nums[i] <= 2000`

#### 解题思路

- 利用**双指针**计算对于每个固定右边界的窗口有几个窗口满足条件，并加总。

#### 复杂度

- 时间复杂度：$$O(n^2)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        target = len(set(nums))
        left = 0
        res = 0
        
        for right in range(len(nums)):
            track = Counter(nums[:right + 1])
            if len(track) == target:
                temp = left
                while len(track) == target and temp <= right:
                    track[nums[temp]] -= 1
                    if track[nums[temp]] == 0:
                        del track[nums[temp]]
                    temp += 1
                res += temp - 1 - left + 1
        
        return res
```

---

## Shortest String That Contains Three Strings (2800)

#### 难度

- **Medium**

#### 问题描述

Given three strings `a`, `b`, and `c`, your task is to find a string that has the **minimum** length and contains all three strings as **substrings**.

If there are multiple such strings, return the **lexicographically smallest** one.

Return _a string denoting the answer to the problem._

**Notes**

- A string `a` is **lexicographically smaller** than a string `b` (of the same length) if in the first position where `a` and `b` differ, string `a` has a letter that appears **earlier** in the alphabet than the corresponding letter in `b`.
- A **substring** is a contiguous sequence of characters within a string.

**Example 1:**

**Input:** a = "abc", b = "bca", c = "aaa"  
**Output:** "aaabca"  
**Explanation:**  We show that "aaabca" contains all the given strings: a = ans[2...4], b = ans[3..5], c = ans[0..2]. It can be shown that the length of the resulting string would be at least 6 and "aaabca" is the lexicographically smallest one.  

**Example 2:**

**Input:** a = "ab", b = "ba", c = "aba"  
**Output:** "aba"  
**Explanation:** We show that the string "aba" contains all the given strings: a = ans[0..1], b = ans[1..2], c = ans[0..2]. Since the length of c is 3, the length of the resulting string would be at least 3. It can be shown that "aba" is the lexicographically smallest one.  

**Constraints:**

- `1 <= a.length, b.length, c.length <= 100`
- `a`, `b`, `c` consist only of lowercase English letters.

#### 解题思路

- **模拟**  
该题求三个字符串的各种排列衔接后`最短字符串`的长度。模拟各个情况并返回答案。

#### 复杂度

- 时间复杂度：$$O(n)$$,n为字符串长度。
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def concatenate(firstString, secondString):
            length = min(len(firstString), len(secondString))
            res = 0
            if firstString in secondString:
                return secondString
            if secondString in firstString:
                return firstString
            for i in range(length):
                word = secondString[:i + 1]
                if word == firstString[-i - 1:]:
                    res = i + 1
                    continue
            return firstString + secondString[res:]
        
        words = [a, b, c]
        res = []
            
        for i in range(3):
            leftWord, midWord, rightWord = words[(i + 1) % 3], words[i], words[(i + 2) % 3]
            temp = concatenate(concatenate(leftWord, midWord), rightWord)
            temp2 = concatenate(leftWord, concatenate(midWord, rightWord))
            res.append(temp)
            res.append(temp2)
            leftWord, midWord, rightWord = words[(i + 2) % 3], words[i], words[(i + 1) % 3]
            temp = concatenate(concatenate(leftWord, midWord), rightWord)
            temp2 = concatenate(leftWord, concatenate(midWord, rightWord))
            res.append(temp2)
            res.append(temp)

        res.sort(key=lambda x: (len(x), x))
        return res[0]
```

---

## Count Stepping Numbers in Range (2801)

#### 难度

- **Hard**

#### 问题描述

Given two positive integers `low` and `high` represented as strings, find the count of **stepping numbers** in the inclusive range `[low, high]`.

A **stepping number** is an integer such that all of its adjacent digits have an absolute difference of **exactly** `1`.

Return _an integer denoting the count of stepping numbers in the inclusive range_ `[low, high]`_._

Since the answer may be very large, return it **modulo** `109 + 7`.

**Note:** A stepping number should not have a leading zero.

**Example 1:**

**Input:** low = "1", high = "11"  
**Output:** 10  
**Explanation:** The stepping numbers in the range [1,11] are 1, 2, 3, 4, 5, 6, 7, 8, 9 and 10. There are a total of 10 stepping numbers in the range. Hence, the output is 10.  

**Example 2:**

**Input:** low = "90", high = "101"  
**Output:** 2  
**Explanation:** The stepping numbers in the range [90,101] are 98 and 101. There are a total of 2 stepping numbers in the range. Hence, the output is 2.   

**Constraints:**

- `1 <= int(low) <= int(high) < 10100`
- `1 <= low.length, high.length <= 100`
- `low` and `high` consist of only digits.
- `low` and `high` don't have any leading zeros.

#### 解题思路

- 经典**数位DP**题目。  
DP状态为`[digitIdx, digitVal, greaterThanTarget, smallerThanTarget]`

#### 复杂度

假设$n$为上界的字母数。
- 时间复杂度：$$O(10\times{3}\times{n}) = O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7
        
        def dp(num):
            if num == '0':
                return 0
            res = 0
            firstDigit = int(num[0])
            for i in range(1, 10):
                res += dfs(0, i, num, i > firstDigit, i < firstDigit) % MOD
            return res % MOD
            
        @cache
        def dfs(idx, digit, target, greater, smaller):
            if digit < 0 or digit > 9:
                return 0
            if not greater and not smaller:
                greater = digit > int(target[idx])
                smaller = digit < int(target[idx])
            if idx == len(target) - 1:
                return 0 if greater else 1
            res = 1
            for i in range(-1, 2, 2):
                res += dfs(idx + 1, digit + i, target, greater, smaller) % MOD
            return res % MOD
        
        return (dp(high) - dp(str(int(low) - 1)) + MOD) % MOD
```
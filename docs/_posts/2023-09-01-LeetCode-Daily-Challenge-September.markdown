---
layout: post
title:  "LeetCode Daily Challenge September"
date:   2023-09-01 13:16:49 +0800
categories: LeetCode Algorithm
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

This post is for LeetCode Daily Challenge August.

{% include toc %}

## Counting Bits (338)

#### 难度

- **Easy**

#### 问题描述

Given an integer `n`, return _an array_ `ans` _of length_ `n + 1` _such that for each_ `i` (`0 <= i <= n`)_,_ `ans[i]` _is the **number of**_ `1`_**'s** in the binary representation of_ `i`.

**Example 1:**

**Input:** n = 2
**Output:** [0,1,1]
**Explanation:**
0 --> 0
1 --> 1
2 --> 10

**Example 2:**

**Input:** n = 5
**Output:** [0,1,1,2,1,2]
**Explanation:**
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101

**Constraints:**

- `0 <= n <= 105`

#### 解题思路

- **线性规划**
对于元素`i`，它所含有的1的个数为`res[i // 2] + i % 2`。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```java
class Solution {
    public int[] countBits(int n) {
        int[] res = new int[n + 1];
        
        for (int i = 0; i < res.length; i++) {
            res[i] = res[i >> 1] + (i & 1);
        }
        
        return res;
    }
}
```

--- 

## Extra Characters in a String (2707)

#### 难度

- **Medium**

#### 问题描述

You are given a **0-indexed** string `s` and a dictionary of words `dictionary`. You have to break `s` into one or more **non-overlapping** substrings such that each substring is present in `dictionary`. There may be some **extra characters** in `s` which are not present in any of the substrings.

Return _the **minimum** number of extra characters left over if you break up_ `s` _optimally._

**Example 1:**

**Input:** s = "leetscode", dictionary = ["leet","code","leetcode"]
**Output:** 1
**Explanation:** We can break s in two substrings: "leet" from index 0 to 3 and "code" from index 5 to 8. There is only 1 unused character (at index 4), so we return 1.

**Example 2:**

**Input:** s = "sayhelloworld", dictionary = ["hello","world"]
**Output:** 3
**Explanation:** We can break s in two substrings: "hello" from index 3 to 7 and "world" from index 8 to 12. The characters at indices 0, 1, 2 are not used in any substring and thus are considered as extra characters. Hence, we return 3.

**Constraints:**

- `1 <= s.length <= 50`
- `1 <= dictionary.length <= 50`
- `1 <= dictionary[i].length <= 50`
- `dictionary[i]` and `s` consists of only lowercase English letters
- `dictionary` contains distinct words

#### 解题思路

- **线性规划**
DP的状态为`idx`，检查从各个idx出发可以利用的字母个数，答案为`s.length() - dp(0)`.

#### 复杂度

假设n为`s`的长度，m为`dictionary`的元素个数，k为最长单词的长度。
- 时间复杂度：$$O(nmk)$$
- 空间复杂度：$$O(n)$$

#### 代码

```java
class Solution {
    public int minExtraChar(String s, String[] dictionary) {
        Map<Integer, Integer> memo = new HashMap<>();
        int res = dp(0, s, dictionary, memo);
        return s.length() - res;
    }
    
    public int dp(int idx, String s, String[] dictionary, Map<Integer, Integer> memo) {
        if (memo.containsKey(idx)) { return memo.get(idx); }
        if (idx == s.length()) { return 0; }
        int res = 0;
        for (String word : dictionary) {
            if (idx + word.length() - 1 >= s.length() || s.charAt(idx) != word.charAt(0)) { continue; }
            if (s.substring(idx, idx + word.length()).equals(word)) {
                res = Math.max(res, word.length() + dp(idx + word.length(), s, dictionary, memo));
            }
        }
        res = Math.max(res, dp(idx + 1, s, dictionary, memo));
        memo.put(idx, res);
        return res;
    }
}
```
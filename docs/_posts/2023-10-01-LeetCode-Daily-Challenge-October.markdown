---
layout: post
title:  "[LeetCode] Daily Challenge October"
date:   2023-10-01 13:16:49 +0800
categories: LeetCode Algorithm
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

This post is for LeetCode Daily Challenge October.

{% include toc %}

## Reverse Words in a String III (557)

#### 难度

- **Easy**

#### 问题描述

Given a string `s`, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

**Example 1:**

**Input:** s = "Let's take LeetCode contest"  
**Output:** "s'teL ekat edoCteeL tsetnoc"  

**Example 2:**

**Input:** s = "God Ding"  
**Output:** "doG gniD"  

**Constraints:**

- `1 <= s.length <= 5 * 104`
- `s` contains printable **ASCII** characters.
- `s` does not contain any leading or trailing spaces.
- There is **at least one** word in `s`.
- All the words in `s` are separated by a single space.

#### 解题思路

- **字符串**  
将字符串分离成一个个单词后，将每个单词做倒序处理，然后接成一个字符串。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```java
class Solution {
    public String reverseWords(String s) {
        String[] words = s.split(" ");
        StringBuilder temp = new StringBuilder();
        
        for (int i = 0; i < words.length; i++) {
            temp.delete(0, temp.length());
            temp.append(words[i]);
            temp.reverse();
            words[i] = temp.toString();
        }
        
        return String.join(" ", words);
    }
}
```

---

## Remove Colored Pieces if Both Neighbors are the Same Color (2038)

#### 难度

- **Medium**

#### 问题描述

There are `n` pieces arranged in a line, and each piece is colored either by `'A'` or by `'B'`. You are given a string `colors` of length `n` where `colors[i]` is the color of the `ith` piece.

Alice and Bob are playing a game where they take **alternating turns** removing pieces from the line. In this game, Alice moves **first**.

- Alice is only allowed to remove a piece colored `'A'` if **both its neighbors** are also colored `'A'`. She is **not allowed** to remove pieces that are colored `'B'`.
- Bob is only allowed to remove a piece colored `'B'` if **both its neighbors** are also colored `'B'`. He is **not allowed** to remove pieces that are colored `'A'`.
- Alice and Bob **cannot** remove pieces from the edge of the line.
- If a player cannot make a move on their turn, that player **loses** and the other player **wins**.

Assuming Alice and Bob play optimally, return `true` _if Alice wins, or return_ `false` _if Bob wins_.

**Example 1:**

**Input:** colors = "AAABABB"  
**Output:** true  
**Explanation:**  
AAABABB -> AABABB  
Alice moves first.  
She removes the second 'A' from the left since that is the only 'A' whose neighbors are both 'A'.  

Now it's Bob's turn.  
Bob cannot make a move on his turn since there are no 'B's whose neighbors are both 'B'.  
Thus, Alice wins, so return true.  

**Example 2:**

**Input:** colors = "AA"  
**Output:** false  
**Explanation:**  
Alice has her turn first.  
There are only two 'A's and both are on the edge of the line, so she cannot move on her turn.  
Thus, Bob wins, so return false.  

**Example 3:**

**Input:** colors = "ABBBBBBBAAA"  
**Output:** false  
**Explanation:**  
ABBBBBBBAAA -> ABBBBBBBAA  
Alice moves first.  
Her only option is to remove the second to last 'A' from the right.  

ABBBBBBBAA -> ABBBBBBAA  
Next is Bob's turn.  
He has many options for which 'B' piece to remove. He can pick any.  

On Alice's second turn, she has no more pieces that she can remove.  
Thus, Bob wins, so return false.  

**Constraints:**

- `1 <= colors.length <= 105`
- `colors` consists of only the letters `'A'` and `'B'`

#### 解题思路

- **贪心**  
在这个游戏中，一方的操作不会对另一方之后的操作产生任何影响，所以只要记录`Alice`可以操作的次数是否大于`Bob`。遍历字符串，记录两方可以进行操作的次数。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        AliceCount = 0
        BobCount = 0
        
        for i in range(1, len(colors) - 1):
            if colors[i] == 'A' and colors[i - 1] == 'A' and colors[i + 1] == 'A':
                AliceCount += 1
            if colors[i] == 'B' and colors[i - 1] == 'B' and colors[i + 1] == 'B':
                BobCount += 1
        
        return AliceCount >= BobCount + 1
```

---

## Number of Good Pairs (1512)

#### 难度

- **Easy**

#### 问题描述

Given an array of integers `nums`, return _the number of **good pairs**_.

A pair `(i, j)` is called _good_ if `nums[i] == nums[j]` and `i` < `j`.

**Example 1:**

**Input:** nums = [1,2,3,1,1,3]  
**Output:** 4  
**Explanation:** There are 4 good pairs (0,3), (0,4), (3,4), (2,5) 0-indexed.  

**Example 2:**

**Input:** nums = [1,1,1,1]  
**Output:** 6  
**Explanation:** Each pair in the array are _good_.  

**Example 3:**

**Input:** nums = [1,2,3]  
**Output:** 0  

**Constraints:**

- `1 <= nums.length <= 100`
- `1 <= nums[i] <= 100`

#### 解题思路

- **计数**  
遍历数组，记录到当前为止每个元素所出现的次数。对于当前元素`x`，`good pairs`次数增加`freq[x]`.

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        freq = defaultdict(lambda: 0)
        count = 0
        
        for num in nums:
            count += freq[num]
            freq[num] += 1
        
        return count
```
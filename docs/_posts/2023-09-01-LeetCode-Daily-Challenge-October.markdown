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
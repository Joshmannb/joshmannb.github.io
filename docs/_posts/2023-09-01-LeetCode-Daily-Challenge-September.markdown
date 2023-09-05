---
layout: post
title:  "[LeetCode] Daily Challenge September"
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

---

## Unique Paths (62)

#### 难度

- **Medium**

#### 问题描述

There is a robot on an `m x n` grid. The robot is initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either down or right at any point in time.

Given the two integers `m` and `n`, return _the number of possible unique paths that the robot can take to reach the bottom-right corner_.

The test cases are generated so that the answer will be less than or equal to `2 * 109`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)

**Input:** m = 3, n = 7  
**Output:** 28

**Example 2:**

**Input:** m = 3, n = 2  
**Output:** 3  
**Explanation:** From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down

**Constraints:**

- `1 <= m, n <= 100`

#### 解题思路

- **线性规划**  
DP的状态为`(x, y)`，转移函数为`DP(x, y) = max(DP(x, y), DP(x - 1, y) + DP(x, y - 1))`。

#### 复杂度

- 时间复杂度：$$O(mn)$$
- 空间复杂度：$$O(mn)$$

#### 代码

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[1][1] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = max(dp[i][j], dp[i - 1][j] + dp[i][j - 1])
        
        return dp[m][n]
```

---

## Linked List Cycle (141)

#### 难度

- **Easy**

#### 问题描述

Given `head`, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the `next` pointer. Internally, `pos` is used to denote the index of the node that tail's `next` pointer is connected to. **Note that `pos` is not passed as a parameter**.

Return `true` _if there is a cycle in the linked list_. Otherwise, return `false`.

**Example 1:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist.png)

**Input:** head = [3,2,0,-4], pos = 1  
**Output:** true  
**Explanation:** There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).  

**Example 2:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test2.png)

**Input:** head = [1,2], pos = 0  
**Output:** true  
**Explanation:** There is a cycle in the linked list, where the tail connects to the 0th node.  

**Example 3:**

![](https://assets.leetcode.com/uploads/2018/12/07/circularlinkedlist_test3.png)

**Input:** head = [1], pos = -1  
**Output:** false  
**Explanation:** There is no cycle in the linked list.  

**Constraints:**

- The number of the nodes in the list is in the range `[0, 104]`.
- `-105 <= Node.val <= 105`
- `pos` is `-1` or a **valid index** in the linked-list.

**Follow up:** Can you solve it using `O(1)` (i.e. constant) memory?

#### 解题思路

- **Floyd判圈算法**  
维护一个慢指针和一个快指针，若两个指针会重叠，则有圈，否则无圈。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

---

## Copy List with Random Pointer (138)

#### 难度

- **Medium**

#### 问题描述

A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`.

Construct a [**deep copy**](https://en.wikipedia.org/wiki/Object_copying#Deep_copy) of the list. The deep copy should consist of exactly `n` **brand new** nodes, where each new node has its value set to the value of its corresponding original node. Both the `next` and `random` pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. **None of the pointers in the new list should point to nodes in the original list**.

For example, if there are two nodes `X` and `Y` in the original list, where `X.random --> Y`, then for the corresponding two nodes `x` and `y` in the copied list, `x.random --> y`.

Return _the head of the copied linked list_.

The linked list is represented in the input/output as a list of `n` nodes. Each node is represented as a pair of `[val, random_index]` where:

- `val`: an integer representing `Node.val`
- `random_index`: the index of the node (range from `0` to `n-1`) that the `random` pointer points to, or `null` if it does not point to any node.

Your code will **only** be given the `head` of the original linked list.

**Example 1:**

![](https://assets.leetcode.com/uploads/2019/12/18/e1.png)

**Input:** head = [[7,null],[13,0],[11,4],[10,2],[1,0]]  
**Output:** [[7,null],[13,0],[11,4],[10,2],[1,0]]  

**Example 2:**

![](https://assets.leetcode.com/uploads/2019/12/18/e2.png)

**Input:** head = [[1,1],[2,1]]  
**Output:** [[1,1],[2,1]]  

**Example 3:**

**![](https://assets.leetcode.com/uploads/2019/12/18/e3.png)**

**Input:** head = [[3,null],[3,0],[3,null]]  
**Output:** [[3,null],[3,0],[3,null]]  

**Constraints:**

- `0 <= n <= 1000`
- `-104 <= Node.val <= 104`
- `Node.random` is `null` or is pointing to some node in the linked list.

#### 解题思路

- **哈希表**，**链表**  
对原链表进行两次遍历，第一次遍历时构建新的链表，同时利用哈希表将旧链表中的节点与新链表的节点对应起来。第二次遍历时将`random`加入到新链表中，并利用哈希表找到所要连结的新节点。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if not head:
            return None
        
        now = head
        newHead = Node(head.val)
        newNow = newHead
        track = dict()
        track[head] = newHead

        while now and now.next:
            now = now.next
            newNow.next = Node(now.val)
            newNow = newNow.next
            track[now] = newNow
        
        now = head
        newNow = newHead
        
        while now:
            newNow.random = track[now.random] if now.random != None else None
            now = now.next
            newNow = newNow.next
        
        return newHead
```
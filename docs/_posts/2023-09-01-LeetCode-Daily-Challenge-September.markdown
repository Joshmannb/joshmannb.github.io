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

This post is for LeetCode Daily Challenge September.

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

---

## Split Linked List in Parts (725)

#### 难度

- **Medium**

#### 问题描述

Given the `head` of a singly linked list and an integer `k`, split the linked list into `k` consecutive linked list parts.

The length of each part should be as equal as possible: no two parts should have a size differing by more than one. This may lead to some parts being null.

The parts should be in the order of occurrence in the input list, and parts occurring earlier should always have a size greater than or equal to parts occurring later.

Return _an array of the_ `k` _parts_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/06/13/split1-lc.jpg)

**Input:** head = [1,2,3], k = 5  
**Output:** [[1],[2],[3],[],[]]  
**Explanation:**  
The first element output[0] has output[0].val = 1, output[0].next = null.  
The last element output[4] is null, but its string representation as a ListNode is [].

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/06/13/split2-lc.jpg)

**Input:** head = [1,2,3,4,5,6,7,8,9,10], k = 3  
**Output:** [[1,2,3,4],[5,6,7],[8,9,10]]  
**Explanation:**  
The input has been split into consecutive parts with size difference at most 1, and earlier parts are a larger size than the later parts.  

**Constraints:**

- The number of nodes in the list is in the range `[0, 1000]`.
- `0 <= Node.val <= 1000`
- `1 <= k <= 50`

#### 解题思路

- **双指针**  
首先将所有的节点存在一个数组中。维护一个双指针，双指针的长度为一个子链表的长度，将链表尾节点指向`null`，并将头节点加入答案数组中。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        temp = []
        now = head
        oldK = k
        
        while now:
            temp.append(now)
            now = now.next
        
        length = len(temp)
        left = 0
        res = []
        
        while length > 0:
            right = left + math.ceil(length / k) - 1
            length -= right - left + 1
            k -= 1
            temp[right].next = None
            res.append(temp[left])
            left = right + 1
        
        while len(res) < oldK:
            res.append(None)
        
        return res
```

---

## Reverse Linked List II (92)

#### 难度

- **Medium**

#### 问题描述

Given the `head` of a singly linked list and two integers `left` and `right` where `left <= right`, reverse the nodes of the list from position `left` to position `right`, and return _the reversed list_.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/02/19/rev2ex2.jpg)

**Input:** head = [1,2,3,4,5], left = 2, right = 4  
**Output:** [1,4,3,2,5]  

**Example 2:**

**Input:** head = [5], left = 1, right = 1  
**Output:** [5]  

**Constraints:**

- The number of nodes in the list is `n`.
- `1 <= n <= 500`
- `-500 <= Node.val <= 500`
- `1 <= left <= right <= n`

**Follow up:** Could you do it in one pass?

#### 解题思路

- **链表**  
遍历链表并记录当前的索引。当进入需要反转的区域时，记录`该区域前的节点`与`该区域的第一个节点`，反转该区域后，将`该区域前的节点`接到`该区域最后一个节点`，将`该区域的第一个节点`接到`该区域之后的第一个节点`。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if left == right:
            return head
        
        prehead = ListNode(val=0, next=head)
        now = prehead
        idx = 0
        
        while idx < left - 1:
            now = now.next
            idx += 1
        
        prev = now
        beforeNode = now
        now = now.next
        lastNode = now
        idx += 1
        
        while idx <= right:
            newNow = now.next
            if idx != left:
                now.next = prev
            prev = now
            now = newNow
            idx += 1
        
        beforeNode.next = prev
        lastNode.next = now

        return prehead.next
```

---

## Pascal's Triangle (118)

#### 难度

- **Easy**

#### 问题描述

Given an integer `numRows`, return the first numRows of **Pascal's triangle**.

In **Pascal's triangle**, each number is the sum of the two numbers directly above it as shown:

![](https://upload.wikimedia.org/wikipedia/commons/0/0d/PascalTriangleAnimated2.gif)

**Example 1:**

**Input:** numRows = 5  
**Output:** [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]  

**Example 2:**

**Input:** numRows = 1  
**Output:** [[1]]  

**Constraints:**

- `1 <= numRows <= 30`

#### 解题思路

- **线性规划**  
第`n`层的元素根据第`n-1`层的元素来构建。

#### 复杂度

- 时间复杂度：$$O(n^2)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        output = []
        for i in range(1, numRows + 1):
            output.append([int(j in {0, i - 1}) for j in range(i)])
        for i in range(1, numRows):
            for j in range(1, len(output[i]) - 1):
                output[i][j] = output[i-1][j] + output[i-1][j-1]
        return output
```

---

## Combination Sum IV (377)

#### 难度

- **Medium**

#### 问题描述

Given an array of **distinct** integers `nums` and a target integer `target`, return _the number of possible combinations that add up to_ `target`.

The test cases are generated so that the answer can fit in a **32-bit** integer.

**Example 1:**

**Input:** nums = [1,2,3], target = 4  
**Output:** 7  
**Explanation:**  
The possible combination ways are:  
(1, 1, 1, 1)  
(1, 1, 2)  
(1, 2, 1)  
(1, 3)  
(2, 1, 1)  
(2, 2)  
(3, 1)  
Note that different sequences are counted as different combinations.  

**Example 2:**

**Input:** nums = [9], target = 3  
**Output:** 0  

**Constraints:**

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 1000`
- All the elements of `nums` are **unique**.
- `1 <= target <= 1000`

**Follow up:** What if negative numbers are allowed in the given array? How does it change the problem? What limitation we need to add to the question to allow negative numbers?

#### 解题思路

- **线性规划**  
dp的状态为`(val)`，转移函数为`dp_val = dp_val + dp_(val - num)`。

#### 复杂度

- 时间复杂度：$$O(mn)$$
- 空间复杂度：$$O(mn)$$

#### 代码

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        Map<Integer, Integer> memo = new HashMap<>();        
        
        return dfs(0, nums, target, memo);
    }
    
    int dfs(int sumNow, int[] nums, int target, Map<Integer, Integer> memo) {
        if (sumNow == target) { return 1; }
        if (sumNow > target) { return 0; }
        if (memo.containsKey(sumNow)) { return memo.get(sumNow); }
        int res = 0;
        for (var num : nums) {
            res += dfs(sumNow + num, nums, target, memo);
        }
        memo.put(sumNow, res);
        return res;
    }
}
```

---

## Count All Valid Pickup and Delivery Options (1359)

#### 难度

- **Hard**

#### 问题描述

Given `n` orders, each order consist in pickup and delivery services. 

Count all valid pickup/delivery possible sequences such that delivery(i) is always after of pickup(i). 

Since the answer may be too large, return it modulo 10^9 + 7.

**Example 1:**

**Input:** n = 1  
**Output:** 1  
**Explanation:** Unique order (P1, D1), Delivery 1 always is after of Pickup 1.  

**Example 2:**

**Input:** n = 2  
**Output:** 6  
**Explanation:** All possible orders:   
(P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1).  
This is an invalid order (P1,D2,P2,D1) because Pickup 2 is after of Delivery 2.  

**Example 3:**

**Input:** n = 3  
**Output:** 90  

**Constraints:**

- `1 <= n <= 500`

#### 解题思路

- **统计**  
当已完成`n - 1`对订单时，第`n`对订单的接取与送达有`1, 2, 3, ..., 2 * (n - 1) + 1`种排列。假设`n - 1`对排列的结果为`dfs(n - 1)`，则答案为`dfs(n - 1) * (1 + 2 + ... 2n + 1)`。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def countOrders(self, n: int) -> int:
        MOD = 10**9 + 7
        
        @cache
        def dfs(idx):
            if idx == 1:
                return 1
            holes = 2 * (idx - 1) + 1 % MOD
            res = (1 + holes) / 2 * holes % MOD
            return int(dfs(idx - 1) * res % MOD)
        
        return dfs(n)
```

---

## Group the People Given the Group Size They Belong To (1282)

#### 难度

- **Medium**

#### 问题描述

There are `n` people that are split into some unknown number of groups. Each person is labeled with a **unique ID** from `0` to `n - 1`.

You are given an integer array `groupSizes`, where `groupSizes[i]` is the size of the group that person `i` is in. For example, if `groupSizes[1] = 3`, then person `1` must be in a group of size `3`.

Return _a list of groups such that each person `i` is in a group of size `groupSizes[i]`_.

Each person should appear in **exactly one group**, and every person must be in a group. If there are multiple answers, **return any of them**. It is **guaranteed** that there will be **at least one** valid solution for the given input.

**Example 1:**

**Input:** groupSizes = [3,3,3,3,3,1,3]  
**Output:** [[5],[0,1,2],[3,4,6]]  
**Explanation:**   
The first group is [5]. The size is 1, and groupSizes[5] = 1.  
The second group is [0,1,2]. The size is 3, and groupSizes[0] = groupSizes[1] = groupSizes[2] = 3.  
The third group is [3,4,6]. The size is 3, and groupSizes[3] = groupSizes[4] = groupSizes[6] = 3.  
Other possible solutions are [[2,1,6],[5],[0,4,3]] and [[5],[0,6,2],[4,3,1]].  

**Example 2:**

**Input:** groupSizes = [2,1,3,3,3,2]  
**Output:** [[1],[0,5],[2,3,4]]  

**Constraints:**

- `groupSizes.length == n`
- `1 <= n <= 500`
- `1 <= groupSizes[i] <= n`

#### 解题思路

- **贪心**  
利用一个哈希表来记录`groupSize`为`n`所对应的`group`。遍历元素并加入相应的`group`，若`groupSize`达到`n`则将该数组加入答案后重置。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        track = defaultdict(lambda: [[]])
        n = len(groupSizes)
        
        for i in range(n):
            if len(track[groupSizes[i]][-1]) == groupSizes[i]:
                track[groupSizes[i]].append(list())
            track[groupSizes[i]][-1].append(i)
        
        res = []
        
        for key, val in track.items():
            res.extend(val)
        
        return res
```

---

## Minimum Deletions to Make Character Frequencies Unique (1647)

#### 难度

- **Medium**

#### 问题描述

A string `s` is called **good** if there are no two different characters in `s` that have the same **frequency**.

Given a string `s`, return _the **minimum** number of characters you need to delete to make_ `s` _**good**._

The **frequency** of a character in a string is the number of times it appears in the string. For example, in the string `"aab"`, the **frequency** of `'a'` is `2`, while the **frequency** of `'b'` is `1`.

**Example 1:**

**Input:** s = "aab"  
**Output:** 0  
**Explanation:** `s` is already good.  

**Example 2:**

**Input:** s = "aaabbbcc"  
**Output:** 2  
**Explanation:** You can delete two 'b's resulting in the good string "aaabcc".  
Another way it to delete one 'b' and one 'c' resulting in the good string "aaabbc".  

**Example 3:**

**Input:** s = "ceabaacb"  
**Output:** 2  
**Explanation:** You can delete both 'c's resulting in the good string "eabaab".  
Note that we only care about characters that are still in the string at the end (i.e. frequency of 0 is ignored).  

**Constraints:**

- `1 <= s.length <= 105`
- `s` contains only lowercase English letters.

#### 解题思路

- **贪心**  
首先计算各个字母的出现次数并存入一个数组。遍历该数组并记录所见过的出现次数，若当前出现次数已出现过，则递减直到新的次数没出现过或等于0。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def minDeletions(self, s: str) -> int:
        temp = Counter(s)
        freq = temp.values()
        track = set()
        res = 0
        for val in freq:
            while val in track and val > 0:
                val -= 1
                res += 1
            track.add(val)
        return res
```

---

## Candy (135)

#### 难度

- **Hard**

#### 问题描述

There are `n` children standing in a line. Each child is assigned a rating value given in the integer array `ratings`.

You are giving candies to these children subjected to the following requirements:

- Each child must have at least one candy.
- Children with a higher rating get more candies than their neighbors.

Return _the minimum number of candies you need to have to distribute the candies to the children_.

**Example 1:**

**Input:** ratings = [1,0,2]  
**Output:** 5  
**Explanation:** You can allocate to the first, second and third child with 2, 1, 2 candies respectively.  

**Example 2:**

**Input:** ratings = [1,2,2]  
**Output:** 4  
**Explanation:** You can allocate to the first, second and third child with 1, 2, 1 candies respectively.  
The third child gets 1 candy because it satisfies the above two conditions.  

**Constraints:**

- `n == ratings.length`
- `1 <= n <= 2 * 104`
- `0 <= ratings[i] <= 2 * 104`

#### 解题思路

- **贪心**  
分别从正向以及反向遍历数组。若对于索引`i`，若不满足题目条件则将`candies[i]`增加。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1 for _ in range(n)]
        
        for i in range(1, n):
            if ratings[i] > ratings[i - 1] and candies[i] <= candies[i - 1]:
                candies[i] = candies[i - 1] + 1

        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1] and candies[i] <= candies[i + 1]:
                candies[i] = candies[i + 1] + 1
        
        return sum(candies)
```

---

## Reconstruct Itinerary (332)

#### 难度

- **Hard**

#### 问题描述

You are given a list of airline `tickets` where `tickets[i] = [fromi, toi]` represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from `"JFK"`, thus, the itinerary must begin with `"JFK"`. If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

- For example, the itinerary `["JFK", "LGA"]` has a smaller lexical order than `["JFK", "LGB"]`.

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/03/14/itinerary1-graph.jpg)

**Input:** tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]  
**Output:** ["JFK","MUC","LHR","SFO","SJC"]  

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/03/14/itinerary2-graph.jpg)  
 
**Input:** tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]  
**Output:** ["JFK","ATL","JFK","SFO","ATL","SFO"]  
**Explanation:** Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.  

**Constraints:**

- `1 <= tickets.length <= 300`
- `tickets[i].length == 2`
- `fromi.length == 3`
- `toi.length == 3`
- `fromi` and `toi` consist of uppercase English letters.
- `fromi != toi`

#### 解题思路

- **尤拉回路**  
经典尤拉回路题。在各个节点遍历临近节点时要先排序，从小到大遍历以满足最小字典序。

#### 复杂度

- 时间复杂度：$$O(E\log E)$$
- 空间复杂度：$$O(V + E)$$

#### 代码

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        
        for fromCity, toCity in tickets:
            graph[fromCity].append(toCity)
        
        res = []

        def dfs(city):
            graph[city].sort(reverse=True)
            while graph[city]:
                adj = graph[city].pop()
                dfs(adj)
            res.append(city)
        
        dfs('JFK')
        return res[::-1]
```

---

## Min Cost to Connect All Points (1584)

#### 难度

- **Medium**

#### 问题描述

You are given an array `points` representing integer coordinates of some points on a 2D-plane, where `points[i] = [xi, yi]`.

The cost of connecting two points `[xi, yi]` and `[xj, yj]` is the **manhattan distance** between them: `|xi - xj| + |yi - yj|`, where `|val|` denotes the absolute value of `val`.

Return _the minimum cost to make all points connected._ All points are connected if there is **exactly one** simple path between any two points.

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/08/26/d.png)

**Input:** points = [[0,0],[2,2],[3,10],[5,2],[7,0]]  
**Output:** 20  
**Explanation:**   
![](https://assets.leetcode.com/uploads/2020/08/26/c.png)  
We can connect the points as shown above to get the minimum cost of 20.  
Notice that there is a unique path between every pair of points.  

**Example 2:**

**Input:** points = [[3,12],[-2,5],[-4,1]]  
**Output:** 18  
  
**Constraints:**

- `1 <= points.length <= 1000`
- `-106 <= xi, yi <= 106`
- All pairs `(xi, yi)` are distinct.

#### 解题思路

- **最小生成树**  
经典最小生成树题目。用Kruskal方法求解，利用并查集判断是否已连结。

#### 复杂度

- 时间复杂度：$$O(E\log E)$$
- 空间复杂度：$$O(E)$$

#### 代码

```python
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        edges = []
        
        for i in range(n):
            for j in range(i):
                a, b = points[i]
                c, d = points[j]
                dist = abs(a - c) + abs(b - d)
                edges.append((i, j, dist))
        
        edges.sort(key=lambda x: x[2])
        uf = UnionFind(n)
        res = 0
        
        for fromNode, toNode, dist in edges:
            if uf.find(fromNode) == uf.find(toNode):
                continue
            res += dist
            uf.union(fromNode, toNode)
        return res

class UnionFind:
    def __init__(self, size) -> None:
        self.uf = [x for x in range(size)]
        
    def find(self, node):
        if self.uf[node] != node:
            self.uf[node] = self.find(self.uf[node])
        return self.uf[node]
    
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        self.uf[pa] = pb
```

---

## Path With Minimum Effort (1631)

#### 难度

- **Medium**

#### 问题描述

You are a hiker preparing for an upcoming hike. You are given `heights`, a 2D array of size `rows x columns`, where `heights[row][col]` represents the height of cell `(row, col)`. You are situated in the top-left cell, `(0, 0)`, and you hope to travel to the bottom-right cell, `(rows-1, columns-1)` (i.e., **0-indexed**). You can move **up**, **down**, **left**, or **right**, and you wish to find a route that requires the minimum **effort**.

A route's **effort** is the **maximum absolute difference** in heights between two consecutive cells of the route.

Return _the minimum **effort** required to travel from the top-left cell to the bottom-right cell._

**Example 1:**

![](https://assets.leetcode.com/uploads/2020/10/04/ex1.png)

**Input:** heights = [[1,2,2],[3,8,2],[5,3,5]]  
**Output:** 2  
**Explanation:** The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.  
This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.  

**Example 2:**

![](https://assets.leetcode.com/uploads/2020/10/04/ex2.png)  

**Input:** heights = [[1,2,3],[3,8,4],[5,3,5]]  
**Output:** 1  
**Explanation:** The route of [1,2,3,4,5] has a maximum absolute difference of 1 in consecutive cells, which is better than route [1,3,5,3,5].  

**Example 3:**

![](https://assets.leetcode.com/uploads/2020/10/04/ex3.png)

**Input:** heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]  
**Output:** 0  
**Explanation:** This route does not require any effort.  

**Constraints:**

- `rows == heights.length`
- `columns == heights[i].length`
- `1 <= rows, columns <= 100`
- `1 <= heights[i][j] <= 106`

#### 解题思路

- **二分搜索**，**深度优先搜索**  
利用二分搜索来确定答案，利用深度优先搜索来判定给定一个相对高度时，是否有可能路径存在。

#### 复杂度

- 时间复杂度：$$O(mn\log D)$$，D为最大高度
- 空间复杂度：$$O(mn)$$

#### 代码

```python
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = len(heights), len(heights[0])
        vis = set()
        vis.add((0, 0))

        def dfs(x, y, diff):
            if x == m - 1 and y == n - 1:
                return True
            vis[x][y] = True
            for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                newX, newY = x + i, y + j
                if newX < 0 or newY < 0 or newX >= m or newY >= n or vis[newX][newY] == True:
                    continue
                if abs(heights[newX][newY] - heights[x][y]) > diff:
                    continue
                if dfs(newX, newY, diff):
                    return True
            return False
        
        left, right = 0, 10**7
        
        while left < right:
            vis = [[False] * n for _ in range(m)]
            mid = (left + right) >> 1
            if dfs(0, 0, mid):
                right = mid
            else:
                left = mid + 1
        
        return left
```

---

## Shortest Path Visiting All Nodes (847)

#### 难度

- **Hard**

#### 问题描述

You have an undirected, connected graph of `n` nodes labeled from `0` to `n - 1`. You are given an array `graph` where `graph[i]` is a list of all the nodes connected with node `i` by an edge.

Return _the length of the shortest path that visits every node_. You may start and stop at any node, you may revisit nodes multiple times, and you may reuse edges.

**Example 1:**

![](https://assets.leetcode.com/uploads/2021/05/12/shortest1-graph.jpg)

**Input:** graph = [[1,2,3],[0],[0],[0]]  
**Output:** 4  
**Explanation:** One possible path is [1,0,2,0,3]  

**Example 2:**

![](https://assets.leetcode.com/uploads/2021/05/12/shortest2-graph.jpg)

**Input:** graph = [[1],[0,2,4],[1,3,4],[2],[1,2]]  
**Output:** 4  
**Explanation:** One possible path is [0,1,4,2,3]  

**Constraints:**

- `n == graph.length`
- `1 <= n <= 12`
- `0 <= graph[i].length < n`
- `graph[i]` does not contain `i`.
- If `graph[a]` contains `b`, then `graph[b]` contains `a`.
- The input graph is always connected.

#### 解题思路

- **广度优先算法**，**位掩码**  
本题的限制很小，所以直接用BFS暴力求解。BFS的状态为`idx, bitmask`。

#### 复杂度

- 时间复杂度：$O(n^2\times 2^n)$
- 空间复杂度：$O(n\times 2^n)$

#### 代码

```python
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        n = len(graph)
        dq = deque()
        
        for i in range(n):
            dq.append((i, 1 << i))
        
        res = 0
        vis = set()
        
        while dq:
            for _ in range(len(dq)):
                node, bitmask = dq.popleft()
                for adj in graph[node]:
                    newMask = bitmask | (1 << adj)
                    if (adj, newMask) in vis:
                        continue
                    if newMask == (1 << n) - 1:
                        return res + 1
                    dq.append((adj, newMask))
                    vis.add((adj, newMask))
            res += 1
        
        return 0
```
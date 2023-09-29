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

- 时间复杂度：$$O(n^2\times 2^n)$$
- 空间复杂度：$$O(n\times 2^n)$$

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

---

## The K Weakest Rows in a Matrix (1337)

#### 难度

- **Easy**

#### 问题描述

You are given an `m x n` binary matrix `mat` of `1`'s (representing soldiers) and `0`'s (representing civilians). The soldiers are positioned **in front** of the civilians. That is, all the `1`'s will appear to the **left** of all the `0`'s in each row.

A row `i` is **weaker** than a row `j` if one of the following is true:

- The number of soldiers in row `i` is less than the number of soldiers in row `j`.
- Both rows have the same number of soldiers and `i < j`.

Return _the indices of the_ `k` _**weakest** rows in the matrix ordered from weakest to strongest_.

**Example 1:**

**Input:** mat =   
[[1,1,0,0,0],  
 [1,1,1,1,0],  
 [1,0,0,0,0],  
 [1,1,0,0,0],  
 [1,1,1,1,1]],   
k = 3  
**Output:** [2,0,3]  
**Explanation:**   
The number of soldiers in each row is:   
- Row 0: 2   
- Row 1: 4   
- Row 2: 1 
- Row 3: 2 
- Row 4: 5 
The rows ordered from weakest to strongest are [2,0,3,1,4].

**Example 2:**

**Input:** mat =   
[[1,0,0,0],  
 [1,1,1,1],  
 [1,0,0,0],  
 [1,0,0,0]],   
k = 2  
**Output:** [0,2]  
**Explanation:**   
The number of soldiers in each row is:   
- Row 0: 1 
- Row 1: 4 
- Row 2: 1 
- Row 3: 1 
The rows ordered from weakest to strongest are [0,2,3,1].

**Constraints:**

- `m == mat.length`
- `n == mat[i].length`
- `2 <= n, m <= 100`
- `1 <= k <= m`
- `matrix[i][j]` is either 0 or 1.

#### 解题思路

- **二分查找**，**优先队列**  
对于每一行，用二分查找快速找到`1`的数量，然后用优先队列来查找`1`数量最少以及编号最小的`k`个元素。

#### 复杂度

- 时间复杂度：$$O(n\log n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```java
class Solution {
    public int[] kWeakestRows(int[][] mat, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>(
            (a, b) -> {
                if (a[0] != b[0]) { return a[0] - b[0]; }
                else { return a[1] - b[1]; }
            }
        );
        int[] res = new int[k];
        
        for (int i = 0; i < mat.length; i++) {
            int[] row = mat[i];
            int idx = countOnes(row);
            pq.offer(new int[] {idx, i});
        }
        
        for (int i = 0; i < res.length; i++) { res[i] = pq.poll()[1]; }
        
        return res;
    }
    
    int countOnes (int[] row) {
        int left = 0, right = row.length, mid;
        
        while (left < right) {
            mid = left + (right - left) / 2;
            if (row[mid] == 1) { left = mid + 1; }
            else { right = mid; }
        }
        
        return left;
    }
}
```

---

## Find the Duplicate Number (287)

#### 难度

- **Medium**

#### 问题描述

Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive.

There is only **one repeated number** in `nums`, return _this repeated number_.

You must solve the problem **without** modifying the array `nums` and uses only constant extra space.

**Example 1:**

**Input:** nums = [1,3,4,2,2]  
**Output:** 2  

**Example 2:**

**Input:** nums = [3,1,3,4,2]  
**Output:** 3  

**Constraints:**

- `1 <= n <= 105`
- `nums.length == n + 1`
- `1 <= nums[i] <= n`
- All the integers in `nums` appear only **once** except for **precisely one integer** which appears **two or more** times.

**Follow up:**

- How can we prove that at least one duplicate number must exist in `nums`?
- Can you solve the problem in linear runtime complexity?

#### 解题思路

- **Floyd判圈法**  
可将本题转化为判断链表中的环的起点的题目，相同的数字即为环的起点。利用Floyd判圈法找到该起点。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow, fast = nums[0], nums[nums[0]]
        
        while slow != fast:
            slow = nums[slow]
            fast = nums[nums[fast]]
        
        slow = 0
        
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        
        return slow
```

---

## Minimum Operations to Reduce X to Zero (1658)

#### 难度

- **Medium**

#### 问题描述

You are given an integer array `nums` and an integer `x`. In one operation, you can either remove the leftmost or the rightmost element from the array `nums` and subtract its value from `x`. Note that this **modifies** the array for future operations.

Return _the **minimum number** of operations to reduce_ `x` _to **exactly**_ `0` _if it is possible__, otherwise, return_ `-1`.

**Example 1:**

**Input:** nums = [1,1,4,2,3], x = 5  
**Output:** 2  
**Explanation:** The optimal solution is to remove the last two elements to reduce x to zero.  

**Example 2:**

**Input:** nums = [5,6,7,8,9], x = 4  
**Output:** -1  

**Example 3:**

**Input:** nums = [3,2,20,1,1,3], x = 10  
**Output:** 5  
**Explanation:** The optimal solution is to remove the last three elements and the first two elements (5 operations in total) to reduce x to zero.  

**Constraints:**

- `1 <= nums.length <= 105`
- `1 <= nums[i] <= 104`
- `1 <= x <= 109`

#### 解题思路

- **前缀和**，**哈希表**  
利用前缀和和后缀和先将子数组和的结果储存起来，并将后缀和存入哈希表。遍历前缀和的元素，检查与`x`的差值是在哈希表中存在。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        n = len(nums)
        pre = [0] * n
        suff = [0] * n
        pre[0] = nums[0]
        suff[-1] = nums[-1]
        track = dict()
        
        for i in range(1, n):
            pre[i] = pre[i - 1] + nums[i]
        
        for i in range(n - 2, -1, -1):
            suff[i] = suff[i + 1] + nums[i]
            
        suff.reverse()

        for idx, val in enumerate(suff):
            track[val] = idx
        
        res = 10**6 if x not in track else track[x] + 1

        for i in range(-1, n):
            leftVal = 0 if i == -1 else pre[i]
            target = x - leftVal
            if target == 0:
                res = min(res, i + 1)
                continue
            idx = track[target] if target in track else -1
            if idx == -1:
                continue
            if (i + 1 + idx + 1) >= n:
                continue
            res = min(res, i + 1 + idx + 1)
        
        return -1 if res == 10**6 else res
```

---

## Median of Two Sorted Arrays (4)

#### 难度

- **Hard**

#### 问题描述

Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be `O(log (m+n))`.

**Example 1:**

**Input:** nums1 = [1,3], nums2 = [2]  
**Output:** 2.00000  
**Explanation:** merged array = [1,2,3] and median is 2.  

**Example 2:**

**Input:** nums1 = [1,2], nums2 = [3,4]  
**Output:** 2.50000  
**Explanation:** merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.  

**Constraints:**

- `nums1.length == m`
- `nums2.length == n`
- `0 <= m <= 1000`
- `0 <= n <= 1000`
- `1 <= m + n <= 2000`
- `-106 <= nums1[i], nums2[i] <= 106`

#### 解题思路

- **二分搜索**  
将数组归并后做二分搜索。

#### 复杂度

- 时间复杂度：$$O(m + n)$$
- 空间复杂度：$$O(m + n)$$

#### 代码

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums = []
        i, j = 0, 0

        while i < len(nums1) or j < len(nums2):
            if i >= len(nums1):
                nums.append(nums2[j])
                j += 1
                continue
            if j >= len(nums2):
                nums.append(nums1[i])
                i += 1
                continue
            if nums1[i] < nums2[j]:
                nums.append(nums1[i])
                i += 1
            else:
                nums.append(nums2[j])
                j += 1
        
        if len(nums) % 2 == 1:
            return nums[len(nums) // 2]
        else:
            return (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2
```

---

## Is Subsequence (392)

#### 难度

- **Easy**

#### 问题描述

Given two strings `s` and `t`, return `true` _if_ `s` _is a **subsequence** of_ `t`_, or_ `false` _otherwise_.

A **subsequence** of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., `"ace"` is a subsequence of `"abcde"` while `"aec"` is not).

**Example 1:**

**Input:** s = "abc", t = "ahbgdc"  
**Output:** true  

**Example 2:**

**Input:** s = "axc", t = "ahbgdc"  
**Output:** false  

**Constraints:**

- `0 <= s.length <= 100`
- `0 <= t.length <= 104`
- `s` and `t` consist only of lowercase English letters.

**Follow up:** Suppose there are lots of incoming `s`, say `s1, s2, ..., sk` where `k >= 109`, and you want to check one by one to see if `t` has its subsequence. In this scenario, how would you change your code?

#### 解题思路

- **双指针**  
遍历`t`，若当前字母与`s`所指的字母相同，则指针+1。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        
        idx = 0
        
        for letter in t:
            if letter == s[idx]:
                idx += 1
            if idx == len(s):
                return True
        
        return False
```

---

## Longest String Chain (1048)

#### 难度

- **Medium**

#### 问题描述

You are given an array of `words` where each word consists of lowercase English letters.

`wordA` is a **predecessor** of `wordB` if and only if we can insert **exactly one** letter anywhere in `wordA` **without changing the order of the other characters** to make it equal to `wordB`.

- For example, `"abc"` is a **predecessor** of `"abac"`, while `"cba"` is not a **predecessor** of `"bcad"`.

A **word chain** is a sequence of words `[word1, word2, ..., wordk]` with `k >= 1`, where `word1` is a **predecessor** of `word2`, `word2` is a **predecessor** of `word3`, and so on. A single word is trivially a **word chain** with `k == 1`.

Return _the **length** of the **longest possible word chain** with words chosen from the given list of_ `words`.

**Example 1:**

**Input:** words = ["a","b","ba","bca","bda","bdca"]  
**Output:** 4  
**Explanation**: One of the longest word chains is ["a","ba","bda","bdca"].  

**Example 2:**

**Input:** words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]  
**Output:** 5  
**Explanation:** All the words can be put in a word chain ["xb", "xbc", "cxbc", "pcxbc", "pcxbcf"].  

**Example 3:**

**Input:** words = ["abcd","dbqca"]  
**Output:** 1  
**Explanation:** The trivial word chain ["abcd"] is one of the longest word chains.  
["abcd","dbqca"] is not a valid word chain because the ordering of the letters is changed.  

**Constraints:**

- `1 <= words.length <= 1000`
- `1 <= words[i].length <= 16`
- `words[i]` only consists of lowercase English letters.

#### 解题思路

- **动态规划**  
DP的状态为`word`，一个`word`只能向长度加一且能组成`chain`的`word`做转移。

#### 复杂度

- 时间复杂度：$$O(n^2 * L)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        dic = defaultdict(list)
        
        for word in words:
            length = len(word)
            dic[length].append(word)
        
        def isChain(a, b):
            if not len(b) == len(a) + 1:
                return False
            j = 0
            for i in range(len(b)):
                if j == len(a):
                    continue
                if b[i] == a[j]:
                    j += 1
            return j == len(a)

        @cache
        def dfs(word):
            length = len(word)
            res = 0
            for adj in dic[length + 1]:
                if isChain(word, adj):
                    res = max(res, 1 + dfs(adj))
            return res
        
        res = 0
        
        for word in words:
            res = max(res, 1 + dfs(word))
        
        return res
```

---

## Champagne Tower (799)

#### 难度

- **Medium**

#### 问题描述

We stack glasses in a pyramid, where the **first** row has `1` glass, the **second** row has `2` glasses, and so on until the 100th row.  Each glass holds one cup of champagne.

Then, some champagne is poured into the first glass at the top.  When the topmost glass is full, any excess liquid poured will fall equally to the glass immediately to the left and right of it.  When those glasses become full, any excess champagne will fall equally to the left and right of those glasses, and so on.  (A glass at the bottom row has its excess champagne fall on the floor.)

For example, after one cup of champagne is poured, the top most glass is full.  After two cups of champagne are poured, the two glasses on the second row are half full.  After three cups of champagne are poured, those two cups become full - there are 3 full glasses total now.  After four cups of champagne are poured, the third row has the middle glass half full, and the two outside glasses are a quarter full, as pictured below.

![](https://s3-lc-upload.s3.amazonaws.com/uploads/2018/03/09/tower.png)

Now after pouring some non-negative integer cups of champagne, return how full the `jth` glass in the `ith` row is (both `i` and `j` are 0-indexed.)

**Example 1:**

**Input:** poured = 1, query_row = 1, query_glass = 1  
**Output:** 0.00000  
**Explanation:** We poured 1 cup of champange to the top glass of the tower (which is indexed as (0, 0)). There will be no excess liquid so all the glasses under the top glass will remain empty.  

**Example 2:**

**Input:** poured = 2, query_row = 1, query_glass = 1  
**Output:** 0.50000  
**Explanation:** We poured 2 cups of champange to the top glass of the tower (which is indexed as (0, 0)). There is one cup of excess liquid. The glass indexed as (1, 0) and the glass indexed as (1, 1) will share the excess liquid equally, and each will get half cup of champange.  

**Example 3:**

**Input:** poured = 100000009, query_row = 33, query_glass = 17
**Output:** 1.00000

**Constraints:**

- `0 <= poured <= 109`
- `0 <= query_glass <= query_row < 100`

#### 解题思路

- **模拟**  
模拟整个倒香槟的过程。若某个杯子上会有`x`杯的量的话，它会取走`1`杯的量，将剩余的均匀分给它下方两侧的杯子。

#### 复杂度

- 时间复杂度：$$O(row^2)$$
- 空间复杂度：$$O(row^2)$$

#### 代码

```python
class Solution:
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        tower = [[0.] * row for row in range(1, query_row + 3)]
        tower[0][0] = poured
        
        for i in range(query_row + 1):
            for j in range(len(tower[i])):
                left = (tower[i][j] - 1.) / 2.
                if left > 0:
                    tower[i + 1][j] += left
                    tower[i + 1][j + 1] += left
        
        return min(1, tower[query_row][query_glass])
```

## Find the Difference (389)

#### 难度

- **Easy**

#### 问题描述

You are given two strings `s` and `t`.

String `t` is generated by random shuffling string `s` and then add one more letter at a random position.

Return the letter that was added to `t`.

**Example 1:**

**Input:** s = "abcd", t = "abcde"  
**Output:** "e"  
**Explanation:** 'e' is the letter that was added.  

**Example 2:**

**Input:** s = "", t = "y"  
**Output:** "y"  

**Constraints:**

- `0 <= s.length <= 1000`
- `t.length == s.length + 1`
- `s` and `t` consist of lowercase English letters.

#### 解题思路

- **数组**  
记录字母在`s`中的出现次数。遍历`t`中的字母，并将对应字母的出现次数-1，若减之前次数为0，则该字母为新加的字母。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```java
class Solution {
    public char findTheDifference(String s, String t) {
        int[] track = new int[26];
        
        for (int i = 0; i < s.length(); i++) {
            int idx = Integer.valueOf(s.charAt(i)) - 97;
            track[idx] += 1;
        }
        
        for (int i = 0; i < t.length(); i++) {
            int idx = Integer.valueOf(t.charAt(i)) - 97;
            if (track[idx] == 0) { return t.charAt(i); }
            track[idx] -= 1;
        }
        
        return '?';
    }
}
```

## Remove Duplicate Letters (316)

#### 难度

- **Medium**

#### 问题描述

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is

**the smallest in lexicographical order**

among all possible results.

**Example 1:**

**Input:** s = "bcabc"  
**Output:** "abc"  

**Example 2:**

**Input:** s = "cbacdcbc"  
**Output:** "acdb"  

**Constraints:**

- `1 <= s.length <= 104`
- `s` consists of lowercase English letters.

**Note:** This question is the same as 1081: [https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/)

#### 解题思路

- **单调栈**  
题目要求回传字典序最小的答案，所以可以用一个`单调栈`的数据结构来维护答案。单调栈为单调递增栈，且只当当前栈顶元素大于当前元素且后续有另一个栈顶元素时弹出。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(n)$$

#### 代码

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack = []
        lastAppear = dict()
        
        for i in range(len(s)):
            lastAppear[s[i]] = i
        
        vis = set()
        
        for i in range(len(s)):
            if s[i] in vis:
                continue
            while stack and stack[-1] > s[i] and lastAppear[stack[-1]] > i and s[i] not in vis:
                vis.remove(stack.pop())
            stack.append(s[i])
            vis.add(s[i])
        
        return ''.join(stack)
```

---

## Decoded String at Index (880)

#### 难度

- **Medium**

#### 问题描述

You are given an encoded string `s`. To decode the string to a tape, the encoded string is read one character at a time and the following steps are taken:

- If the character read is a letter, that letter is written onto the tape.
- If the character read is a digit `d`, the entire current tape is repeatedly written `d - 1` more times in total.

Given an integer `k`, return _the_ `kth` _letter (**1-indexed)** in the decoded string_.

**Example 1:**

**Input:** s = "leet2code3", k = 10  
**Output:** "o"  
**Explanation:** The decoded string is "leetleetcodeleetleetcodeleetleetcode".  
The 10th letter in the string is "o".  

**Example 2:**

**Input:** s = "ha22", k = 5  
**Output:** "h"  
**Explanation:** The decoded string is "hahahaha".  
The 5th letter is "h".  

**Example 3:**

**Input:** s = "a2345678999999999999999", k = 1  
**Output:** "a"  
**Explanation:** The decoded string is "a" repeated 8301530446056247680 times.  
The 1st letter is "a".  

**Constraints:**

- `2 <= s.length <= 100`
- `s` consists of lowercase English letters and digits `2` through `9`.
- `s` starts with a letter.
- `1 <= k <= 109`
- It is guaranteed that `k` is less than or equal to the length of the decoded string.
- The decoded string is guaranteed to have less than `263` letters.

#### 解题思路

- **数学**，**模拟**  
用一个变量记录一个个解码密文时，明文的长度。当明文长度大于`k`时，进行反向编码，并找到第`k`个元素。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```java
class Solution {
    public String decodeAtIndex(String s, int k) {
        long idx = 0;
        int i;

        for (i = 0; i < s.length(); i++) {
            if (Character.isDigit(s.charAt(i))) {
                idx *= (s.charAt(i) - '0');
            } else {
                idx += 1;
            }
        }
        
        for (i--; i > -1; i--) {
            if (Character.isDigit(s.charAt(i))) {
                idx /= (s.charAt(i) - '0');
                k %= idx;
            } else {
                if (k == 0 || idx == k) {
                    return Character.toString(s.charAt(i));
                }
                idx--;
            }
        }
        
        return "0";
    }
}
```

---

## Sort Array By Parity (905)

#### 难度

- **Easy**

#### 问题描述

Given an integer array `nums`, move all the even integers at the beginning of the array followed by all the odd integers.

Return _**any array** that satisfies this condition_.

**Example 1:**  

**Input:** nums = [3,1,2,4]  
**Output:** [2,4,3,1]  
**Explanation:** The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.  

**Example 2:**

**Input:** nums = [0]  
**Output:** [0]  

**Constraints:**

- `1 <= nums.length <= 5000`
- `0 <= nums[i] <= 5000`

#### 解题思路

- **数组**  
维护一个链表。遍历数组的同时，若元素为偶数则加到链表头，若元素为奇数则加到链表尾。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        dq = deque()
        
        for val in nums:
            if val % 2 == 0:
                dq.appendleft(val)
            else:
                dq.append(val)
        
        return list(dq)
```

---

## Monotonic Array (896)

#### 难度

- **Easy**

#### 问题描述

An array is **monotonic** if it is either monotone increasing or monotone decreasing.

An array `nums` is monotone increasing if for all `i <= j`, `nums[i] <= nums[j]`. An array `nums` is monotone decreasing if for all `i <= j`, `nums[i] >= nums[j]`.

Given an integer array `nums`, return `true` _if the given array is monotonic, or_ `false` _otherwise_.

**Example 1:**

**Input:** nums = [1,2,2,3]  
**Output:** true  

**Example 2:**

**Input:** nums = [6,5,4,4]  
**Output:** true  

**Example 3:**

**Input:** nums = [1,3,2]  
**Output:** false  

**Constraints:**

- `1 <= nums.length <= 105`
- `-105 <= nums[i] <= 105`

#### 解题思路

- **数组**  
遍历数组。首先决定到现在的元素前为单调递增还是单调递减，然后判断增加现在的元素后是否保持单调。

#### 复杂度

- 时间复杂度：$$O(n)$$
- 空间复杂度：$$O(1)$$

#### 代码

```python
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        sign = None
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                continue
            if sign == None:
                sign = (nums[i] - nums[i - 1]) > 0
                continue
            if (nums[i] - nums[i - 1] > 0) != sign:
                return False
        
        return True
```
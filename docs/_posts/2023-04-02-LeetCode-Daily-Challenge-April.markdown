---
layout: post
title:  "LeetCode Daily Challenge April"
date:   2023-04-02 13:16:49 +0800
categories: LeetCode Algorithm
# usemathjax: true
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
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
  - 判断该船在`救援当前最重的人`的同时是否可以塞`当前最轻的人`上船，然后收缩左右指针，同时船数+2。

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

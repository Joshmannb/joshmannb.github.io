---
layout: post
title:  "LeetCode: Search"
date:   2023-04-05 20:57:31 +0800
categories: LeetCode Algorithm
katex: true
author:
    name: 陈家惠
    picture: "/images/avatar.jpg"
# category_archive_path: "/categories/"
---

This post is for solutions of searching problems.

{% include toc %}

## Count All Possible Routes (1575)

#### 问题描述

- [LeetCode](https://leetcode.com/problems/count-all-possible-routes/)

#### 解题思路

- 记忆化搜索
  - 该题为`top-down dp`,可以用`DFS`加上`Memorization`解
  - 在DFS中：
    - 若当前`state`的`fuel < 0`，则当前的`path`不存在，返回0
    - 若当前`state`的`start city`为`finish city`，则证明找到一条可行的`path`，结果+1
    - 通过`@cache`将走过的状态作缓存

#### 时间复杂度

- DP: $$O(fuel\times{n^{2}})$$

#### 代码

```python
class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        graph = defaultdict()
        for city, location in enumerate(locations):
            graph[city] = location
        
        @cache
        def dfs(start, fuel):
            if fuel < 0:
                return 0
            output = 1 if start == finish else 0
            for city in graph.keys():
                if city == start:
                    continue
                output += dfs(city, fuel - abs(graph[start] - graph[city]))
            return int(output % (1e9 + 7))
        
        return int(dfs(start, fuel) % (1e9 + 7))
```
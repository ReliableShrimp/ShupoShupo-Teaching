#  Graphs — connecting our bonds.

> *"Everybody is a genius. But if you judge a fish by its ability to climb a tree, it will live its whole life believing that it is stupid."* — Einstein
>
> Sadly, he never had to survive the big-tech hiring era of 2026, where if you can't climb the tree, nobody needs you. So we swim, we climb, we limp — whatever it takes. Let's learn graphs.

Beating dragons wasn't enough — manually deriving linear regression, logistic regression, and the rest. Now we get to build an [evidence board](https://en.wikipedia.org/wiki/Evidence_board) of our own: **graphs**.

(You might be asking yourself: *why all the casual detours on a math topic?* That's just the format — bear with it.)

You will not be finding any murderers with this. You will, more realistically, be staring at a screen wondering why your neurons — sorry, **nodes** — aren't connecting (**edges**).

We'll start from mega-basics, because if I open with GraphSAGE on day one, your brain fries — and so does mine. Baby steps first.

##  Table of Contents

1. Nodes and Edges
2. The Burger-Man Problem
3. Zachary's Karate Club
4. Joining the Club: Shrimp's Story
5. Measuring Power: Centrality Stats
6. Finding the Most-Connected Node
7. Types of Graphs
8. BFS and DFS
9. The Final Bosses of the Month
10. Dijkstra's Algorithm
11. Degree Centrality
12. Connected Components
13. Cycle Detection
14. Topological Sort
15. PageRank
16. Betweenness Centrality
17. Final Project — Progetto Finale

---

## 1. Nodes and Edges

- **Nodes (a.k.a. Vertex)** — the dots on the graph. A node can be a person, an atom, a contact, a dog, a neuron... that's it.
- **Edges** — the connections between nodes. If the nodes are contacts, the edges are the relationships between them. If the nodes are atoms, the edges are atomic bonds. And so on.

---

## 2. The Burger-Man Problem

Imagine Contact **G** is a big, famous burger-man — rich, and he gives out free burgers. But you (Contact **A**) got into an argument with *every single one* of the other contacts (nodes). Now you're allowed to forgive exactly one of them. Who do you choose?

The friendships look like this:

- **A** is friends with B, E, C
- **B** is friends with A, E, C
- **C** is friends with B, A, E
- **D** is friends with E, C
- **E** is friends with A, B, C, D, F
- **F** is friends with E, G
- **G** is friends with F

> [!TIP]
> **You choose to forgive Contact E.** Trace the paths: A can already reach B, C, D through its existing friendships — but the *only* route from A to the burger-man G runs through E → F → G. E is the bridge. Forgive anyone else, and you're still cut off from the burgers.

<img width="465" height="372" alt="image" src="https://github.com/user-attachments/assets/98573f2b-6abc-43f3-bfc2-a3733399bcfc" />


---

## 3. Zachary's Karate Club

Now for something classic: the **Zachary Karate Club**. Yes, an actual karate club.

The story: the club's administrator (**Mr. Hi**) got into an argument with an important member (**John A**) over club fees. The club split — half the students sided with Mr. Hi, half with John A. Data scientists were later able to mathematically predict which side *every single member* would land on, with 97% accuracy. (They missed exactly one guy — he probably didn't pick a side and just quietly left.)

This dataset ships with `networkx` as a ready-made playground:

```python
import networkx as nx

G = nx.karate_club_graph()

print(f"Total Members (Nodes): {G.number_of_nodes()}")
print(f"Total Friendships (Edges): {G.number_of_edges()}")

"""
Output:

Total Members (Nodes): 34
Total Friendships (Edges): 78
"""

# Look at the instructor (Node 0)
print(G.nodes[0])
# Output: {'club': 'Mr. Hi'}

# Look at the president (Node 33)
print(G.nodes[33])
# Output: {'club': 'Officer'}
```

Now we know how to check our node/edge counts, and inspect any individual member. Baby steps — but worth it, because it gets more fun from here (no promises, though).

---

## 4. Joining the Club: Shrimp's Story

> While wandering around as a homeless Shrimp, he found a poster:
>
> *"Mr. Hi and John A argued over some fees... and I heard Mr. Hi will give the money to whoever joins him."* — Written by Mr. Hi
>
> Shrimp was mesmerized. Free money (not a scam, surely), so he immediately signed up:

```python
import networkx as nx

G = nx.karate_club_graph()

G.add_node("Shrimp", club="Mr. Hi")
# General pattern: .add_node("<name>", club="<the_club>")
```

> After joining, Shrimp realized there were already 17 students on Mr. Hi's side. His odds of a payout just dropped to 1/18 ≈ 5.5%. He hated it — but he needed that money to stop learning and get on with his life. Then he tripped, and bumped into a beautiful, cute young woman: **Airi Sezaki**.
>
> He stood up, cleared his throat, and apologized. She accepted. They talked a while. Shrimp had just made a friend — his second, actually, since Mr. Hi auto-befriends every clan member. He wired it up immediately:

```python
G.add_edge("Shrimp", 9)  # He added Airi first — she was student #9 on the list.
                          # (This creates an edge from Node "Shrimp" to Node 9.)
G.add_edge("Shrimp", 0)  # Then Mr. Hi.

print(f"New Total Members: {G.number_of_nodes()}")
print(f"New Total Friendships: {G.number_of_edges()}")

"""
Output:

New Total Members: 35
New Total Friendships: 80
"""
```

> Shrimp checked whether he was, in fact, friends with Airi — and, less importantly, Mr. Hi:

```python
print(list(G.neighbors("Shrimp")))
# We wrap it in list() because .neighbors() returns an iterator, not a list.
# Output: [9, 0]
```

> Now Shrimp presses on — driven by the goal of getting the money, or being loved by Airi. Being a greedy shrimp, he wants both.

---

## 5. Measuring Power: Centrality Stats

Shrimp just joined. Realistically, nobody needs him — if he vanished tomorrow, nobody would notice. Naturally, he wants receipts:

```python
# Calculate the power dynamics now that Shrimp has altered the network
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# Let's check Shrimp's specific scores
print(f"Shrimp's Social Butterfly Score (Degree):    {degree_centrality['Shrimp']:.4f}")
print(f"Shrimp's Gatekeeper Score (Betweenness):      {betweenness['Shrimp']:.4f}")
print(f"Shrimp's Influencer Score (PageRank):         {pagerank['Shrimp']:.4f}")

"""
Output:

Shrimp's Social Butterfly Score (Degree):    0.0588
Shrimp's Gatekeeper Score (Betweenness):      0.0080
Shrimp's Influencer Score (PageRank):         0.0089
"""
```

Let's break each one down:

- **`nx.degree_centrality(G)`** — how much of a social butterfly everyone is. Shrimp just joined with 2 friends out of 34 possible, so his score is a lowly `0.0588`. Mr. Hi, by contrast, sits near `0.5000` — he's friends with almost half the club.
- **`nx.betweenness_centrality(G)`** — Shrimp's weakest stat, and for good reason: it measures how essential you are as a bridge between other people's shortest paths. Everyone can already reach Airi and Mr. Hi without going through Shrimp, so his score is a near-nothing `0.0080`. Mr. Hi, on the other hand, sits in the structural middle of the whole club — nearly every cross-club message has to pass through him.
- **`nx.pagerank`** — here Shrimp gets to cash in. PageRank cares less about *how many* friends you have and more about *how important* they are. Shrimp is friends with Mr. Hi — a genuinely important node — so he gets a disproportionate algorithmic boost just from standing next to the boss. (Airi helps too — she's popular in her own right.)

---

## 6. Finding the Most-Connected Node

Let's say we want to find whoever has the single highest degree centrality score in the whole club.

```python
degree_dict = nx.degree_centrality(G)

# Illustrative shape of the dictionary:
# degree_dict = {
#     0: 0.4706,        # Mr. Hi   (connected to a huge share of the club)
#     33: 0.5000,       # John A   (the president, heavily connected)
#     9: 0.0888,        # Airi     (a popular, well-liked student)
#     "Shrimp": 0.0588  # Shrimp   (just joined — knows only Mr. Hi & Airi)
# }

sorted_degree = sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)
top_node, top_score = sorted_degree[0]
print(f"The most connected member is Node {top_node} with a score of {top_score:.4f}")
```

Now let's take that apart, slowly:

- **`sorted()`** — sorts data. For numbers, that means smallest to largest by default.
- **`.items()`** — turns our dictionary into a list of `(key, value)` tuples:
  ```python
  degree_dict.items()
  # [(0, 0.4706), (33, 0.5000), (9, 0.0888), ("Shrimp", 0.0588)]
  ```
  Each item is now `(node, score_of_the_node)`.
- **But sort by what — the node, or the score?** Python has no idea unless we tell it. That's what `key=lambda item: item[1]` is for: "for each item, look at index `1`" — index `0` is the node, index `1` is the score. So this sorts by score.

  > [!WARNING]
  > A classic typo here is writing `items[1]` instead of `item[1]` — referencing the *wrong* variable name (the plural, undefined one) instead of the loop variable the lambda actually receives. That throws a `NameError`. Always match the lambda's parameter name exactly.

- **`reverse=True`** — flips it from smallest→largest to largest→smallest, since we want the *most* connected node first.
- **`sorted_degree[0]`** — grabs the very first tuple, which is now the highest-scoring one.

```python
"""
Output:

[
    (33, 0.5000),
    (0, 0.4706),
    (9, 0.0888),
    ("Shrimp", 0.0588)
]
"""

top_node, top_score = sorted_degree[0]
# top_node = 33
# top_score = 0.5000

print(f"The most connected member is Node {top_node} with a score of {top_score:.4f}")
# The most connected member is Node 33 with a score of 0.5000
```

From this, Shrimp realizes that while he's safe standing next to Mr. Hi, the rival faction holds a genuine structural edge in raw popularity. If a war breaks out, Mr. Hi's money might not be enough to buy everyone over.

Dramatic, sure — but that's normal. We'll keep going this way all the way to GraphSAGE and beyond.


---

## 7. Types of Graphs

Before going further, let's classify the kinds of graphs out there — this determines which strategy applies when you encounter one:

1. **Undirected vs. Directed**
2. **Unweighted vs. Weighted**
3. **Cyclic vs. Acyclic (DAG)**

### Undirected vs. Directed

<img width="550" height="298" alt="image" src="https://github.com/user-attachments/assets/cff443d4-3e7a-4d88-b809-564e19041b21" />

- **Undirected Graphs** — connected by plain lines between nodes, no arrows. If you're at node 1, you can move to 2, then to 3 or 4, or back to 1 — movement is free in both directions along any edge.
- **Directed Graphs** — connected by arrows that explicitly say where you're *allowed* to go. Go from node 1 to node 2, and there may be no way back — only forward to node 3 or 4. The only way back is if a separate arrow explicitly points backward.

### Unweighted vs. Weighted

<img width="544" height="291" alt="image" src="https://github.com/user-attachments/assets/9c7eb2fd-3eb3-4884-85d7-dd6363994777" />


- **Unweighted Graphs** — every edge is identical. No numbers attached. "Distance" is purely the number of hops (edges) it takes to get somewhere.
- **Weighted Graphs** — every edge carries a number (a weight): cost, distance, time, whatever the domain calls for. This lets us reason about tradeoffs — e.g. if the weights represent kilometers, a route of 10 km direct might lose to a "longer-looking" route via a middle stop that only costs 4 + 1 = 5 km total.

### Cyclic vs. Acyclic

<img width="511" height="281" alt="image" src="https://github.com/user-attachments/assets/1100a28f-f5a8-45da-b401-bb489350dc35" />


- **Cyclic Graphs** — contain at least one loop: start at a node, follow a path of edges, and return to the *start*. For example: A → C → B → A → C → B → A → ... forever, unless you explicitly track visited nodes.

  > [!WARNING]
  > Loop forever in real code and you'll get a `Segmentation fault` or `Stack Overflow`. High algorithmic risk if visited-node tracking is missing.

- **Acyclic Graphs** — no matter where you start or which path you follow, you always hit a dead end eventually; no loop is possible. Directed acyclic graphs are commonly called **DAGs**. Lower risk here — worst case, you just hit a dead end.


---

## 8. BFS and DFS

### Queues, Stacks, FIFO, and LIFO

**BFS (Breadth-First Search)** works like [ripples](https://dictionary.cambridge.org/dictionary/english/ripple) in a pond after a rock is thrown in. It doesn't shoot off in one direction — it expands outward in circular waves, all sides growing at the same uniform speed.

BFS relies on a **queue** — a **FIFO** structure (First In, First Out): the first thing added is the first thing removed.

> Imagine three pans on the stove. You put down pan B, then C, then A. FIFO means B comes off first, then C, then A — first placed, first removed.

A queue only supports two operations:

- **Enqueue** (join the line) — new data goes to the **back**.
- **Dequeue** (get served) — the system takes whatever is at the **front**.

Think of a fast-food drive-thru: Car A pulls up front and orders, Car B lines up behind it, Car C behind B. They're single-file. Serving starts with A, then B, then C — in that order.

Or a checkout line at the market: A, B, C are already waiting, then D arrives.

- **Enqueue**: D goes straight to the *rear*. No searching, no shifting — just append. Time = **O(1)**, constant time, whether there are 3 people in line or 3 million.
- **Dequeue**: the front customer leaves. `[A, B, C]` → `[B, C]`. The front pointer just moves to B; nobody else is touched. Time = **O(1)**.

<img width="530" height="300" alt="image" src="https://github.com/user-attachments/assets/bd426fac-4e00-4e64-bb25-a6a231ad6aaf" />


Now imagine Shrimp needs to deliver a message to someone across a crowded room, but can only speak to the people close enough to touch his shoulder. That's the perfect setup for BFS: Shrimp tells the first layer (his direct contacts), they tell the second layer (their friends), the second layer tells the third, and so on — until the message reaches its target.

> [!TIP]
> **When to use BFS:**
> - The graph is unweighted (or every edge weight is equal)
> - You want the shortest path, measured in number of hops
>
> **When *not* to use BFS:**
> - The graph has unequal weights
> - You're navigating something like a deep maze (memory risk — BFS keeps an entire "frontier" layer in memory at once)

Two classic real-world BFS problems:

1. **How does LinkedIn know someone is a "2nd-degree connection"?** LinkedIn runs BFS starting from *your* node. Layer 1 is your direct connections, layer 2 is friends-of-friends. If it's searching for a connection path, BFS guarantees the minimum number of people in between.
2. **Reaching the supermarket by the fewest roads, not the shortest distance.** Google Maps' "fewest hops" mode is BFS on an unweighted road graph — it finds the path with the fewest turns/segments, then discards the rest.

**DFS (Depth-First Search)** is different. Imagine falling into a massive hedge maze and wanting to find the exit, or just map the whole thing. You don't clone yourself to check every path in parallel (that's BFS) — you pick one path and follow it all the way to a dead end, then backtrack to the last unexplored branch and go again, repeating until you're done. This uses **LIFO** (Last In, First Out) — a **stack**.

> Think of a stack of dinner plates: wash one, put it down; wash another, stack it on top; wash a third, stack it on top of that. When you need a plate, you grab the one on top — the last one placed.

<img width="618" height="523" alt="image" src="https://github.com/user-attachments/assets/af2b95c8-2306-4580-9192-bbd7bac6de8d" />


> [!TIP]
> **When to use DFS:**
> - You need to explore all the way to the deepest part of the graph
> - You need a specific ordering (topological sort — more on this later)
> - You need to detect hidden cycles (is the graph cyclic or acyclic?)

### BFS in Code

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/53b9c63a-ac37-4c56-b4e8-74652f6c7d24" />


Let's build a graph by hand and run BFS on it:

```python
custom_graph = {
    'A': ['B', 'C', 'H'],   # 'node': ['neighbors']
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B', 'G'],
    'E': ['B', 'G', 'H'],
    'F': ['C', 'I'],
    'G': ['D', 'E', 'J'],
    'H': ['A', 'E', 'J', 'I'],
    'I': ['F', 'H', 'J'],
    'J': ['G', 'H', 'I']
}
# If you're working with a real networkx graph instead of a plain dict,
# swap graph.get(current, []) below for graph.neighbors(current).
```

```python
from collections import deque

def pure_bfs(graph, start_node):
    visited = set()
    queue = deque()
    traversal_order = []

    queue.append(start_node)
    visited.add(start_node)

    print("----STARTING THE EXTRACTION----\n")
    while len(queue) > 0:
        current = queue.popleft()
        traversal_order.append(current)
        print(f"Popped node from front '{current}'. Current queue: {list(queue)}")

        for nb in graph.get(current, []):
            if nb not in visited:
                queue.append(nb)
                visited.add(nb)

    return traversal_order

result = pure_bfs(custom_graph, 'A')
print(f"\nFinal BFS Traversal Order: {result}")

"""
Output:

----STARTING THE EXTRACTION----

Popped node from front 'A'. Current queue: []
Popped node from front 'B'. Current queue: ['C', 'H']
Popped node from front 'C'. Current queue: ['H', 'D', 'E']
Popped node from front 'H'. Current queue: ['D', 'E', 'F']
Popped node from front 'D'. Current queue: ['E', 'F', 'J', 'I']
Popped node from front 'E'. Current queue: ['F', 'J', 'I', 'G']
Popped node from front 'F'. Current queue: ['J', 'I', 'G']
Popped node from front 'J'. Current queue: ['I', 'G']
Popped node from front 'I'. Current queue: ['G']
Popped node from front 'G'. Current queue: []

Final BFS Traversal Order: ['A', 'B', 'C', 'H', 'D', 'E', 'F', 'J', 'I', 'G']
"""
```

Before you hyperventilate, here's the whole thing in plain words:

- **`custom_graph`** — an *adjacency list*. Each key is a node, each value is its list of neighbors.
- **`visited = set()`** — an empty set (like a list, but it can't hold duplicates).
- **`queue = deque()`** — a double-ended queue, supporting:
  ```python
  queue.append('E')      # add to the rear
  queue.appendleft('Z')  # add to the front
  queue.pop()             # remove from the rear
  queue.popleft()          # remove from the front
  ```
- **`while len(queue) > 0:`** — keep going as long as there's still something waiting to be processed.
- **`current = queue.popleft()`** — removes and returns the *leftmost* (frontmost) item.
- **`queue.append(start_node)` / `visited.add(start_node)`** — we seed the queue with the starting node, and mark it visited **immediately**, not when it's later popped. This matters *a lot*: skip this step and the same node can get added to the queue multiple times by different neighbors, all asking "is this visited yet?" before any of them get an answer.
- **`for nb in graph.get(current, []):`** — grabs the neighbor list for `current`. Using `.get(current, [])` instead of `graph[current]` means a missing key returns an empty list instead of crashing.
- **`if nb not in visited:`** — skip anyone we've already met.
- **`queue.append(nb)` / `visited.add(nb)`** — add the neighbor to the back of the line, and mark it visited the instant it's added — not when it's eventually processed.
- **`return traversal_order`** — once the queue empties (everyone reachable from `'A'` has been visited), return the order we visited them in.

<details>
<summary><strong>🔍 Step-by-step queue walkthrough (click to expand)</strong></summary>

We start from `'A'`, with `visited = {'A'}` and `queue = ['B', 'C', 'H']` after A's neighbors are added.

Since BFS is FIFO, new neighbors join the rear, and we always process from the front:

1. Pop **B**. Neighbors: `['A', 'D', 'E']`. `A` is already visited and ignored; `D` and `E` are new → `queue = ['C', 'H', 'D', 'E']`, `visited = {'A', 'B', 'D', 'E'}`.
2. Pop **C**. Neighbors: `['A', 'F']`. `A` is visited; `F` is new → `queue = ['H', 'D', 'E', 'F']`.
3. Pop **H**. Neighbors: `['A', 'E', 'J', 'I']`. `A` and `E` are visited/queued already; `J` and `I` are new → `queue = ['D', 'E', 'F', 'J', 'I']`.
4. ...and so on, until the queue empties and we land on the traversal order shown above.

</details>

### DFS in Code

Good news: the DFS code is nearly identical to BFS — just swap the queue for a stack. (Yes, after all that time spent with Airi Sezaki, we're right back to another traversal problem. At least this one terminates — no A → B → C → A loop here.)

```python
def dfs(graph, start_node):
    visited = set()
    stack = []
    traversal_order = []

    stack.append(start_node)
    visited.add(start_node)

    print("----STARTING THE EXTRACTION----\n")
    while len(stack) > 0:
        current = stack.pop()
        traversal_order.append(current)
        print(f"Popped node from top '{current}'. Current stack: {list(stack)}")

        for nb in graph.get(current, []):
            if nb not in visited:
                stack.append(nb)
                visited.add(nb)

    return traversal_order

result = dfs(custom_graph, 'A')
print(f"\nFinal DFS Traversal Order: {result}")

"""
Output:

----STARTING THE EXTRACTION----

Popped node from top 'A'. Current stack: []
Popped node from top 'H'. Current stack: ['B', 'C']
Popped node from top 'I'. Current stack: ['B', 'C', 'E', 'J']
Popped node from top 'F'. Current stack: ['B', 'C', 'E', 'J']
Popped node from top 'J'. Current stack: ['B', 'C', 'E']
Popped node from top 'G'. Current stack: ['B', 'C', 'E']
Popped node from top 'D'. Current stack: ['B', 'C', 'E']
Popped node from top 'E'. Current stack: ['B', 'C']
Popped node from top 'C'. Current stack: ['B']
Popped node from top 'B'. Current stack: []

Final DFS Traversal Order: ['A', 'H', 'I', 'F', 'J', 'G', 'D', 'E', 'C', 'B']
"""
```

<details>
<summary><strong>🔍 Step-by-step stack walkthrough (click to expand)</strong></summary>

We start at `'A'`. `stack = ['B', 'C', 'H']` after A's neighbors are pushed — but since a stack is a pile of plates, we grab whatever's on **top**, which is `H`.

1. Pop **H**. Neighbors: `['A', 'E', 'J', 'I']`. `A` is visited; `E`, `J`, `I` are new → `stack = ['B', 'C', 'E', 'J' , 'I']`. Top is now `I`.
2. Pop **I**. Neighbors: `['F', 'H', 'J']`. `H` visited, `J` already queued; `F` is new → `stack = ['B', 'C', 'E', 'J', 'F']`. Top is `F`.
3. Pop **F**. Neighbors: `['C', 'I']`. Both already accounted for → `stack = ['B', 'C', 'E', 'J']` unchanged. Top is now `J`.
4. Pop **J**. Neighbors: `['G', 'H', 'I']`. `H`, `I` visited; `G` is new → `stack = ['B', 'C', 'E', 'G']`.
5. ...and so on, diving deep before ever backtracking, until we land on the traversal order shown above.

</details>


---

## 9. The Final Bosses of the Month

- **Dijkstra**
- **Degree Centrality** (easy — revisited with the formula)
- **Connected Components**
- **Cycle Detection**
- **Topological Sort**
- **PageRank**
- **Betweenness Centrality**

We'll go through these one at a time. Fair warning: prepare the math, because we're about to crunch numbers like actual mathematicians. No stress though — we'll tackle them together, baby steps and all.

---

## 10. Dijkstra's Algorithm

**What is it for?** Finding the shortest path (the minimum total edge weight) from a single starting node to every other node, on a weighted graph.

**Requirement:** every edge weight must be **non-negative** (`weight >= 0`). A single negative weight breaks the entire algorithm. The graph can be directed or undirected, cyclic or acyclic — the one hard rule is non-negative weights.

Here's the graph we'll trace by hand:

<img width="1400" height="687" alt="image" src="https://github.com/user-attachments/assets/2d53ffa3-4c1b-4db6-add6-4d53a04c34e4" />


| Edge | Weight | Edge | Weight |
|---|---|---|---|
| A–B | 4 | D–F | 14 |
| A–H | 8 | E–F | 10 |
| B–C | 8 | F–G | 2 |
| B–H | 11 | G–H | 1 |
| C–D | 7 | G–I | 6 |
| C–F | 4 | H–I | 7 |
| C–I | 2 | | |
| D–E | 9 | | |

We start with everything at infinity except A itself:

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | **0** | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ |
| Parent | — | — | — | — | — | — | — | — | — |

**From A**, we can reach B (cost 4) or H (cost 8):

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | **4** | ∞ | ∞ | ∞ | ∞ | ∞ | **8** | ∞ |
| Parent | — | **A** | — | — | — | — | — | **A** | — |

The **parent** column stores the previous node on the current cheapest known path — so the cheapest way to B, right now, is via A, at cost 4.

We always move to the *unvisited* node with the smallest tentative distance. That's **B (4)**.

**From B**, we can reach H or C:
- B → H = 4 + 11 = 15. Current H is 8. 15 > 8 → no update, ignore.
- B → C = 4 + 8 = 12. C is empty → update.

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | 4 | **12** | ∞ | ∞ | ∞ | ∞ | 8 | ∞ |
| Parent | — | A | **B** | — | — | — | — | A | — |

Smallest unvisited now: **H (8)**. From H we can reach B, I, G:
- H → B = 8 + 11 = 19 > 4 → no update.
- H → I = 8 + 7 = 15. I is empty → update.
- H → G = 8 + 1 = 9. G is empty → update.

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | 4 | 12 | ∞ | ∞ | ∞ | **9** | 8 | **15** |
| Parent | — | A | B | — | — | — | **H** | A | **H** |

Smallest unvisited: **G (9)**. From G we can reach I, F:
- G → I = 9 + 6 = 15 — ties the existing value, so no update.
- G → F = 9 + 2 = 11. F is empty → update.

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | 4 | 12 | ∞ | ∞ | **11** | 9 | 8 | 15 |
| Parent | — | A | B | — | — | **G** | H | A | H |

Smallest unvisited: **F (11)**. From F we can reach C, D, E:
- F → C = 11 + 4 = 15 > current 12 → no update.
- F → D = 11 + 14 = 25. D is empty → update.
- F → E = 11 + 10 = 21. E is empty → update.

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | 4 | 12 | **25** | **21** | 11 | 9 | 8 | 15 |
| Parent | — | A | B | **F** | **F** | G | H | A | H |

Smallest unvisited: **C (12)**. From C we can reach I, F, D:
- C → I = 12 + 2 = 14 < 15 → update, parent changes too.
- C → F = 12 + 4 = 16 > 11 → no update.
- C → D = 12 + 7 = 19 < 25 → update.

| Node | A | B | C | D | E | F | G | H | I |
|---|---|---|---|---|---|---|---|---|---|
| Distance | 0 | 4 | 12 | **19** | 21 | 11 | 9 | 8 | **14** |
| Parent | — | A | B | **C** | F | G | H | A | **C** |

From here the algorithm keeps going the exact same way until every node is processed. Now, to read off the cheapest path to any node, just walk the parent chain backward:

- **Node E:** parent is F (F → E), F's parent is G (G → F → E), G's parent is H, H's parent is A → **A → H → G → F → E**
- **Node D:** parent is C (C → D), C's parent is B (B → C → D), B's parent is A → **A → B → C → D**

And in code, this entire hand-worked table collapses into one function call:

```python
import networkx as nx

G = nx.Graph()

# Format: (Node_1, Node_2, weight)
edges = [
    ('A', 'B', 4), ('A', 'H', 8),
    ('B', 'C', 8), ('B', 'H', 11),
    ('C', 'D', 7), ('C', 'F', 4), ('C', 'I', 2),
    ('D', 'E', 9), ('D', 'F', 14),
    ('E', 'F', 10),
    ('F', 'G', 2),
    ('G', 'H', 1), ('G', 'I', 6),
    ('H', 'I', 7)
]
G.add_weighted_edges_from(edges)

distances, paths = nx.single_source_dijkstra(G, source='A', weight='weight')

for target in sorted(G.nodes()):
    print(f"{target} | {distances[target]} | {' -> '.join(paths[target])}")

"""
Output:

A | 0 | A
B | 4 | A -> B
C | 12 | A -> B -> C
D | 19 | A -> B -> C -> D
E | 21 | A -> H -> G -> F -> E
F | 11 | A -> H -> G -> F
G | 9 | A -> H -> G
H | 8 | A -> H
I | 14 | A -> B -> C -> I
"""
```

- **`G = nx.Graph()`** — creates an empty graph. An empty notebook page: no nodes, no edges, just blank potential.
- **`G.add_weighted_edges_from(edges)`** — dumps all our `(node, node, weight)` tuples in at once.
- **`nx.single_source_dijkstra(G, source='A', weight='weight')`** — this one line does *everything* we just did by hand with the table and the infinity symbols. `source='A'` means "start counting from A, just like we did." `weight='weight'` tells it to use the stored edge weight for cost, not just hop-count. It hands back **two** dictionaries at once: `distances` (cheapest cost to every node from A) and `paths` (the actual route, as a list of nodes, to get there cheapest — exactly what we just derived by hand).


---

## 11. Degree Centrality

We already met this one informally — it's the simplest way to gauge how important a node is, just by counting how many connections (its **degree**) it has. Now let's look at the actual formula:

$$\text{Degree Centrality}(v) = \frac{\text{Degree}(v)}{N - 1}$$

We need two things: the node's **Degree**, and the **Maximum Possible Connections** (`N - 1`, since a node can't connect to itself).

Take an 8-node graph as an example:

<img width="640" height="368" alt="image" src="https://github.com/user-attachments/assets/8955f638-d59d-4fe3-a273-bbd156d3cfa7" />


- **Node A** connects only to B → Degree = 1. Max possible connections = 8 − 1 = 7. So `Degree Centrality(A) = 1/7 ≈ 0.143`. Translation: A is nearly irrelevant — remove it, and almost nothing changes.
- **Node E** has 4 connections out of a possible 7 → `Degree Centrality(E) = 4/7 ≈ 0.571`. E is genuinely important: a signal starting there reaches more than half the reachable network directly.

In code:

```python
import networkx as nx

dc = nx.degree_centrality(G)

for node in sorted(G.nodes()):
    print(f"{node} | degree {G.degree(node)} | {G.degree(node)} / 7 = {dc[node]:.4f}")
    # The "/ 7" here is specific to this 8-node example.
    # For any other graph, just divide by (number_of_nodes - 1).
```

Moving on — that one was too easy to dwell on.

---

## 12. Connected Components

A **connected component** is a group of nodes that are all reachable from each other, but completely cut off from any other group of nodes elsewhere in the graph.

<img width="640" height="368" alt="image" src="https://github.com/user-attachments/assets/5ac36de9-db6c-4acb-b3a6-d2bfd9430ebb" />


```python
import networkx as nx

G = nx.Graph()

edges = [
    ('A', 'B'), ('B', 'C'),
    ('D', 'E'), ('D', 'H'),
    ('E', 'F'), ('E', 'G'), ('E', 'H'),
    ('F', 'G'),
    ('G', 'H')
]
G.add_edges_from(edges)

print(f"Is the graph connected? {nx.is_connected(G)}")
# False — {A, B, C} has no path to {D, E, F, G, H}.

print(f"Number of components: {nx.number_connected_components(G)}")
# 2

components = list(nx.connected_components(G))
for component in components:
    print(f"Component: {component}")
```

- **`nx.connected_components(G)`** returns a *generator*, so we wrap it in `list()` to actually store all the components:
  - First component: `{'A', 'B', 'C'}`
  - Second component: `{'D', 'E', 'F', 'G', 'H'}`

---

## 13. Cycle Detection

A cycle is exactly what it sounds like: a loop. Start at a node, follow a path of edges, and somehow arrive back where you started — that's a cycle (e.g. A → B → C → A).

<img width="640" height="368" alt="image" src="https://github.com/user-attachments/assets/2352e533-ce69-4eb5-a0c6-e34d2098e668" />


```python
import networkx as nx

edges = [
    ('A', 'B'), ('B', 'C'),
    ('D', 'E'), ('D', 'H'),
    ('E', 'F'), ('E', 'G'), ('E', 'H'),
    ('F', 'G'),
    ('G', 'H')
]

G = nx.Graph()
G.add_edges_from(edges)

cycles = nx.cycle_basis(G)

print(f"The total number of cycles found: {len(cycles)}")
for index, cycle in enumerate(cycles, 1):
    print(f"Cycle {index}: {cycle}")

"""
Output:

The total number of cycles found: 3
Cycle 1: ['E', 'H', 'G']
Cycle 2: ['E', 'D', 'H']
Cycle 3: ['E', 'F', 'G']
"""
```

`nx.cycle_basis()` finds the smallest set of *independent* cycles that can describe every other cycle in the graph — it doesn't list every possible loop, only enough of them to fully characterize the graph's "loopiness." If A → B → C → A is a cycle, there's no point also listing B → C → A → B or C → A → B → C — same loop, different starting point.

Note there's no `A → B → C` cycle here: A, B, C form a simple chain with no edge closing the loop back to A, so it's *not* cyclic at all.

That's the whole idea. Next up: sorting the mess into order.

---

## 14. Topological Sort

A **topological sort** arranges a graph into a single valid linear order — e.g. A → B → C → D → E → F → G — but it only works on **Directed Acyclic Graphs (DAGs)**. If the graph has a loop, or if it's undirected, this doesn't apply.

Think of the nodes as tasks: if there's an edge A → B, you cannot start B before finishing A.

<img width="640" height="368" alt="image" src="https://github.com/user-attachments/assets/e4edeb54-59e1-4dfb-824b-1d79ffd7a1a7" />

Given a web of dependencies, the question becomes: *what's a valid order to complete every task without ever breaking a dependency?* That's exactly what topological sort answers.

```python
import networkx as nx

# nx.Graph() gives an undirected graph — we need nx.DiGraph() for a directed one.
DAG = nx.DiGraph()

dependencies = [
    ('A', 'B'), ('A', 'H'),
    ('B', 'C'),
    ('C', 'D'), ('C', 'I'),
    ('H', 'G'), ('H', 'I'),
    ('G', 'F'),
    ('I', 'F'),
    ('D', 'E'),
    ('F', 'E')
]
DAG.add_edges_from(dependencies)

is_valid_dag = nx.is_directed_acyclic_graph(DAG)
print(f"Is this graph a valid DAG? {is_valid_dag}")

ordered_workflow = list(nx.topological_sort(DAG))

print("\n=== Valid Execution Order ===")
print(' -> '.join(ordered_workflow))

"""
Output:

Is this graph a valid DAG? True

=== Valid Execution Order ===
A -> B -> H -> C -> G -> D -> I -> F -> E
"""
```

That order respects every single dependency we listed: A before B, A before H, B before C, C before D and I, H before G and I, G before F, I before F, D before E, F before E. Nothing shows up before something it depends on.

- **`nx.is_directed_acyclic_graph(DAG)`** — checks whether the graph is, in fact, a valid DAG.
- **`list(nx.topological_sort(DAG))`** — produces one valid linear ordering that respects every dependency edge.


---

## 15. PageRank

PageRank measures a node's importance not by counting how many links it has, but by looking at **who** is linking to it.

> [!NOTE]
> This one is brain-tiring. If you're reading this at a late hour, maybe go to sleep first — I nearly went nuts writing it. But let's see if it clicks in five minutes anyway.

The core idea: imagine two kids — a popular one (Kevin) and a smart one (Michael). Kevin is recommended by 100 kids, all fairly average. Michael is recommended by 6 professors and the school president. Under **Degree Centrality**, Kevin wins easily — he has more raw connections. Under **PageRank**, Michael wins — because the *7* people vouching for him carry serious weight.

The formula (scary-looking, stay with me):

$$PR(u) = \frac{1 - d}{N} + d \sum_{v \in In(u)} \frac{PR(v)}{L(v)}$$

- **$PR(u)$** — the PageRank score of the node $u$ we're solving for.
- **$N$** — the total number of nodes in the graph.
- **$d$** — the *damping factor*, usually `0.85`.

**Why 0.85?** Picture 12 stores in a mall, each starting with equal foot-traffic points. Store A's crowd splits between B and C. Store G has doors leading in from 6 other stores, so it racks up points fast. Meanwhile Stores J and K only have doors leading to *each other* — an isolated loop nobody finds on purpose.

> [!WARNING]
> If `d = 1`, disaster: shoppers who wander into the J–K loop get trapped forever, bouncing between the two. Over time *everyone* ends up stuck there, G empties out, A drops to zero, and two nobody-stores steal 100% of the mall's traffic. Completely unrealistic.

That's why `d = 0.85`: at every store, there's an 85% chance someone follows the doors normally, and a 15% chance they get bored, leave entirely, and teleport to a *random* store anywhere in the mall. That 15% "boredom leak" constantly frees the J–K shoppers and redistributes them elsewhere, so the mall's traffic stabilizes into a realistic ranking: **G** on top (six pipelines feed it), **B** and **C** next (steady stream from A), and **J**/**K** rock-bottom — forgotten even by their own owners.

- $\frac{1-d}{N}$ — the "boredom leak," split evenly across all `N` stores. Every single node gets this same tiny guaranteed baseline, popular or not.
- $In(v)$ — the set of nodes with an edge pointing *into* $v$.

<img width="640" height="368" alt="image" src="https://github.com/user-attachments/assets/875ca96e-875e-4da8-8204-786dd9c38ba9" />


- $PR(v)$ — the PageRank of each neighbor feeding into our target node.
- $L(v)$ — **not** how many people flow *into* $v$ — this is how many outgoing doors $v$ itself has. If Store A splits its crowd between B and C, $L(A) = 2$: A only has 2 outgoing doors, so it must split its points evenly between them. That's why we divide $PR(v)$ by $L(v)$ before adding it in — a node never hands its *entire* score to a single neighbor, only a fair share.

### Worked Example: Methane

<img width="577" height="539" alt="image" src="https://github.com/user-attachments/assets/9932b070-e8f3-4dcf-a794-0ab8cd0a2f00" />


Let's solve something practical — chemistry. A methane molecule is one Carbon atom bonded to four Hydrogen atoms — structurally, a "star" graph with Carbon at the center.

$$PR(C) = \frac{1 - d}{N} + d \sum_{v \in In(u)} \frac{PR(v)}{L(v)}$$

With $d = 0.85$ and $N = 5$ (5 atoms total): $\frac{0.15}{5} = 0.0300$.

$$PR(C) = 0.0300 + 0.85 \left( \frac{PR(H_1)}{L(H_1)} + \frac{PR(H_2)}{L(H_2)} + \frac{PR(H_3)}{L(H_3)} + \frac{PR(H_4)}{L(H_4)} \right)$$

Each Hydrogen has exactly **one** connection (back to Carbon), so $L(H_1) = L(H_2) = L(H_3) = L(H_4) = 1$:

$$PR(C) = 0.0300 + 0.85 \left( PR(H_1) + PR(H_2) + PR(H_3) + PR(H_4) \right)$$

If we seed every atom equally at $\frac{1}{N} = 0.2000$ and plug that in for one pass:

$$PR(C) = 0.0300 + 0.85(0.2 + 0.2 + 0.2 + 0.2) = 0.0300 + 0.68 = \mathbf{0.71}$$

Pull up your phone, call Newton, Einstein, Archimedes, and Gauss — after a month of arguing in Latin, they'd confirm: **0.71**.

> [!IMPORTANT]
> Small catch, though: that `0.71` is only the result of **one pass** through the formula, starting from an equal guess for everyone. PageRank isn't a one-shot calculation — it's a fixed-point algorithm. You take the freshly computed scores, feed them back into the *same* formula for every node (Hydrogens included), and repeat until the numbers stop moving. `nx.pagerank()` does exactly that under the hood, dozens of times, not just once.
>
> Run it all the way to convergence on this exact star graph, and Carbon actually settles at **≈ 0.4757 (47.6%)**, with each Hydrogen landing around **0.1311 (13.1%)** — and yes, `0.4757 + 4 × 0.1311 ≈ 1.0`, as it should. Carbon is still comfortably the most important atom (more than 3.5× any single Hydrogen) — it's just not quite the 71%-dominant powerhouse that first pass made it look like.

In code — same pattern as the Shrimp example from earlier:

```python
pagerank = nx.pagerank(G)

print(f"The PageRank of Carbon is: {pagerank['C']:.4f}")

"""
Output:

The PageRank of Carbon is: 0.4757
"""
```

Bottom line to remember: **PageRank looks at *who* links to a node, not how many nodes link to it.**


---

## 16. Betweenness Centrality

The final boss of the month — and by most accounts, the one harder in concept and math than PageRank. Clench your teeth.

Remember Shrimp's low betweenness score? Not because his friends weren't important (that's PageRank), but because everyone else could already reach each other without going through him. (Message from the future: it's not as bad as it looks.)

**Betweenness Centrality** doesn't care how important your friends are, or how many you have. It asks one question: *if every node sends a message to every other node along the shortest possible path, how many of those paths are forced to pass through you?* If you're the sole bridge between two groups of 100 people each, your score skyrockets.

The formula looks messy — don't be fooled, it's not as bad as it looks:

$$C_{B}(v)=\sum _{s\ne v\ne t}\frac{\sigma _{st}(v)}{\sigma _{st}}$$

- **$v$** — the node we're targeting.
- **$\sum_{s \neq v \neq t}$** — sum over every pair of nodes $(s, t)$, excluding the target node $v$ itself from either role.
- **$\sigma_{st}$** — the total number of shortest paths between a chosen source $s$ and target $t$.
- **$\sigma_{st}(v)$** — how many of those shortest paths are forced to pass directly through $v$.

### Worked Example

<img width="644" height="410" alt="image" src="https://github.com/user-attachments/assets/3cf2c5ab-a1b1-4d6b-b15f-8abcc3151ca8" />


Say we pick **D** as our target node, out of 8 total nodes (A, B, C, D, E, F, G, H). Excluding D leaves 7 nodes, and the number of unique pairs among them is:

$$\text{Total Pairs} = \frac{N \times (N - 1)}{2} = \frac{7 \times 6}{2} = 21$$


We evaluate the path fraction for all 21 pairs and sum them:

| Pair | Shortest path(s) | Contribution |
|---|---|---|
| A,B | Direct edge — bypasses D | 0/1 = 0 |
| A,C | Direct edge — bypasses D | 0/1 = 0 |
| A,E | Direct edge — bypasses D | 0/1 = 0 |
| A,F | A → E → F — bypasses D | 0/1 = 0 |
| A,G | A → C → G — bypasses D | 0/1 = 0 |
| A,H | A → C → H — bypasses D | 0/1 = 0 |
| B,C | B → A → C — bypasses D | 0/1 = 0 |
| B,E | Two tied paths: B→A→E and B→D→E | 1/2 = 0.5 |
| B,F | Unique path B → D → F | 1/1 = 1 |
| B,G | Unique path B → D → G | 1/1 = 1 |
| B,H | Unique path B → D → H | 1/1 = 1 |
| C,E | C → A → E — bypasses D | 0/1 = 0 |
| C,F | C → G → F — bypasses D | 0/1 = 0 |
| C,G | Direct edge — bypasses D | 0/1 = 0 |
| C,H | Direct edge — bypasses D | 0/1 = 0 |
| E,F | Direct edge — bypasses D | 0/1 = 0 |
| E,G | Two tied paths: E→D→G and E→F→G | 1/2 = 0.5 |
| E,H | Unique path E → D → H | 1/1 = 1 |
| F,G | Direct edge — bypasses D | 0/1 = 0 |
| F,H | Unique path F → D → H | 1/1 = 1 |
| G,H | Two tied paths: G→D→H and G→C→H | 1/2 = 0.5 |

Summing every non-zero contribution:

$$0.5 + 1 + 1 + 1 + 0.5 + 1 + 1 + 0.5 = 6.5$$

$$C_B(D) = \frac{6.5}{21} \approx 0.3095$$

Node D single-handedly sits on roughly **31%** of all optimal traffic in the graph. Not bad for one node.

```python
betweenness = nx.betweenness_centrality(G)

print(f"D's Betweenness Centrality: {betweenness['D']:.4f}")

"""
Output:

D's Betweenness Centrality: 0.3095
"""
```

To see every node's score at once:

```python
for index, (node, bc) in enumerate(betweenness.items(), 1):
    print(f"{index}. Betweenness of {node}: {bc:.4f}")
```

> [!WARNING]
> A dict remembers its keys and values together, but looping `for index, bc in enumerate(betweenness, 1):` (without `.items()`) only ever hands you the **node names**, not the scores — `bc` would silently hold something like `'D'`, not `0.3095`. Always unpack `.items()` when you need both.


---

## 17. Final Project — Progetto Finale

Time to put everything together. Here's the graph we'll use, and the code to generate it:

<img width="640" height="480" alt="Graph_end" src="https://github.com/user-attachments/assets/a36d4814-c6bb-430a-9850-e33297e32cd7" />


```python
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

edges = [
    ("Central", "Park"),
    ("Central", "Museum"),
    ("Park", "University"),
    ("Museum", "University"),
    ("University", "Airport"),
    ("University", "Stadium"),
    ("Airport", "Shopping"),
    ("Stadium", "Shopping"),
]

G = nx.Graph()
G.add_edges_from(edges)

# Draw it, if you want to see the figure:
nx.draw(G, node_color='red', edge_color='black', node_size=1000, with_labels=True)
plt.show()


def bfs(graph, start_node):
    visited = set()
    queue = deque()
    traversal_order = []

    visited.add(start_node)
    queue.append(start_node)

    while len(queue) > 0:
        current = queue.popleft()
        traversal_order.append(current)

        for nb in graph.neighbors(current):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return traversal_order


result = bfs(G, 'Central')
print(f"\nFinal BFS Traversal Order: {result}")


def dfs(graph, start_node):
    visited = set()
    stack = []
    traversal_order = []

    visited.add(start_node)
    stack.append(start_node)

    while len(stack) > 0:
        current = stack.pop()
        traversal_order.append(current)

        for nb in graph.neighbors(current):
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)

    return traversal_order


result = dfs(G, 'Central')
print(f"\nFinal DFS Traversal Order: {result}")

cycles = nx.simple_cycles(G)

print("----Cycles----\n")
for index, cycle in enumerate(cycles, 1):
    print(f"{index} Cycle: {' -> '.join(cycle)}")

print(f"Is the graph connected? {nx.is_connected(G)}")
print(f"Number of components: {nx.number_connected_components(G)}")

"""
Output:

<Image of the graph>

Final BFS Traversal Order: ['Central', 'Park', 'Museum', 'University', 'Airport', 'Stadium', 'Shopping']

Final DFS Traversal Order: ['Central', 'Museum', 'University', 'Stadium', 'Shopping', 'Airport', 'Park']

----Cycles----

1 Cycle: Central -> Park -> University -> Museum
2 Cycle: University -> Airport -> Shopping -> Stadium

Is the graph connected? True
Number of components: 1
"""
# For reaching 'Shopping' specifically, DFS gets there in fewer steps here.
```

### Now Let's Weight It

Same city map, but every road now has a real travel cost:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6ca6aaaf-f662-4181-98d2-94d78c9c670a" />


```python
import networkx as nx

weighted_edges = [
    ("Central", "Park", 2),
    ("Central", "Museum", 4),
    ("Park", "University", 5),
    ("Museum", "University", 1),
    ("University", "Airport", 6),
    ("University", "Stadium", 3),
    ("Airport", "Shopping", 2),
    ("Stadium", "Shopping", 4)
]
G = nx.Graph()
G.add_weighted_edges_from(weighted_edges)

distances, paths = nx.single_source_dijkstra(G, source='Central', weight='weight')

for target in G.nodes():
    print(f"{target} | {distances[target]} | {' -> '.join(paths[target])}")

degree_centrality = nx.degree_centrality(G)
page_rank = nx.pagerank(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

print("\n----Total Stats----\n")

for node, score in degree_centrality.items():
    print(f"The degree centrality of {node} | {score:.4f}")

print()
for node, score in page_rank.items():
    print(f"The importance of {node} (due to neighbors) | {score}")

print()
for node, score in betweenness_centrality.items():
    print(f"How much of a gatekeeper is {node} | {score}")

"""
Output:

Central | 0 | Central
Park | 2 | Central -> Park
Museum | 4 | Central -> Museum
University | 5 | Central -> Museum -> University
Airport | 11 | Central -> Museum -> University -> Airport
Stadium | 8 | Central -> Museum -> University -> Stadium
Shopping | 12 | Central -> Museum -> University -> Stadium -> Shopping

----Total Stats----

The degree centrality of Central    | 0.3333
The degree centrality of Park       | 0.3333
The degree centrality of Museum     | 0.3333
The degree centrality of University | 0.6667
The degree centrality of Airport    | 0.3333
The degree centrality of Stadium    | 0.3333
The degree centrality of Shopping   | 0.3333

The importance of Central (due to neighbors)    | 0.12554954818573194
The importance of Park (due to neighbors)       | 0.12913469450799694
The importance of Museum (due to neighbors)     | 0.10700042117576271
The importance of University (due to neighbors) | 0.2545872696474754
The importance of Airport (due to neighbors)    | 0.1403089331311898
The importance of Stadium (due to neighbors)    | 0.1293488984221484
The importance of Shopping (due to neighbors)   | 0.11407023492969458

How much of a gatekeeper is Central    | 0.03333333333333333
How much of a gatekeeper is Park       | 0.0
How much of a gatekeeper is Museum     | 0.26666666666666666
How much of a gatekeeper is University | 0.6333333333333333
How much of a gatekeeper is Airport    | 0.0
How much of a gatekeeper is Stadium    | 0.26666666666666666
How much of a gatekeeper is Shopping   | 0.06666666666666667
"""
```

> [!NOTE]
> **Park scores a flat `0.0` on betweenness** — nobody's shortest path is ever forced through it. Museum → University is a straight, cheap shot (cost 1), while Park → University costs 5. Given the choice, every rational route prefers Museum. Same story for Airport: once University → Stadium → Shopping is cheaper, Airport stops being anyone's forced gateway. A textbook toxic relationship: the shortcut wins, and the "prettier" route gets ignored.

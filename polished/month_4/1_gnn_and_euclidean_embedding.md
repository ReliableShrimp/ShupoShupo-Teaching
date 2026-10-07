Well, we're already mentally unstable from the embedding mess, yet now we're going to study some GNNs (which are really useful, and one of the main resources of this plan), along with simple embeddings (in Euclidean space — which we'll do in Python) and how we use them.

Disclaimer: this topic will be long, because I took several months of GNN material and packed it into a single month's worth of markdown. Besides that, we're going full-GNN here, so in the next part there will be even more advanced topics.

# Chapter 1. Euclidean Embeddings Basics

Table of contents:
1. Word2Vec intuition: Skip-Gram
2. DeepWalk
3. Node2Vec
4. Implement Node2Vec on the Karate Club graph; visualize with t-SNE

## 1. Word2Vec

We already know what an embedding is, so I won't repeat that — instead let's focus on an interesting question:
"How can a model learn that two things should have a similar vector, just from the things that appear around them?"

Interesting question — so let's forget about graphs for a second and look at some sentences:
```
the cat eats fish
the cat likes milk
the dog eats fish
the dog likes meat
```

Suppose we choose the word `cat`, with a context window of size `2`.

(Before continuing, let me explain what a "context window of size `n`" is. It means looking up to `n` words to the left and `n` words to the right. For example, with corpus word `cat`:

2 words to the left: `["the"]` (only 1 word is available before "cat")
2 words to the right: `["eats", "fish"]`
Context words extracted: `["the", "eats", "fish"]`)

So our training examples start looking like:
```
(target, context)

(cat, the)
(cat, eats)

(cat, the)
(cat, likes)

(dog, the)
(dog, eats)

(dog, the)
(dog, likes)
...
```

So, Skip-Gram takes the target word and tries to predict the words appearing around it.

Now, say every node gets a vector:
```
cat  → [ 0.21, -0.73,  0.44, ... ]
dog  → [ 0.18, -0.69,  0.41, ... ]
fish → [-0.52,  0.31,  0.77, ... ]
```

You don't tell the model that:
```
cat ≈ dog
```
or that:
```
cat = animal
```
Instead, you repeatedly give it relationships like:
```
cat → the
cat → eats
cat → likes

dog → the
dog → eats
dog → likes
```

Through optimization, the model discovers that cat and dog have similar representations, since they occur in similar contexts. This is an important idea: similarity can emerge purely from similar interaction patterns.

Let:

$`w_t`$ = the target word
$`w_c`$ = a context word
$`v_{w_t}`$ = vector associated with the target
$`u_{w_c}`$ = vector associated with the context

Notice that Word2Vec maintains 2 internal vector representations:
```
target/input vectors       context/output vectors

v_cat                       u_the
v_dog                       u_eats
v_fish                      u_likes
...
```

We'll take:
```
cat, eats
```

We want to turn raw scores into probabilities, using:

```math
P(w_c \mid w_t) = \frac{\exp(v_{w_t}^\top u_{w_c})}{\sum_{w \in \mathcal{V}} \exp(v_{w_t}^\top u_w)}
```

The first part,

```math
P(w_c | w_t)
```

basically means: "what is the probability that the word $`w_c`$ is a context word for our target word $`w_t`$?"

So if `cat` is the target and `eats` the context word, the formula asks: "how likely is 'eats' to be a context word when the target is 'cat'?"

Let's work through an example. Say we choose `cat` as target and `eats`, `meat` as context words (using $`d=3`$ for simplicity):

$`v_{\text{cat}} = [0.85, -0.12, 0.44]`$ (target vector for "cat")
$`u_{\text{eats}} = [0.78, -0.05, 0.51]`$ (context vector for "eats")
$`u_{\text{meat}} = [-0.10, 0.92, -0.60]`$ (context vector for "meat")

Using:

```math
P(w_c \mid w_t) = \frac{\exp(v_{w_t}^\top u_{w_c})}{\sum_{w \in \mathcal{V}} \exp(v_{w_t}^\top u_w)}
```

the dot product $`v_{w_t}^\top u_{w_c}`$ tells us how compatible target $`w_t`$ is with context word $`w_c`$. So we get:

```math
v_{\text{cat}}^\top u_{\text{eats}} = (0.85 \cdot 0.78) + (-0.12 \cdot -0.05) + (0.44 \cdot 0.51) \approx 0.893
```

Since `eats` frequently co-occurs with `cat`, the gradient pulls $`v_{cat}`$ and $`u_{eats}`$ closer together in the vector space.

While:

```math
v_{\text{cat}}^\top u_{\text{meat}} = (0.85 \cdot -0.10) + (-0.12 \cdot 0.92) + (0.44 \cdot -0.60) \approx -0.459
```

Since `meat` never appears near `cat` in the corpus, $`v_{cat}`$ and $`u_{meat}`$ get pushed *apart* in the vector space.

So, after many iterations:
```
cat
 │
 ├── the
 ├── eats
 └── likes

dog
 │
 ├── the
 ├── eats
 └── likes
```

the model discovers the structural similarity.

Now that we understand the idea of Word2Vec, let's learn how to use it in Python!

First, let's install it:
```shell
pip install gensim
```

We can verify it installed correctly:
```python
import gensim

print(gensim.__version__)
```

Now let's take some sentences — the ones from before:
```python
sentences = [
    ["the", "cat", "eats", "fish"],
    ["the", "cat", "likes", "milk"],
    ["the", "dog", "eats", "fish"],
    ["the", "dog", "likes", "meat"],
    ["the", "cat", "plays", "outside"],
    ["the", "dog", "plays", "outside"],
]
```

As you can see, we deliberately created similar contexts:
```
cat → eats, likes, plays
dog → eats, likes, plays
```

so we expect the embeddings to come out similar.

The Python code:
```python
from gensim.models import Word2Vec

model = Word2Vec(sentences=sentences, vector_size=32, window=2, min_count=1, sg=1, workers=4, epochs=100,)
```

That's the whole training pipeline for Skip-Gram.

Now let's understand what we just set up.

- `sentences = sentences`:

This is the training corpus. Gensim expects something like:
```
[
    ["the", "cat", "eats", "fish"],
    ["the", "dog", "eats", "fish"],
    ...
]
```
since it takes tokenized sentences, not full raw sentences.

- `vector_size`:

This sets our embedding's dimension — typically something like:
```
50
100
200
300
```
but it depends heavily on dataset size, compute budget, downstream task, and so on.

- `window`:

Our context window, as already explained. So if we have:
```
the cat eats fish
```
with target `cat` and `w = 1`, we'd get:
```
the
   \
   cat
   /
eats
```
and even further tokens with a bigger window.

- `sg`:

This toggles Skip-Gram. Let's see what `sg = 1` vs. `sg = 0` do.

With `sg = 1`, we work the normal way: we get a target word and context words, and Skip-Gram tries to predict the surrounding words:
```
Target                   Context Words
           ┌────────────►    "the"
  "cat" ───┤
           └────────────►    "eats"
```

What about `sg = 0`? This is what we call Continuous Bag of Words (CBOW) — the opposite direction. We give the model a sentence and try to predict the single target word in the middle, using many context words:
```
Context Words              Target
  "the"  ──┐
           ├─► [ Average / Sum ] ──►  "cat"
  "eats" ──┘
```

> [!NOTE]
> Skip-Gram predicts outward (target → context), CBOW predicts inward (context → target). CBOW basically averages or sums the context vectors $`u_{\text{the}} + u_{\text{eats}}`$ to predict $`v_{\text{cat}}`$.

- `min_count`:

This tells Gensim to drop words appearing fewer than $`n`$ times. If we choose `min_count = 1`, we're telling the model not to discard words appearing at least 1 time. For example:
```
I really, really like Yuris
```
If we choose `min_count = 2` instead, we'd get:

| Token | Frequency | Kept (`min_count = 2`) | Status |
| --- | --- | --- | --- |
| `really` | 2 | Yes | Kept |
| `I` | 1 | No | Pruned |
| `like` | 1 | No | Pruned |
| `Yuris` | 1 | No | Pruned |

We'd end up with just `really, really`, and everything else gets discarded.

Now let's look at what we can print:
```python
print("Vocabulary:")
print(model.wv.index_to_key)
# Prints the vocabulary the model learned.

print("\nCat embedding:")
print(model.wv["cat"])
# Prints the embedding of the word 'cat', something like:
# [ 0.012 ... -0.284 ... ]

print("\nCat vector shape:")
print(model.wv["cat"].shape)
# Prints the dimension of our vector — if we chose d=128, we'd see 128.

print("\nCat ↔ Dog similarity:")
print(model.wv.similarity("cat", "dog"))
# Since cat and dog share similar contexts, we expect relatively high cosine similarity.

print("\nWords similar to cat:")
print(model.wv.most_similar("cat", topn=5))
# Prints the top 5 words most similar to cat.
```

But... where is it actually used? Well, Word2Vec and its variants (like Item2Vec) show up in all kinds of places.

For example, Item2Vec is used in e-commerce & recommendations. Let's see how.

Suppose users frequently browse phone accessories in a single session:
```
Session 1: "iPhone_15  MagSafe_Case  Screen_Protector  AirPods"
Session 2: "iPhone_15  Screen_Protector  USB_C_Cable"
```

If we choose `MagSafe_Case` as the target word, the Skip-Gram engine maximizes the dot product between the vector for `MagSafe_Case` and context items like `Screen_Protector`. When a user views a `MagSafe_Case` on the website, the recommendation engine computes cosine similarity against pre-computed item vectors and immediately shows `Screen_Protector` in the "Frequently Bought Together" widget.

That's roughly how it works. Now let's continue with DeepWalk.

## 2. DeepWalk

Before explaining DeepWalk, we need to understand the **random walk**. As the name suggests, it's the idea of starting at a node and following a probability distribution to choose the next step.

Here's an example:
```
(A) ─── (B) ─── (C) ─── (D)
           │
          (E)
```

The idea is simple:

```math
P(X_{t+1} = v \mid X_t = u) = \begin{cases} \frac{1}{d_u} & \text{if } (u, v) \in E \\ 0 & \text{otherwise} \end{cases}
```

where $`d_u`$ is the degree of node $`u`$. For example, our degrees here are:

```math
d_A = 1, \quad d_B = 3, \quad d_C = 2, \quad d_D = 1, \quad d_E = 1
```

Say we start at node A. Since A only connects to B:

```math
P(B \mid A) = 1.0
```

so we move: `A → B`.

B has 3 neighbors, so:

```math
P(A \mid B) = \frac{1}{3}, \hspace{1cm}P(C \mid B) = \frac{1}{3}, \hspace{1cm} P(E \mid B) = \frac{1}{3}
```

Say we land on C: `A → B → C`.

C has 2 neighbors, so:

```math
P(B \mid C) = \frac{1}{2}, \hspace{1cm}P(D \mid C) = \frac{1}{2}
```

Say we land on D: `A → B → C → D`.

So, DeepWalk is a graph-embedding algorithm that learns node embeddings by applying Word2Vec/Skip-Gram to random walks over a graph.

Before we confuse anything, here's the key idea:
```
               ┌─────────────────────────────────────────┐
               │                Word2Vec                 │
               │      (Parent Embedding Framework)       │
               └────────────────────┬────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                      CBOW                  Skip-Gram
                 (Context -> Target)     (Target -> Context)
                                                │
                                ┌───────────────┴───────────────┐
                                │                               │
                            DeepWalk                         Node2Vec
                   (Unbiased Uniform Walks)             (Biased p/q Walks)
```

So let's not larp like Word2Vec is just some incidental strategy *inside* DeepWalk — it's the engine DeepWalk is actually built on.

The whole DeepWalk pipeline is:
```
                GRAPH
                  │
                  ▼
           Random Walks
                  │
                  ▼
       sequences of node IDs
                  │
                  ▼
             Skip-Gram
                  │
                  ▼
          Node Embeddings
```

Say we have a small graph:
```
       A ─── B
       │     │
       │     │
       C ─── D ─── E
                   │
                   F
```

DeepWalk walks randomly through it, giving us walks like:
```
A → B → D → C → A → B
```
and:
```
E → D → B → A → C → D
```
and:
```
F → E → D → C → A → B
```

DeepWalk treats these like sentences:
```
"A B D C A B"
"E D B A C D"
"F E D C A B"
```

Then Skip-Gram/Word2Vec learns from those sentences.

When do we use DeepWalk? Whenever we want the answer to this question: "given a graph, how can I create sequences that capture its structure, so Word2Vec can learn node embeddings from them?"

DeepWalk has two main stages:

1. First stage:

Generate the random walks:
```
Graph
 │
 ├── walk 1: A B C D A
 ├── walk 2: C D E F D
 ├── walk 3: B A D E F
 └── ...
```
This is our "artificial corpus."

2. Second stage:

Feed the walks into Skip-Gram, which creates the embeddings.
```
A B C D A
C D E F D
B A D E F
       │
       ▼
   Skip-Gram
       │
       ▼
vectors for A, B, C, D, E, F
```

So if somebody asks, "what makes DeepWalk different from Node2Vec?", we can simply say: "Word2Vec learns from an existing sequence corpus, while DeepWalk generates its own corpus from graph random walks, then uses the Skip-Gram mechanism to learn node representations."

Now let's use it on a famous graph — `networkx`'s classic. So we're back with the "Shrimp and Airi" story.

Let's first download networkx (even though I suspect we got it already):
```shell
pip install networkx
```

A quick check:
```python
import networkx as nx
from gensim.models import Word2Vec

graph = nx.karate_club_graph()

print(graph.number_of_nodes(), "nodes")
print(graph.number_of_edges(), "edges")

"""
Output:

34 nodes
78 edges
"""
```

To use DeepWalk, let's install it:
```shell
pip install karateclub
```

Or, if that doesn't work, in your venv:
```shell
pip install karateclub --no-deps

# and then:
pip install python-louvain pygsp six

# You may see a lot of red text — ignore it.
```

Now let's write:
```python
from karateclub import DeepWalk

model = DeepWalk(walk_number=10, walk_length=20, dimensions=64, window_size=5, epochs=10)

model.fit(graph)

embeddings = model.get_embedding()

print(embeddings.shape, "<--- this is the shape of our embedding")

"""
Output:

(34, 64) <--- this is the shape of our embedding
"""
```

This runs `Word2Vec(...)` internally for you, so you're left with just the embeddings. Now let's continue with Node2Vec.

But first, let me give you an example of where DeepWalk shines: **fraud detection**.

Here's the key idea to understand first: a fraudster wouldn't steal someone's card and send money directly to their own account, since that's easy to spot. Instead, they evade detection by inserting buffer accounts (mules or shell entities) between the stolen funds and the destination account. This is why detection fails.

So we get a small graph like:
```
Victim's account, under the fraudster's control
  ↓
  A ──────► B ──────► C  ← the original fraud account
            ↑
       mule / shell
```

As you can see, the fraudster sends money to the mule, then to their original account — so now that original account appears to have no direct relationship with the victim's account.

Let's run DeepWalk on this graph:
```
 [ Stolen Card Source ] (Node S)
               │
      ┌────────┴────────┐
      ▼                 ▼
 [ Buffer A1 ]     [ Buffer A2 ]
      │                 │
      ▼                 ▼
 [ Mule X1 ]       [ Mule X2 ]
      └────────┬────────┘
               ▼
       [ Mule X3 ] (Central Cash-Out Hub)
               │
               ▼
    [ Final Account C ] (Clean Bank Account)
```

Where $`A_1, A_2, X_1, X_2, X_3`$ are newly created, synthetic accounts with no prior bad history.

Now let's launch DeepWalk with a maximum walk length of `5`, so it takes up to 5 hops from the starting node, and each walk can start from a different node. We might get:

Walk 1 (starts at $`S`$): `S  ->  A1  ->  X1  ->  X3  ->  C`
Walk 2 (starts at $`S`$): `S  ->  A2  ->  X2  ->  X3  ->  C`
Walk 3 (starts at $`A_1`$): `A1 ->  X1  ->  X3  ->  C  ->  X3`
Walk 4 (starts at $`X_1`$): `X1 ->  A1  ->  S   ->  A2  ->  X2`

Now we feed these node sequences into Word2Vec Skip-Gram with a context window, targeting node `X3`. The loss over a set of context co-occurrences looks like:

```math
\mathcal{L} = \sum_{w_c \in \{A_1, X_1, C, S\}} \log P(w_c \mid X_3) = \sum_{w_c} \log \left( \frac{\exp(v_{X_3}^\top u_{w_c})}{\sum_{y \in \mathcal{V}} \exp(v_{X_3}^\top u_y)} \right)
```

and we end up with something like:

| Node Pair | Graph Distance | Cosine Similarity $`\cos(v_i, v_j)`$ | Fraud Interpretation |
| :--- | :--- | :--- | :--- |
| $`(S, A_1)`$ | 1 Hop | 0.92 | Very High Risk |
| $`(S, X_3)`$ | 3 Hops | 0.87 | High Risk (caught by DeepWalk) |
| $`(S, \text{Legit User})`$ | $`\infty`$ Hops | $`-0.12`$ | Normal / Safe |

Now that we understand how it works, let's continue with Node2Vec.

## 3. Node2Vec

First, let's explain the difference between Node2Vec and DeepWalk.

For DeepWalk, the random walk is simply:
```
current node (D)
     │
     ├── neighbor A  ← equal chance
     ├── neighbor B  ← equal chance
     └── neighbor C  ← equal chance
```
If we move toward one of these nodes, we have no memory of the previous one — we won't remember that we came from node `D`.

Node2Vec, on the other hand, remembers:
```
A → B → ?
    ↑
    │
"We came from A"
```
It asks: "given that I came from A and I'm currently at B, how much should I prefer each possible next node?"

This is where I want to introduce the `p` and `q` parameters — but first, an important idea Node2Vec relies on.

Suppose we're walking:
```
t    ───>      v         ───> x
previous    current      candidate
```
where:
```
t = the previous node
v = the current node
x = a possible next node
```
Node2Vec looks at the relationship between `t` and `x`, which comes in three types:

![The three t-x relationship types](Node%20to%20can.png)

1. $`t = x`$

How can the previous node equal the possible next node? It can! Imagine we go from `A` to `B`; `B`'s neighbors are `C`, `D`, `A`. If we move back to `A`, our sequence is `A -> B -> A`.

While standing on `B`:
```
t = A
v = B
x = A
```
so $`t=x`$, with distance $`d(t,x)=0`$.

2. $`x \in \mathcal{N}(t)`$

This happens when the previous node is connected to the possible next node — so `t - v - x`, with `t` connected to both `v` and `x`. This gives us $`d(t,x)=1`$: `t` and `x` are directly connected by a single edge.

3. $`x \notin \mathcal{N}(t)`$

This happens when the previous node is *not* connected to the possible next node, giving us a shortest path of:

```math
d(t, x) = 2
```

Where $`d(t,x)`$ = the number of edges in the shortest path between the previous node `t` and candidate next node `x`.

Node2Vec's transition-rule formula looks like:

```math
\pi_{vx} = \alpha_{pq}(t,x)w_{vx}
```

Let's break the symbols apart. Starting with what we know:

- $`t`$ — our previous node
- $`x`$ — our possible next node
- $`v`$ — our current node

And the new pieces:
- $`w_{vx}`$ — the weight of the edge from our current node ($`v`$) to the candidate node ($`x`$). (If the graph is unweighted, this is just 1.)

The whole $`\alpha_{pq}(t,x)`$ is equal to:

```math
\alpha_{pq}(t, x) = \begin{cases} \frac{1}{p} & d(t, x) = 0 \\ 1 & d(t, x) = 1 \\ \frac{1}{q} & d(t, x) = 2 \end{cases}
```

Let's unpack this.

What is $`p`$? The *return* parameter:

```math
p - \text{the return parameter}
```

It controls how willing we are to immediately go back to where we came from. Say we moved from `A` to `B`, and the graph looks like:
```
        A
        ↑
        │
C ←──── B ──── D
```
The probability of going back from `B` to `A` is characterized by:

```math
\frac{1}{p}
```

For example, a large $`p`$, like $`p=10`$, gives:

```math
\frac{1}{10} = 0.1
```

strongly discouraging us from going back. So:
```
p ↑
│
├── immediate return ↓
│
└── exploration ↑
```

What about $`q`$? It controls whether the walk prefers to:
```
stay around the current neighborhood
```
or
```
move outward and explore new territory
```

```math
\frac{1}{q}
```

controls candidates two hops away from the previous node.

A large $`q`$, like `4`, gives:

```math
\frac{1}{4} = 0.25
```

meaning the model prefers staying close to the previous node — more of a BFS-like (breadth-first search-like) exploration.

A small $`q`$, like `0.25`, gives:

```math
\frac{1}{0.25} = 4
```

so the model prefers moving further outward — more of a DFS-like (depth-first search-like) exploration.

So:
```
q large
    ↓
stay local
    ↓
BFS-like
```
and:
```
q small
    ↓
explore outward
    ↓
DFS-like
```

That doesn't mean Node2Vec is literally running DFS or BFS — it's just doing something *similar* while running a random walk.

|                             | **DeepWalk**                | **Node2Vec**             |
| --------------------------- | --------------------------- | ------------------------ |
| Random walks                | Unbiased                    | Biased                   |
| Remembers previous node?    | No meaningful bias from it  | **Yes**                  |
| $`p`$                         | No                          | Yes                      |
| $`q`$                         | No                          | Yes                      |
| Local exploration control   | Limited                     | Yes                      |
| Outward exploration control | Limited                     | Yes                      |
| BFS-like behavior           | Not explicitly controllable | $`q>1`$                    |
| DFS-like behavior           | Not explicitly controllable | $`q < 1`$                  |
| Complexity                  | Simpler                     | More sophisticated       |
| Main idea                   | Random walks + Skip-Gram    | Biased walks + Skip-Gram |

So the whole picture is:
```
DeepWalk
    │
    ├── random walks
    │
    └── Word2Vec
             ↓
        embeddings


Node2Vec
    │
    ├── biased random walks
    │       ↑
    │      p,q
    │
    └── Word2Vec
             ↓
        embeddings
```

So we're not building a whole new algorithm from scratch — we're reusing the same algorithm with a different sampling strategy.

Now let's implement it in Python. First, let's install:
```shell
pip install torch-geometric
```

And write:
```python
model = Node2Vec(
    edge_index=edge_index,
    embedding_dim=EMBEDDING_DIM,
    walk_length=WALK_LENGTH,
    context_size=CONTEXT_SIZE,
    walks_per_node=WALKS_PER_NODE,
    p=P,
    q=Q,
    num_negative_samples=NEGATIVE_SAMPLES,
)
```

Since we know some of these already, let's cover only what we don't. We'll try these ideas on this graph:
```
Cluster A                  Bridge                 Cluster B
 [0] ─── [1]                                      [5] ─── [6]
  │   ✕   │                                        │       │
 [2] ─── [3] ═════════════════════════════════════ [4]     [7]
                                                   │       │
                                                  [9] ─── [8]
```

- `walk_length` — the number of steps the model takes in a random walk, so if `walk_length = 5`, the model hops 5 times before stopping, ending up with 5 explored nodes total (including the start). We might end up with `Walk sequence: [0, 2, 3, 4, 5]`.

- `context_size` — our window size, nothing new to explain here.

- `walks_per_node` — the number of independent walks it takes per node. If `walks_per_node = 2`, we might get:
```
From Node 0 -> Walk 1: [0, 1, 2, 3, 4]
             -> Walk 2: [0, 2, 0, 1, 2]
From Node 1 -> Walk 1: [1, 2, 3, 4, 5]
...
```
Each node gets 2 walkthroughs.

That's the whole idea of Node2Vec. Now let's continue with our first GNN...

# Chapter 2. Graph Neural Networks

Table of contents:
1. Message Passing Framework (MPNN)
2. GCN (Kipf & Welling, 2017)
3. GraphSAGE
4. GAT: attention coefficients

## 1. Message Passing Framework (MPNN)

This is our first topic, because we want you to understand something before actually coding GCN, GraphSAGE, GAT, or others.

Why? Because GCN, GraphSAGE, and GAT are basically different ways of implementing the same idea: **message passing**.

Let's return to plain neural networks. We know that with a neural network we usually have:
```
x₁ ──┐
x₂ ──┼──> Neural Network ──> output
x₃ ──┤
x₄ ──┘
```
Each input has a relatively fixed structure.

But graphs are different:
```
        B
       / \
      /   \
     A     D
      \   /
       \ /
        C
```

Each node has some number of neighbors, but that number isn't fixed — it varies. For example:
```
A → [B, C]
B → [A, D]
C → [A, D]
D → [B, C]
```
And there's no natural ordering, like:
```
neighbor 1
neighbor 2
neighbor 3
```

So if we have:
```
A's neighbors = {B, C}
```
we understand that the set `{B, C}` is the same as `{C, B}`.

So we need a neural network that works with variable-sized, unordered neighborhoods. This is where message passing comes in.

Imagine each node carries information attached to it:
```
A: disease features
B: gene features
C: pathway features
D: drug features
```
Each node looks at its neighbors and receives information from them. The main idea:
```
┌─────────┐
│ MESSAGE │
└────┬────┘
     ↓
┌───────────┐
│ AGGREGATE │
└────┬──────┘
     ↓
┌────────┐
│ UPDATE │
└────┬───┘
     ↓
  new node
  features
```
This is MPNN.

We know what a node feature is: each node carries a feature vector of dimension $`d`$. For node $`i`$:

```math
h_i \in \mathbb{R}^d
```

where:
- $`h_i`$ — the feature vector of node $`i`$.
- $`d`$ — the number of features.
- $`\mathbb{R}^d`$ — a vector containing $`d`$ real-valued numbers.

For example, if:

```math
h_A = \begin{bmatrix} 0.2 \\ 0.7 \\ -0.1 \\ 0.4\end{bmatrix}
```

then $`d=4`$, and our graph might look like:
```
A: [0.2, 0.7, -0.1, 0.4]
B: [0.8, 0.1,  0.3, 0.2]
C: [0.4, 0.9, -0.2, 0.6]
D: [0.1, 0.2,  0.8, 0.5]
```

Now let's build an MPNN layer. Focus on one node, $`i`$, with neighbors:

```math
\mathcal{N}(i)
```

the set of nodes connected to $`i`$. So if we have:
```
       B
       ↓
       A ← C
       ↑
       D
```
then $`\mathcal{N}(A) = \{B, C, D\}`$.

MPNN does three things.

---
Step 1 — Message

Each neighbor sends a message to node $`i`$:

```math
m_{ij} = M(h_i,h_j)
```

where:
- $`j`$ — neighbor
- $`i`$ — receiving node
- $`h_i`$ — current features of the receiving node ($`i`$)
- $`h_j`$ — features of neighbor ($`j`$)
- $`M`$ — message function
- $`m_{ij}`$ — message sent from $`j`$ to $`i`$

For example:
```
B ── message ──> A
C ── message ──> A
D ── message ──> A
```
The message function can be as simple as:

```math
M(h_i, h_j) = h_j
```

"just send the neighbor's features." Or it could be a neural network:

```math
M(h_i, h_j) = MLP([h_i \| h_j])
```

($`\|`$ = concatenation). So if:
```
h_i = [0.2, 0.7]
h_j = [0.8, 0.1]

[h_i || h_j]
=
[0.2, 0.7, 0.8, 0.1]
```

That's the message step (I'll walk through real examples shortly).

---
Step 2 — Aggregate

Now node $`i`$ has several messages:
```
B ──> [message B]
C ──> [message C]
D ──> [message D]
```
We need to turn them into one representation — that's the aggregation's job:

```math
m_i = AGG(\{m_{ij} : j \in \mathcal{N}(i)\})
```

> [!IMPORTANT]
> The aggregation function must be **permutation-invariant** — feeding it $`\{A, B, C\}`$ or $`\{C, A, B\}`$ must give the exact same result, since there's no natural ordering among a node's neighbors. This single requirement is what rules out things like plain concatenation or an RNN over the neighbor list (unless you explicitly sort or otherwise canonicalize them first).

There are different types of aggregation:

1. Mean:

```math
m_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}
```

2. Sum:

```math
m_i =\sum_{j \in \mathcal{N}(i)} m_{ij}
```

3. Maximum:

```math
m_i = \max_{j \in \mathcal{N}(i)} m_{ij}
```

That's the whole idea of aggregation. This step is really important — it's the crucial piece that differentiates one GNN architecture from another.

---
Step 3 — Update

Now node $`i`$ has its aggregated neighbor information, $`m_i`$ — but we don't want to throw away its old information, $`h_i`$. So we combine them:
```
             old node
                hᵢ
                 │
                 ↓
neighbor info → UPDATE → new hᵢ
    mᵢ
```
Mathematically:

```math
h_i' = U(h_i, m_i)
```

where:
- $`h_i`$ — our old feature for node $`i`$
- $`m_i`$ — our new aggregated neighbor information
- $`U`$ — the update function
- $`h_i'`$ — our new node representation

A simple example:

```math
h_i'=\sigma(W[h_i \| m_i])
```

where:
- $`W`$ = learnable weight matrix
- $`\sigma`$ = activation function, such as ReLU
- $`\|`$ = concatenation

Putting it all together:

**Message**

```math
m_{ij} = M(h_i, h_j)
```

**Aggregate**

```math
m_i = AGG(\{m_{ij} : j \in \mathcal{N}(i)\})
```

**Update**

```math
h_i' = U(h_i , m_i)
```

So we do:
```
            NEIGHBORS
          ┌─────┬─────┐
          ↓     ↓     ↓
          B     C     D
          │     │     │
          └──┬──┴──┬──┘
                │
             MESSAGE
                │
                ↓
           AGGREGATION
                │
                ↓
        aggregated message
                │
                ↓
         ┌──────────────┐
 old hᵢ →│    UPDATE    │
         └──────┬───────┘
                ↓
                hᵢ'
```

Why does this help us? Imagine we start with:
```
A: disease
B: gene
C: pathway
```
and:
```
B ─── A ─── C
```
After one message-passing layer:
```
A knows about:
    A + B + C
```
so A's representation now contains information from its 1-hop neighborhood. One more layer gets us:
```
A
│
├── B
│   └── D
│
└── C
    └── E
```
So:

One layer: `A ← immediate neighbors`
Two layers: `A ← neighbors ← neighbors`
Three layers: `A ← 3-hop neighborhood`

Stacking message-passing layers expands each node's receptive field.

So GCN, GraphSAGE, and GAT aren't unrelated architectures — they're mostly different answers to the same question: "how do I send the message, aggregate it, and then update the node?"

| Architecture  | Main idea                                   |
| ------------- | ------------------------------------------- |
| **GCN**       | normalized neighbor aggregation             |
| **GraphSAGE** | learn from sampled/aggregated neighborhoods |
| **GAT**       | learn how important each neighbor is        |

```
                 MPNN
                  │
        ┌─────────┼─────────┐
        ↓         ↓         ↓
       GCN    GraphSAGE     GAT
        │         │         │
    normalized   sampled   attention
    aggregation  neighbors  weights
```

After one example, let's learn the most basic GNN: GCN.

---
Example:

Say we have 2 carbon atoms and 1 oxygen atom. Cute! Why do we care? Well...

This can take two forms:

Dimethyl Ether ($`C - O - C`$): a comparatively low-toxicity, stable gas used as a propellant.
Or
Ethylene Oxide (a 3-membered ring: $`C - C - O`$ forming a triangle): a highly reactive, carcinogenic, toxic gas.

Both have the same atoms — the difference is that one has a 3-membered, strained ring structure, like a coiled spring waiting to snap and damage human DNA.

1. Layer 0 (toxicity prediction: $`10\%`$)

The model knows only the atomic numbers: Carbon = 6, Oxygen = 8.
```
      [C1: 6]
      /     \
  [C2: 6] --- [O: 8]
```
What the atoms "know": nothing about shape or strain. Readout head: "I see two carbons and an oxygen — most common organic molecules with this formula are safe."

2. Layer 1 (toxicity prediction: $`30\%`$)

Each atom collects information about its neighbors. $`C_1`$ collects from $`C_2`$ and $`O`$: "I'm connected to one carbon and one oxygen." $`O`$ collects from $`C_1`$ and $`C_2`$: "I'm connected to two carbons." But that still doesn't reveal the answer — Dimethyl Ether also has two carbons and one oxygen.

3. Layer 2 (toxicity prediction: $`92\%`$)

Message passing happens a second time. Now neighbor $`C_2`$ sends $`C_1`$ the information it gathered in Layer 1: "my neighbor $`O`$ is the *same* neighbor you're connected to." Now the model has picked up on the ring — this is Ethylene Oxide.

We'd feed the model a dataset of 100,000 existing molecules that scientists spent decades testing in physical lab dishes — Compound A tested toxic, Compound B tested safe, Compound C tested toxic, and so on. The model learns the difference between toxic and non-toxic structures. Then, when a chemist designs 5,000,000 brand-new virtual molecules that have never existed on Earth — molecules no human can test, since nobody's synthesized them yet — the trained model can recall the geometric patterns associated with toxicity and flag, say, 4,990,000 as toxic or non-functional, handing the scientist a shortlist of the top 10,000 safe, promising candidates.

So instead of spending years trying things aimlessly, the AI hands us only the smart choices to try.

Now let's continue with our first GNN.

## 2. GCN (Graph Convolutional Network)

First question: what are we even trying to calculate?

Suppose our graph has $`N`$ nodes, and each node currently has $`d_l`$ features. We store all of this in a matrix:

```math
H^{(l)} \in \mathbb{R}^{N \times d_l}
```

where:
- $`N`$ — number of nodes
- $`d_l`$ — number of features at GNN layer $`l`$
- $`H^{(l)}`$ — node-feature matrix at layer $`l`$
- $`\mathbb{R}^{N \times d_l}`$ — a matrix with $`N`$ rows and $`d_l`$ columns

```math
H^{(l)} = \begin{bmatrix} 0.2 & 0.7 & 0.1 \\ 0.8 & 0.1 & 0.4 \\ 0.3 & 0.9 & 0.2 \\ 0.5 & 0.2 & 0.8 \end{bmatrix}
```

Each row corresponds to a node:
```
         feature 1  feature 2  feature 3

node A      0.2        0.7        0.1
node B      0.8        0.1        0.4
node C      0.3        0.9        0.2
node D      0.5        0.2        0.8
```

GCN transforms this entire matrix. Let's build up the idea.

We represent the graph with the adjacency matrix:

```math
A \in \mathbb{R}^{N \times N}
```

So for a graph like:
```
A ─── B
│
C
```
the adjacency matrix is:

```math
A = \begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 0 \\ 1 & 0 & 0\end{bmatrix}
```

Now we compute:

```math
\tilde{A} = A + I
```

where $`I`$ is the identity matrix. For our three-node graph:

```math
\tilde{A} =  \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 0 \\ 1 & 0 & 1\end{bmatrix}
```

Notice what happened — each node effectively gained a self-loop:
```
Before:

A ─── B
│
C

After adding I:

A ─── B
│
C

A ─── A
B ─── B
C ─── C
```

Why does GCN want self-loops? Remember the MPNN lesson — a node does:
```
neighbors
    ↓
aggregate
    ↓
update itself
```
If we use the adjacency matrix alone, $`AH`$, the aggregation only contains neighbor information — not the node's own current representation, which we also need. Adding $`I`$ makes the node its own neighbor:
```
      B
      ↓
C →   A   ← D
      ↑
      A
   (itself)
```
so when $`A`$ aggregates, it receives `B + C + D + itself`. That's the whole idea behind $`\tilde{A} = A + I`$.

But what about $`\tilde{A}H`$? This is where it starts getting beautiful. Say:

```math
\tilde{A} =  \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 0 \\ 1 & 0 & 1\end{bmatrix} \quad \text{and} \quad H = \begin{bmatrix} h_A \\ h_B \\ h_C\end{bmatrix}
```

We get:

```math
\tilde{A}H = \begin{bmatrix} h_A +h_B + h_C \\ h_A + h_B \\ h_A + h_C\end{bmatrix}
```

Look at node $`A`$: $`h_A' \sim h_A + h_B + h_C`$. It's aggregated A itself, plus B, plus C — that's already message passing! So $`\tilde{A}H`$ means "give every node the sum of its own features and its neighbors' features."

But there's a problem, as expected. Imagine:
```
A is connected to 2 nodes

B is connected to 1000 nodes
```
`A` receives information from 3 nodes (2 neighbors + itself), while `B` receives from 1001 (1000 neighbors + itself). The *magnitude* of their representations can become wildly different purely because of degree:
```
A:
h₁ + h₂ + h₃

B:
h₁ + h₂ + h₃ + ... + h₁₀₀₁
```
This isn't a stable aggregation, so we normalize it:

```math
D^{-1/2}\tilde{A}D^{-1/2}
```

We already know $`D`$ is our degree matrix, telling us how many neighbors each node has. Since:
```
A connected to B and C
B connected to A
C connected to A
```
(and remembering we add self-loops first), we get:
```
A → 3 connections
B → 2 connections
C → 2 connections
```
which mathematically looks like:

```math
D =  \begin{bmatrix} 3 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2\end{bmatrix}
```

our degree matrix.

What does $`D^{-1/2}`$ equal? This:

```math
D^{-1/2} = \begin{bmatrix} \frac{1}{\sqrt{3}} & 0 & 0 \\ 0 & \frac{1}{\sqrt{2}} & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}}\end{bmatrix}
```

As we can see, each node gets a normalization factor related to its own degree — higher-degree nodes receive a smaller factor.

But why two normalizations, $`D^{-1/2}\tilde{A}D^{-1/2}`$? This is called **symmetric normalization**. Think about an individual connection between node $`i`$ and node $`j`$ — its normalized contribution is:

```math
\frac{1}{\sqrt{\tilde d_i \tilde d_j}}
```

where $`\tilde d_i`$ = degree of node $`i`$ (after self-loops), $`\tilde d_j`$ = degree of node $`j`$. So the contribution depends on *both* nodes' degrees, which prevents popular nodes from dominating the whole graph.

Our full GCN propagation is:

```math
H' = D^{-1/2} \tilde{A} D^{-1/2} H
```

which basically means: "take each node's feature, include the node's own feature, normalize the contribution appropriately, and use that to update the feature matrix."

We still need a couple more pieces. Suppose $`d_l = 64`$ and we want the next layer's dimension to be $`d_{l+1}=128`$. We need a learnable transform — introduce:

```math
W^{(l)}
```

with shape:

```math
W^{(l)} \in \mathbb{R}^{d_l \times d_{l + 1}}
```

So with $`d_l=64`$ and $`d_{l+1}=128`$, our weight matrix is $`W^{64\times128}`$, and applying it:
```
H(l)                    W(l)
N × 64                  64 × 128

             ↓

           N × 128
```
This gets folded into the propagation as $`H^{(l)}W^{(l)}`$.

Finally, we finish with an activation function:

```math
\sigma = \text{ReLU} = \max(0, x)
```

Our full formula:

```math
H^{(l + 1)} = \sigma \Big(D^{-1/2} \tilde{A} D^{-1/2} H^{(l)} W^{(l)}\Big)
```

So the whole idea of GCN is:
```
H(l)
 │
 │  node features
 ↓
add self-loops
 │
 │  Ã = A + I
 ↓
normalize graph connections
 │
 │  D^(-1/2) Ã D^(-1/2)
 ↓
aggregate neighbor information
 │
 ↓
apply learnable feature transformation
 │
 │  W(l)
 ↓
activation
 │
 │  σ
 ↓
H(l+1)
```

How does this map back to MPNN?

| MPNN concept         | GCN                                |
| -------------------- | ----------------------------------- |
| **Message**          | Neighbor features                  |
| **Aggregate**        | Degree-normalized sum              |
| **Update**           | Linear transformation + activation |
| Self information     | Added through $`I`$                  |
| Learnable parameters | $`(W^{(l)})`$                        |

But this isn't the end — after theory, we go to practice. This is why we'll use two CSVs (`edges.csv` and `nodes.csv`).

We'll have a good friend we use everywhere: `Data`.
```python
data = Data(
    x=...,
    edge_index=...,
    y=...,
    train_mask=...,
    val_mask=...,
    test_mask=...,
)
```
We can think of it as:
```
PyG Data
│
├── x             - node features
├── edge_index    - graph connections
├── y             - labels
├── train_mask    - which nodes train
├── val_mask      - which nodes validate
└── test_mask     - which nodes test
```
That's what we'll work with.

Imagine a fraud graph, and we want to predict whether a node is fraudulent. Say we have 100 features, stored in:
```python
x
```
with shape:
```python
x.shape

"""
Output:

torch.Size([10000, 100])  < --- 10000 nodes, each with 100 features
"""

# To check an individual node's features:
x[0]
# Gives us the features of our first node.
```

`edge_index` usually has shape:

```math
[2, E]
```

— two rows and $`E`$ columns, with `src` as the first row and `dst` as the second, since that's what `edge_index` is expected to be fed as:
```
edge_index =
[
    [0, 0, 0, 0, 0, ...],
    [6, 8, 9, 11, 13, ...]
]
```
so we can remember:
```
edge_index[0, k] = source
edge_index[1, k] = destination
```

Instead of writing:
```python
A_norm @ X @ W
```
we'll use Python's top-tier library for this, `torch_geometric`:
```python
from torch_geometric.nn import GCNConv
```
where we'll write:
```python
self.conv1 = GCNConv(
    in_channels=100,
    out_channels=64,
)
```
meaning: feed it 100 inputs (our 100 features), get back 64. Then:
```python
self.conv2 = GCNConv(
	in_channels=64,
	out_channels=2,
)
```
which is:
```
64 hidden features
       ↓
     GCN
       ↓
2 output values
```
These two values tell us whether the node is fraudulent or safe.

Now let's continue with GraphSAGE — I'll explain the difference and what we gain from it.

## 3. GraphSAGE (Graph Sample and Aggregate)

Now I'll introduce GraphSAGE, but first I need to explain the problem it solves. If GCN says:
```
Take a node's feature, aggregate it, and update the node
```
GraphSAGE says:
```
Can I learn an aggregation function that I can apply to nodes I've never seen before?
```

Say we have a fraud graph like:
```
             Account B
                │
                │
Account A ──────┼────── Account C
                │
                │
             Account D
```
Every node has features like:
```
Account A
├── transaction amount
├── transaction frequency
├── account age
├── merchant diversity
└── ...
```

A GCN's parameters, at this moment, look like:
```
neighbors
    ↓
normalized aggregation
    ↓
linear transformation W
    ↓
new representation
```
while GraphSAGE's look more like:
```
neighbor features
       ↓
   AGGREGATOR
       ↓
neighbor representation
       ↓
combine with own representation
       ↓
new representation
```
The important part: the model learns *how* to aggregate features, rather than learning an aggregation specific to each node.

Let's walk through an example, assuming all nodes have a 2D feature vector:

```math
\mathbf{x} = [\text{Transaction Amount (\$k)}, \text{Account Age (years)}]
```

Target Account A: $`[10.0, 0.5]`$
Neighbor B: $`[8.0, 1.0]`$
Neighbor C: $`[12.0, 0.2]`$
Neighbor D: $`[1.0, 5.0]`$

Say our weight matrix is:

```math
W = \begin{bmatrix} 0.5 & -0.2 \\ -0.8 & 1.2 \\ 0.3 & -0.1 \\ -0.4 & 0.9 \end{bmatrix}
```

Using a mean aggregator:

```math
\mathbf{h}_{N(A)} = \text{Mean}\left(\begin{bmatrix} 8.0 & 1.0 \end{bmatrix}, \begin{bmatrix} 12.0 & 0.2 \end{bmatrix}, \begin{bmatrix} 1.0 & 5.0 \end{bmatrix}\right) = \begin{bmatrix} 7.00 & 2.07 \end{bmatrix}
```

Now concatenate the target account with the result:

```math
\mathbf{z}_A = [\mathbf{x}_A \parallel \mathbf{h}_{N(A)}] = \begin{bmatrix} 10.00 & 0.50 & 7.00 & 2.07 \end{bmatrix}
```

And apply the weight matrix:

```math
\text{Raw Projection} = \mathbf{z}_A \cdot W = \begin{bmatrix} 5.87 & -0.24 \end{bmatrix}
```

```math
\mathbf{h}_A^{\text{new}} = \text{ReLU}(\begin{bmatrix} 5.87 & -0.24 \end{bmatrix}) = \begin{bmatrix} 5.87 & 0.00 \end{bmatrix}
```

Now say tomorrow, Account E joins the network, completing transactions with two existing nodes, $`X`$ and $`Y`$.
New Node E: $`[15.0, 0.1]`$
Neighbor X: $`[11.0, 0.3]`$
Neighbor Y: $`[9.0, 0.8]`$

This account never existed before, so it's not in our original adjacency matrix. Instead of retraining the full GNN or rebuilding a full graph Laplacian, GraphSAGE applies the *same learned* aggregation and the *same learned* weight matrix ($`W`$):

```math
\mathbf{h}_{N(E)} = \text{Mean}\left(\begin{bmatrix} 11.0 & 0.3 \end{bmatrix}, \begin{bmatrix} 9.0 & 0.8 \end{bmatrix}\right) = \begin{bmatrix} 10.00 & 0.55 \end{bmatrix}
```

Concatenating with E's features:

```math
\mathbf{z}_E = [\mathbf{x}_E \parallel \mathbf{h}_{N(E)}] = \begin{bmatrix} 15.00 & 0.10 & 10.00 & 0.55 \end{bmatrix}
```

Transforming via the same $`W`$:

```math
\text{Raw Projection} = \mathbf{z}_E \cdot W = \begin{bmatrix} 10.20 & -3.39 \end{bmatrix}
```

```math
\mathbf{h}_E^{\text{new}} = \text{ReLU}(\begin{bmatrix} 10.20 & -3.39 \end{bmatrix}) = \begin{bmatrix} 10.20 & 0.00 \end{bmatrix}
```

So GCN is like taking a fixed class photo — if a new student joins, you have to rearrange everyone and take another photo (retrain the GCN). GraphSAGE is more like a universal recipe: instead of memorizing the whole crowd, it learns a reusable rule — "combine a user's profile with the average profile of their direct contacts." When a new user joins, it just applies the rule, and that's it.

A simplified GraphSAGE layer looks like:

```math
h_{\mathcal{N}(v)}^{(l)} = \text{AGGREGATE}\Big\{h_u^{(l)} : u \in \mathcal{N}(v)\Big \}
```

Where:
- $`v`$ — our target node.
- $`\mathcal{N}(v)`$ — the set of neighbors directly connected to target node $`v`$.
- $`u`$ — an individual neighbor node, $`u \in \mathcal{N}(v)`$.
- $`l`$ — the current layer index (or hop distance).
- $`h_u^{(l)}`$ — the feature vector of neighbor node $`u`$ at layer $`l`$.
- $`h_{\mathcal N(v)}^{(l)}`$ — the aggregated neighborhood representation of our target node $`v`$ at layer $`l`$.

Then:

```math
h^{(l + 1)}_v = \sigma \Big \{ W^{(l)} \Big [h^{(l)}_v \| h_{\mathcal N(v)}^{(l)}\Big ] \Big \}
```

One more piece: $`h_v^{(l)}`$ — target node $`v`$'s own feature vector at layer $`l`$.

Now let's introduce two new words: "inductive" and "transductive" — what do they mean?

Think about training a graph:
```
TRAINING GRAPH

A ─── B ─── C
│
D
```
but tomorrow a new node appears:
```
NEW ACCOUNT
     E
     │
     B
```
We never saw E during training, but GraphSAGE still does:
```
E's features
     +
B's representation/features
     ↓
GraphSAGE aggregator
     ↓
E representation
     ↓
 prediction
```

So, *inductive* simply means: learning a universal rule from specific training examples, so you can apply it to brand-new, unseen cases. For example, if I say "all the swans I've seen in Europe are white," we can generalize to the rule: "swans are white."

What about *transductive*? This means reasoning only between specific, known cases, without extracting any general rule. For us, it sounds like: "nodes A, B, C, and D are in this specific room — I know the relationships between A, B, C, and D." This is the transductive nature of plain GCN: "I learned position embeddings specifically for Node A, Node B, Node C, and Node D." If Node E walks into the room tomorrow, the system breaks, because Node E was never part of the original closed setup.

This is why GraphSAGE is better than GCN for financial networks, which never sit still — they're always more like:
```
Monday
10,000 accounts

       ↓

Tuesday
10,173 accounts

       ↓

Wednesday
10,481 accounts
```

Before continuing, GraphSAGE translates as: Graph Sample and Aggregate. But why does it *sample* neighbors?

Our toy graph has 60k edges, which isn't much, but imagine a real one where:
```
one node
   │
   ├── 2,000 neighbors
   ├── 3,000 neighbors
   ├── 1,500 neighbors
   └── ...
```
If a GNN recursively expands every neighbor, we get:
```
Layer 1:
100 neighbors

Layer 2:
100 × 100 = 10,000

Layer 3:
100 × 100 × 100 = 1,000,000
```
Since layer 1 checks the neighbors of our target node $`v`$, layer 2 immediately pulls in the neighbors of those neighbors, and by the third hop we're pulling in the neighbors of the neighbors of the neighbors of our target node — a **neighbor explosion**.

This is why GraphSAGE samples them. Say our target node has 50 neighbors, its neighbors have 120 neighbors each, and *their* neighbors have 240 each.

What happens without sampling?

```math
\text{Without Sampling: } 1 \xrightarrow{50} 50 \xrightarrow{\times 120} 6,000 \xrightarrow{\times 240} 1,440,000 \text{ nodes } (\mathbf{1,446,051 \text{ total nodes}})
```

Our GPU explodes — which is why we're forced to sample.

GraphSAGE doesn't say "give me all the nodes" — it samples instead:
```python
NEIGHBOR_SAMPLES = [15, 10, 5]
```

Here's how GraphSAGE processes a network across 3 layers using $`S_1 = 15`$, $`S_2 = 10`$, $`S_3 = 5`$:
```
[ Target Node ] (1 node)
       │
       ▼  (Sample max 15 out of 50)
[ Hop-1 Neighbors ] (15 nodes)
       │
       ▼  (Sample max 10 out of 120 per Hop-1 node)
[ Hop-2 Neighbors ] (15 × 10 = 150 nodes)
       │
       ▼  (Sample max 5 out of 240 per Hop-2 node)
[ Hop-3 Neighbors ] (150 × 5 = 750 nodes)
```
So in the first layer, GraphSAGE chooses just 15 nodes out of the 50 (35 nodes get ignored), and so on.

> [!WARNING]
> Skip sampling and this blows up exponentially — the $`1 \to 50 \to 6{,}000 \to 1{,}440{,}000`$ example above is exactly why real-world GraphSAGE deployments always cap the fan-out per layer. Three unconstrained hops on a node with a few hundred neighbors is enough to try to pull in a sizeable chunk of the whole graph.

What if we write:
```python
NEIGHBOR_SAMPLES = [100, 10, 5]
```
By default, PyTorch Geometric's `NeighborLoader` just takes the 50 nodes we actually have and moves on, since there aren't 100 to sample from in the first layer.

GraphSAGE has another advantage: it doesn't force us into a single aggregation, unlike GCN, which by default only uses the degree-normalized sum. GraphSAGE can use several aggregators, such as:
- Mean Aggregator
- Sum Aggregator
- LSTM Aggregator, and so on

So, where do we reach for GraphSAGE instead of GCN?

1. Real-time systems

For real-time systems (e.g. e-commerce, fraud detection, recommendations, spam filtering) where transactions, spam notifications, or posts appear every second. Why not GCN? Because, as said, GCN learns 3 nodes and panics if a new one appears, while GraphSAGE just applies its general rule and keeps going, instead of falling apart.

2. Massive graphs that don't fit on your GPU

Imagine a graph with millions or billions of nodes/edges (e.g. Pinterest's pin-board graph, Twitter's follower graph) — GCN will greedily try to eat it all, choke, and hand you an out-of-memory error. GraphSAGE samples instead, keeping VRAM usage low and constant.

That's how good GraphSAGE is — now let's introduce another friend.

## 4. GAT (Graph Attention Networks)

Now let's introduce GAT, usually the last of the "basic three." So far we've seen:
```
GCN
 ↓
normalized neighborhood aggregation

GraphSAGE
 ↓
sample neighbors
 ↓
aggregate them
```
Now let's talk about GAT.

Imagine we have a graph:
```
                    Account B
                       │
                       │
Account C ─────── Account A ─────── Account D
                       │
                       │
                    Account E
```
Should all neighbors be equally important? Not necessarily — suppose:
```
B = normal account
C = normal account
D = suspicious merchant network
E = suspicious device network
```
Treating everybody equally isn't the best idea here. Instead, GAT says: "let the model learn how important each node is."

The intuition is simple. Instead of:
```
neighbor 1 ─┐
neighbor 2 ─┼──→ aggregate
neighbor 3 ─┘
```
we do:
```
neighbor 1 ── × α₁ ──┐
neighbor 2 ── × α₂ ──┼──→ aggregate
neighbor 3 ── × α₃ ──┘
```
where $`\alpha_{ij}`$ is how much attention node $`i`$ gives to node $`j`$.

So, for target node A:
```
neighbor B → α = 0.10
neighbor C → α = 0.15
neighbor D → α = 0.60
neighbor E → α = 0.15
```
This means $`D`$ is the most relevant to $`A`$'s representation.

So "attention" basically means: don't treat each neighbor equally — learn which information matters. GAT computes attention over a node's graph neighborhood: for node $`i`$, it computes attention over all of $`i`$'s neighbors:
```
              Graph neighborhood

                   j₁
                    \
                     \
              j₂ ─── i ─── j₃
                     /
                    /
                   j₄

                 ↓

       attention over {j₁,j₂,j₃,j₄}
```

The first GAT formula is:

```math
\alpha_{ij} = \text{softmax}_j\big(\text{LeakyReLU}\big (\mathbf{a}^T[W h_i \| W h_j]\big) \big)
```

Where:
- $`h_i`$ — our target node's features. For example:
```python
# Account A

h_A =
[
  transaction amount,
  transaction frequency,
  account age,
  ...
]
```
- $`Wh_i`$ — a learnable transformation, with $`W`$ learned during training.
- $`\mathbf{a}`$ — another learned parameter vector, which produces a scalar score when combined with a pair of transformed node features:
```
node i
   +
node j
   ↓
attention mechanism
   ↓
score
```
So the *raw* attention score is:

```math
e_{ij}=\mathbf{a}^T[Wh_i \| Wh_j]
```

and the attention coefficient $`\alpha_{ij}`$ is the *normalized* version of this score, after it passes through LeakyReLU and softmax:
```
e_ij
 ↓
LeakyReLU
 ↓
(normalized across j)
 ↓
attention coefficient α_ij
```
The important part: GAT computes a learned compatibility score between node $`i`$ and each of its neighbors $`j`$.

The final piece of the formula is the softmax. If `A`'s raw scores are:
```
B → 1.2
C → 0.3
D → 2.5
E → 0.8
```
softmax converts these into normalized coefficients:
```
B → 0.15
C → 0.06
D → 0.65
E → 0.14
```
giving us `A`'s neighborhood weighted aggregation:
```
A's neighborhood

B ── 0.15 ─┐
C ── 0.06 ─┤
D ── 0.65 ─┼──→ weighted aggregation
E ── 0.14 ─┘
```
Now we aggregate — our new representation is essentially:

```math
h_i' = \sigma \big (\sum_{j \in \mathcal N (i)} \alpha_{ij} W h_j \big )
```

which is basically:
```
neighbor representation
        ×
attention weight
        ↓
weighted neighbor information
        ↓
sum
        ↓
activation
        ↓
new representation
```

GAT typically uses multiple attention heads (e.g., 8). What does that mean? Each head focuses on different features — for example:

- **Head 1** might focus on transaction velocity (flagging rapid transfers).
- **Head 2** might focus on account-age disparities (flagging old accounts connected to brand-new ones).
- **Head 3** might look at shared device/IP footprints.
- **Heads 4–8** might pick up on other subtle topological or attribute patterns.

We'll use GAT for many things in medicine, chemistry, and even heterogeneous graphs.

Now let's build an entire project — you don't have to memorize every single part; once we get to `data.py`, just understand that it's simply how we extract the data and prepare it for feeding to the model.

# Project — Fraud Graph

This project is big and hard, but don't worry — I'll try to explain it in plain words.

Graph info:
- 10,000 nodes
- 60,479 edges (avg degree ~12)
- 11 numerical node features
- 350 fraud cases
- the graph is homogeneous

Now the task: given the graph, build an algorithm that predicts whether a node is fraudulent.

Our features ($`X`$):
```
log_amount
num_transactions_24h
num_transactions_7d
avg_amount_7d
max_amount_30d
unique_merchants_7d
unique_devices_30d
account_age_days
avg_balance
is_night_transaction_ratio
foreign_transaction_ratio
```
and our target ($`y`$): `is_fraud`.

So our data becomes:
```
nodes.csv
     │
     ├── node_id       → identity/index
     │
     ├── 11 features   → X
     │
     ├── is_fraud      → y
     │
     └── split         → train/val/test masks
```
and:
```
edges.csv
     │
     ├── src
     └── dst
          │
          ↓
      graph edges
```

We create a folder called `fraud-graph`:
```
fraud-graph/
│
├── data/
│   ├── nodes.csv
│   └── edges.csv
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── main.py
│
└── pyproject.toml
```

The first step is `src/`, since our data is simply the files holding the nodes and edges.

## `config.py`

Now let's build `config.py`. In case it's unfamiliar: `config.py` is simply a central Python module storing filepaths, model hyperparameters, and environment variables — instead of scattering them throughout the main code, we keep them here.

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent
    nodes_path: Path = project_root / "data/fraud_nodes.csv"
    edges_path: Path = project_root / "data/fraud_edges.csv"

    target_column: str = "is_fraud"

    hidden_channels: int = 64
    num_classes: int = 2

    learning_rate: float = 1e-3
    weight_decay: float = 5e-4

    epochs: int = 600
    patience: int = 20

    # Specific GraphSAGE hyperparameters
    dropout: float = 0.2
    sage_aggregator: str = "mean"  # We choose our aggregation.

    # Specific GAT hyperparameters 
    gat_heads: int = 4
    gat_concat: bool = True

    random_seed: int = 9

    threshold: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


config = Config()

```

Let's take it slowly.

- `class Config(BaseSettings)` — our settings object. We do this so, instead of scattering random numbers around the code, we just call `config.learning_rate`, `config.target_column`, and so on.

- `Path(__file__).resolve().parent` — broken down:
     `__file__` — our current Python file.
     `.resolve().parent` — the folder that contains this file.
  For example, if your script is at `/home/user/fraud_project/config.py`, then `project_root` becomes `/home/user/fraud_project`.

- `nodes_path: Path = project_root / "data/fraud_nodes.csv"` — takes `/home/user/fraud_project` and appends `"data/fraud_nodes.csv"`, giving `"/home/user/fraud_project/data/fraud_nodes.csv"`. Same idea for `edges_path`.

- `target_column` — the name of our label column ($`y`$), so `"is_fraud"` holds `0` and `1`.

- `patience` — if the validation score doesn't improve for `20` epochs in a row, we stop training immediately. So the training loop can run up to `600` times, but may stop earlier if the model stops improving.

- `threshold = 0.5` — if the model predicts $`50\%+`$ fraud probability, we call it fraud; below that, we call it legit. We'll later write: `preds = (probabilities >= config.threshold).astype(int)`.

```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)
```
This looks for a file called `.env`, letting us override any setting without touching the code:
```
learning_rate=0.0005
epochs=100
```

And we finish with `config = Config()`, so we can do `config.hidden_channels` and so on, everywhere.

That was our `config.py`. Now let's continue with a genuinely hard part: `data.py`.

## `data.py`

`data.py` holds all the data ingestion, feature preprocessing, graph construction, and `DataLoader` creation. We use it to take raw files and turn them into something suitable for learning. (This isn't the *only* way to structure it — it's an architecture choice; I just tried to make it safe and reusable.)

```python
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "num_transactions_24h",
    "num_transactions_7d",
    "avg_amount_7d",
    "max_amount_30d",
    "unique_merchants_7d",
    "unique_devices_30d",
    "account_age_days",
    "avg_balance",
    "is_night_transaction_ratio",
    "foreign_transaction_ratio",
]


def load_file(nodes_path: Path, edges_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    return nodes, edges



def validate_node(
    nodes: pd.DataFrame, feature_col: list[str], target_col: str
) -> None:

    required_col = {
        "node_id",
        "split",
        target_col,
        *feature_col,  # The * means that it will add all the elements from feature_col to the required_col
    }

    missing_col = required_col - set(nodes.columns)

    if missing_col:
        raise ValueError(f"missing node columns: {sorted(missing_col)}")

    if nodes["node_id"].duplicated().any():
        raise ValueError("each node must be unique")

    if nodes["node_id"].isna().any():
        raise ValueError("nodes contain missing values")

    invalid_splits = set(nodes["split"].unique()) - {
        "train",
        "val",
        "test"
    }

    if invalid_splits:
        raise ValueError(f"Unexpected value split: {sorted(invalid_splits)}")

    if nodes[feature_col].isna().any().any():
        raise ValueError("feature column contains missing values")

    if nodes[target_col].isna().any():
        raise ValueError("target column contains missing values")


def validate_edges(
    edges: pd.DataFrame,
    *,
    valid_node_ids: set[int],
) -> None:
    valid_edges = {"src", "dst"}

    missing_edges = valid_edges - set(edges.columns)

    if missing_edges:
        raise ValueError(f"Missing edge column: {sorted(missing_edges)}")

    if edges[["src", "dst"]].isna().any().any():
        raise ValueError("Edges contain missing node IDs")

    unknown_nodes = (set(edges["src"]) | set(edges["dst"])) - valid_node_ids

    if unknown_nodes:
        raise ValueError("Edges reference unknown nodes")

    if (edges["src"] == edges["dst"]).any():
        raise ValueError("Self loops are not expected in the raw edges file.")


def build_idx(nodes: pd.DataFrame) -> dict[int, int]:
    return {int(node_id): idx for idx, node_id in enumerate(nodes["node_id"])}


def build_edge_index(
    edges: pd.DataFrame,
    *,
    nodes_to_idx: dict[int, int],
    undirected: bool = True,
) -> torch.Tensor:

    src = [nodes_to_idx[int(s)] for s in edges["src"]]
    dst = [nodes_to_idx[int(d)] for d in edges["dst"]]

    if undirected:
        src = src + dst
        dst = dst + src[: len(dst)]

    return torch.tensor([src, dst], dtype=torch.long)


def fit_feature_scaler(
    nodes: pd.DataFrame,
    *,
    feature_col: list[str],
) -> StandardScaler:
    train_mask = nodes["split"] == "train"
    scaler = StandardScaler()
    scaler.fit(nodes.loc[train_mask, feature_col])

    return scaler


def build_features(
    nodes: pd.DataFrame,
    *,
    feature_col: list[str],
    scaler: StandardScaler,
) -> torch.Tensor:
    scaled = scaler.transform(nodes[feature_col])

    return torch.tensor(scaled, dtype=torch.float32)


def build_labels(
    nodes: pd.DataFrame,
    *,
    target_col: str,
) -> torch.Tensor:
    return torch.tensor(nodes[target_col].to_numpy(), dtype=torch.long)


def build_mask(nodes: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = torch.tensor(nodes["split"] == "train", dtype=torch.bool)
    val_mask = torch.tensor(nodes["split"] == "val", dtype=torch.bool)
    test_mask = torch.tensor(nodes["split"] == "test", dtype=torch.bool)
    return train_mask, val_mask, test_mask


def build_graph(
    node_path: Path,
    edge_path: Path,
    *,
    feature_col: list[str],
    target_col: str,
    undirected: bool,
) -> tuple[Data, StandardScaler]:

    nodes, edges = load_file(node_path, edge_path)

    validate_node(nodes, feature_col, target_col)
    validate_edges(edges=edges, valid_node_ids=set(nodes["node_id"]))

    node_idx = build_idx(nodes)
    edge_idx = build_edge_index(
        edges=edges, nodes_to_idx=node_idx, undirected=undirected
    )

    scaler = fit_feature_scaler(nodes=nodes, feature_col=feature_col)
    X = build_features(nodes=nodes, feature_col=feature_col, scaler=scaler)
    y = build_labels(nodes=nodes, target_col=target_col)
    train_mask, val_mask, test_mask = build_mask(nodes=nodes)

    data = Data(
        x=X,
        edge_index=edge_idx,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    return data, scaler

```

- `def validate_node(...) -> None:` — this function returns nothing; its only job is to raise errors if we get something unexpected.

- `if missing_col:` — raises an error if the CSV is missing a feature. For example, if we're missing `avg_balance`:
```python
required_col = {"node_id", "split", "is_fraud", "log_amount", ..., "avg_balance", ...}
set(nodes.columns) = {"node_id", "split", "is_fraud", "log_amount", ...}   # no avg_balance

missing_col = {"avg_balance"}

→ raises: ValueError: missing node columns: ['avg_balance']
```

- `if nodes["node_id"].duplicated().any():` — raises an error if we have a duplicated node. Let's break down `.duplicated()` and `.any()`:
     - `.duplicated()` — flags each element that's a repeat of an earlier one, returning a boolean Series (marked `True` wherever a duplicate shows up).
     - `.any()` — returns `True` if at least one element in the series is `True`. For example:
```python
nodes["node_id"].duplicated()
→ [False, False, True, False]

.duplicated().any()
→ True   → raises the error
```

- `if nodes["node_id"].isna().any():` — checks whether any element is `NaN`, `None`, or missing, and raises an error if so.
     - `.isna()` — checks for `NaN`/`None`/missing values and returns `True` for each one found; `.any()` then raises the error if at least one node has a missing value:
```python
nodes["node_id"].isna()
→ [False, True, False]

.isna().any()
→ True → raises error
```

For the split column:
```python
invalid_splits = set(nodes["split"].unique()) - {"train", "val", "test"}

if invalid_splits:
    raise ValueError(f"Unexpected value split: {sorted(invalid_splits)}")
```
This checks for unwanted values in `split`. First, it collects all the unique values that appear in that column. Then it checks whether the only values present are `"train"`, `"val"`, and `"test"` — otherwise, it raises an error.
```python
nodes["split"].unique() → ['train', 'val', 'Train', 'test']

set(...) - {"train", "val", "test"} → {"Train"}

→ raises: ValueError: Unexpected value split: ['Train']
```

Now the last part of node validation:
- `if nodes[feature_col].isna().any().any():` — checks whether the feature columns have any missing values. We use a double `.any()`: the first checks per *column*, and the second raises the error if *any* column had a problem. For example, given this table:

| log_amount | num_transactions_24h | avg_balance |
| ---------- | -------------------- | ----------- |
| 2.7        | 3                     | 150.0       |
| 4.1        | NaN                   | 200.0       |
| 3.0        | 1                     | 180.0       |

```python
nodes[feature_col].isna()
→ DataFrame of True/False

.isna().any()          # per column
→ log_amount: False
  num_transactions_24h: True
  avg_balance: False

.isna().any().any()    # overall
→ True → raises error
```

Same idea for `if nodes[target_col].isna().any()`.

Now `def validate_edges()` does the same job as `validate_node`, just for the edges.

Let's skip the easy parts we already understand, and go through the ones that might trip us up:

- `src = src + dst` then `dst = dst + src[: len(dst)]` — together, these turn the graph undirected, by appending a reversed copy of every edge.

Say:
```
src | dst
---------
 1  |  2
 3  |  1
```
After those two lines, we end up with:
```
src becomes: [1, 3, 2, 1]
dst becomes: [2, 1, 1, 3]
```
giving us:
```
1 → 2
3 → 1
2 → 1   ← reverse of the first
1 → 3   ← reverse of the second
```

Now this part:
```python
train_mask = nodes["split"] == "train"
scaler = StandardScaler()
scaler.fit(nodes.loc[train_mask, feature_col])
```

Broken down:

`train_mask = nodes["split"] == "train"` — takes every node satisfying this condition (only the training ones).

If we have something like:

| Node | Split |
| ---: | :---- |
|    1 | train |
|    2 | val   |
|    3 | train |
|    4 | test  |

after `train_mask`, we get:
```
0    True
1    False
2    True
3    False
```

`nodes.loc[train_mask, feature_col]` — looks at the feature columns, and keeps only the rows belonging to the training set.

But what is this `StandardScaler`? It normalizes our features, since something like `account_age_days` can range from 0 to 5,000+ and would otherwise dominate the whole gradient just by sheer magnitude. Feeding the data through `StandardScaler` fixes that.

> [!NOTE]
> Notice that we `fit` the scaler on `train_mask` only, not on the whole dataset — if we fit it on everything (including validation/test rows), statistics from data the model is supposed to never have seen would leak into training. This is a classic way to accidentally inflate your reported metrics.

Now let's continue.

## `model.py`

Here we build the architecture for our two models:
```python
GraphSAGE
GAT
```
and make both work through shared code.

```python
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        dropout: float = 0.2,
        sage_aggregator: str = "mean",
    ) -> None:
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggr=sage_aggregator,
        )

        self.conv2 = SAGEConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            aggr=sage_aggregator,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
    ) -> Tensor:

        x = self.conv1(
            x,
            edge_index,
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        gat_heads: int = 4,
        gat_concat: bool = True,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=gat_heads,
            concat=gat_concat,
            dropout=dropout,
        )

        first_out_channels = (
            hidden_channels * gat_heads if gat_concat else hidden_channels
        )

        self.conv2 = GATConv(
            in_channels=first_out_channels,
            out_channels=out_channels,
            heads=1, # We use one head here, since using more would multiply our output size by the number of heads, instead of giving us exactly 2 outputs.
            concat=False,
            dropout=dropout,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.conv1(x, edge_index)

        x = F.elu(x)

        x = self.conv2(x, edge_index)

        return x
```

This step was easier than `config.py` — now let's continue with training.

## `train.py`

```python
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    test_logits: Tensor


def train_one_epoch(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
) -> float:

    model.train()

    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        logits[data.train_mask],
        data.y[data.train_mask],
    )

    loss.backward()

    optimizer.step()

    return float(loss.item())


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data: Data,
    mask: Tensor,
) -> float:

    model.eval()

    logits = model(data.x, data.edge_index)

    loss = F.cross_entropy(
        logits[mask],
        data.y[mask],
    )

    return float(loss.item())


def train_model(
    model: torch.nn.Module,
    data: Data,
    *,
    learning_rate: float = 1e-3,
    weights_decay: float = 5e-4,
    patience: int = 20,
    epochs: int = 200,
) -> TrainingResult:

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weights_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None

    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
        )

        val_loss = evaluate_loss(
            model=model,
            data=data,
            mask=data.val_mask,
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

        if epochs_without_improvement >= patience:

            print("Model stopped improving — early stop.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    with torch.no_grad():
        test_logits = model(
            data.x,
            data.edge_index,
        )

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        test_logits=test_logits,
    )
```

> [!CAUTION]
> Watch the indentation on that final `if best_state is None / model.load_state_dict(best_state) / with torch.no_grad(): test_logits = ...` block — it has to sit **outside** the `for epoch` loop (same indent level as `for epoch in range(...)` itself), running exactly once, after training finishes. It's an extremely easy bug to introduce by accident: if those lines end up indented one level deeper (inside the loop), you'd reload the best checkpoint and recompute `test_logits` on *every single epoch*, repeatedly discarding whatever the optimizer just did on epochs that didn't improve. The symptoms are subtle too — training *looks* like it's running, loss still prints, it just never actually explores past the current best checkpoint the way it's supposed to.

Let's start with:
- `model.train()` — we always call this before training, since it switches on training-mode behavior (things like `Dropout` and `BatchNorm` updates).

- `optimizer.zero_grad()` — clears old gradients from the previous step, so gradients don't accumulate indefinitely.

- `logits = model(data.x, data.edge_index)` — returns logits (not probabilities — raw scores). Something like:
```python
tensor([
    [ 1.82, -0.95],   # node 0
    [-0.30,  1.45],   # node 1
    [ 0.10,  0.05],   # node 2
    ...
    [ 2.10, -1.80],   # node 9999
])
```

- `logits[data.train_mask]` — keeps only the `train` nodes (the `loss` is simply our training loss).

Now — why do we do essentially the same thing again, under `evaluate_loss`? This actually tells us how good the model currently is. We need it for:
- **Early stopping** — `evaluate_loss` tells us when the model stops improving, so we know when to stop training.
- **Model selection** — we keep whichever version of the model performed best on validation.
- **Monitoring progress** — to see whether training is actually working.

So they serve different purposes. We also wrap `evaluate_loss` in `@torch.no_grad()`, which disables gradient tracking — we're not backpropagating here, just checking the loss, so there's no need to build the computation graph, and it saves memory.

Then we simply train the model while keeping track of the best parameters.

Now for our almost-last step: `evaluate.py`.

## `evaluate.py`

```python
from dataclasses import dataclass

import torch

from sklearn.metrics import (
    precision_score,
    average_precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def evaluate_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:

    probabilities = torch.softmax(logits[mask], dim=-1)[:, 1]

    prediction = (probabilities >= threshold).long()

    y_true = labels[mask].cpu().numpy()
    y_pred = prediction.cpu().numpy()
    y_prob = probabilities.cpu().numpy()

    return ClassificationMetrics(
        precision=precision_score(
            y_true=y_true,
            y_pred=y_pred,
            zero_division=0,
        ),
        recall=recall_score(
            y_true=y_true,
            y_pred=y_pred,
            zero_division=0,
        ),
        f1=f1_score(
            y_true=y_true,
            y_pred=y_pred,
            zero_division=0,
        ),
        roc_auc=roc_auc_score(
            y_true,
            y_prob,
        ),
        pr_auc=average_precision_score(y_true, y_prob),
    )


def print_metrics(
    metrics: ClassificationMetrics,
) -> None:

    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"ROC-AUC: {metrics.roc_auc:.4f}")
    print(f"PR-AUC: {metrics.pr_auc:.4f}")
```

This tells us how good our model actually is, more precisely than ever. But what do these metrics mean? Let's go one by one:

- `precision` — of all the times the model said "positive" (class 1), how many were actually correct?

Rough guidance:
```
0.85+     → excellent
0.70-0.80 → solid
below that is usually too noisy
```

- `recall` — of all the actual positive cases that exist, how many did the model catch?

Rough guidance:
```
0.80+     → excellent
0.60-0.80 → acceptable
< 0.50    → usually unacceptable
```

- `f1` — the harmonic mean of precision and recall. If either one is too low, it drags the score down.

Rough guidance:
```
0.80+     → strong
0.65-0.80 → solid
< 0.55    → needs work
```

- `roc_auc` — how well the model ranks positive examples above negative ones, across all possible thresholds?

Rough guidance:
```
0.90 - 1.00 → excellent
0.80 - 0.90 → good
0.70 - 0.80 → fair
0.50 - 0.70 → weak (barely better than random)
0.50        → pure random guessing
```

`pr_auc` (average precision) follows the same idea as ROC-AUC, it's just more honest when the data is imbalanced (as fraud data almost always is).

Now let's finish our project with the final `main.py`.

## `main.py`

Here we glue everything together and actually see the results.

```python
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import Config
from data import FEATURE_COLUMNS, build_graph
from evaluate import evaluate_predictions, print_metrics
from model import GAT, GraphSAGE
from train import train_model


def set_seed(
    seed: int = 9,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    *,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 2,
    dropout: float = 0.2,
    sage_aggregator: str = "mean",
    gat_heads: int = 4,
) -> torch.nn.Module:

    if model_name == "graphsage":

        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            sage_aggregator=sage_aggregator,
        )

    if model_name == "gat":

        return GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            gat_heads=gat_heads,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_experiment(
    model_name: str,
    config: Config,
) -> None:

    print(f"\n{'=' * 60}")

    print(f"Training {model_name.upper()}")

    print(f"{'=' * 60}")

    set_seed(config.random_seed)

    data, _ = build_graph(
        config.nodes_path,
        config.edges_path,
        feature_col=FEATURE_COLUMNS,
        target_col=config.target_column,
        undirected=True,
    )

    model = build_model(
        model_name,
        in_channels=data.num_node_features,
        hidden_channels=config.hidden_channels,
        out_channels=config.num_classes,
        dropout=config.dropout,
        sage_aggregator=config.sage_aggregator,
        gat_heads=config.gat_heads,
    )

    print(model)

    result = train_model(
        model=model,
        data=data,
        learning_rate=config.learning_rate,
        weights_decay=config.weight_decay,
        epochs=config.epochs,
        patience=config.patience,
    )

    metrics = evaluate_predictions(
        logits=result.test_logits,
        labels=data.y,
        mask=data.test_mask,
        threshold=config.threshold,
    )

    print(f"\nBest epoch: {result.best_epoch}")

    print(f"Best validation loss: " f"{result.best_val_loss:.4f}")

    print("\nTest metrics:")

    print_metrics(metrics)


def main() -> None:

    config = Config()

    run_experiment(
        model_name="graphsage",
        config=config,
    )

    run_experiment(
        model_name="gat",
        config=config,
    )


if __name__ == "__main__":
    main()

"""
Output:

============================================================
Training GRAPHSAGE
============================================================
GraphSAGE(
  (conv1): SAGEConv(11, 64, aggr=mean)
  (conv2): SAGEConv(64, 2, aggr=mean)
  (dropout): Dropout(p=0.2, inplace=False)
)
Epoch 001 | train_loss=0.4996 | val_loss=0.4878
Epoch 025 | train_loss=0.2165 | val_loss=0.2157
Epoch 050 | train_loss=0.1319 | val_loss=0.1247
Epoch 075 | train_loss=0.1052 | val_loss=0.0993
Epoch 100 | train_loss=0.0956 | val_loss=0.0899
Epoch 125 | train_loss=0.0904 | val_loss=0.0851
Epoch 150 | train_loss=0.0868 | val_loss=0.0821
Epoch 175 | train_loss=0.0850 | val_loss=0.0799
Epoch 200 | train_loss=0.0819 | val_loss=0.0783
Epoch 225 | train_loss=0.0815 | val_loss=0.0769
Epoch 250 | train_loss=0.0801 | val_loss=0.0759
Epoch 275 | train_loss=0.0787 | val_loss=0.0750
Epoch 300 | train_loss=0.0784 | val_loss=0.0742
Epoch 325 | train_loss=0.0762 | val_loss=0.0735
Epoch 350 | train_loss=0.0762 | val_loss=0.0728
Epoch 375 | train_loss=0.0746 | val_loss=0.0722
Epoch 400 | train_loss=0.0752 | val_loss=0.0717
Epoch 425 | train_loss=0.0726 | val_loss=0.0712
Epoch 450 | train_loss=0.0745 | val_loss=0.0708
Epoch 475 | train_loss=0.0728 | val_loss=0.0703
Epoch 500 | train_loss=0.0727 | val_loss=0.0699
Epoch 525 | train_loss=0.0713 | val_loss=0.0695
Epoch 550 | train_loss=0.0707 | val_loss=0.0691
Epoch 575 | train_loss=0.0697 | val_loss=0.0687
Epoch 600 | train_loss=0.0698 | val_loss=0.0684

Best epoch: 600
Best validation loss: 0.0684

Test metrics:
Precision: 0.8529
Recall: 0.4143
F1: 0.5577
ROC-AUC: 0.9264
PR-AUC: 0.6769

============================================================
Training GAT
============================================================
GAT(
  (conv1): GATConv(11, 64, heads=4)
  (conv2): GATConv(256, 2, heads=1)
)
Epoch 001 | train_loss=0.7186 | val_loss=0.6732
Epoch 025 | train_loss=0.2399 | val_loss=0.1941
Epoch 050 | train_loss=0.1720 | val_loss=0.1561
Epoch 075 | train_loss=0.1487 | val_loss=0.1302
Epoch 100 | train_loss=0.1337 | val_loss=0.1136
Epoch 125 | train_loss=0.1250 | val_loss=0.1088
Epoch 150 | train_loss=0.1220 | val_loss=0.1048
Epoch 175 | train_loss=0.1171 | val_loss=0.1006
Epoch 200 | train_loss=0.1108 | val_loss=0.0963
Epoch 225 | train_loss=0.1098 | val_loss=0.0917
Epoch 250 | train_loss=0.1072 | val_loss=0.0892
Epoch 275 | train_loss=0.1056 | val_loss=0.0864
Epoch 300 | train_loss=0.1017 | val_loss=0.0859
Epoch 325 | train_loss=0.1003 | val_loss=0.0837
Epoch 350 | train_loss=0.1003 | val_loss=0.0820
Epoch 375 | train_loss=0.0943 | val_loss=0.0779
Epoch 400 | train_loss=0.0952 | val_loss=0.0763
Epoch 425 | train_loss=0.0922 | val_loss=0.0744
Epoch 450 | train_loss=0.0927 | val_loss=0.0728
Epoch 475 | train_loss=0.0933 | val_loss=0.0717
Epoch 500 | train_loss=0.0890 | val_loss=0.0706
Epoch 525 | train_loss=0.0853 | val_loss=0.0693
Epoch 550 | train_loss=0.0858 | val_loss=0.0681
Epoch 575 | train_loss=0.0870 | val_loss=0.0667
Epoch 600 | train_loss=0.0864 | val_loss=0.0660

Best epoch: 600
Best validation loss: 0.0660

Test metrics:
Precision: 0.8438
Recall: 0.3857
F1: 0.5294
ROC-AUC: 0.9460
PR-AUC: 0.6620
```

Why is recall so weak here? Because the threshold is set high, so the model only calls something "fraud" when it's overly confident. On top of that, the synthetic dataset I generated for this example has very few actual fraud cases to learn from.

Now, after all this pain, let's continue toward our more advanced GNNs — and how to secure them.

(Here's the `pyproject.toml`, for completeness):
```toml
[project]
name = "fraud-gcn"
version = "0.1.0"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

then:
```shell
pip install -e .
```

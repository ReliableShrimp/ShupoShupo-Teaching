
> This is a really important topic, y'know? We're going to use it constantly, so...
>
> We're not just covering the basics, because basics are boring — we're going from basics all the way to an intermediate or advanced level. So... we're going to suffer? Yeah, a lot. Because you will forget a lot of what you already learned by the time we will advance.

# Table of contents

Chapter 1
1. Tensor
2. PyTorch Commands and Socrates Moments
3. Tensor From the Inside
    1. Building Tensors With Bare Hands
4. Basic Commands and Ideas
    1. torch.tensor
	2. torch.zeros, torch.ones
	3. torch.rand, torch.randn, torch.arange
5. Data Types of Tensors
	1. torch.int64
	2. torch.float32
	3. torch.bfloat16
	4. torch.bool
6. Tensor Indexing
	1. 1D
	2. 2D
7. Tensor Operations
8. Reshaping Tensors
	1. reshape()
	2. .view()
	3. flatten()
	4. unsqueeze()
	5. transpose() and permute()
9. CPU vs GPU: cuda, device, and .to()
	1. device
	2. CUDA
	3. to()
10. Computation Graph
11. .backward() and .grad
12. Normalize
13. autograd.Function

2. Chapter 2
1. nn.Module
2. Initializations
	1. Glorot (Xavier) Initialization
	2. He (Kaiming) Initialization
3. Optimization
4. Criterion
5. Mixed Precision Training (torch.amp)
6. DataLoader + Dataset

Projects of Chapter 1 and Chapter 2

# Chapter 1

## Tensor

So, what is a Tensor? Imagine this...

In Python, we might have:

```python

x = 12
# That's just a simple number

y = ['15']
# That's just a python list

z = np.array([1, 2, 3, 4])
# That's just a vector

w = np.array([[1, 2], [3, 4]])
# That's just a matrix
```

But what are these, really? They're all **tensors**. Why? Because a tensor is simply a general name for data organized into one or more dimensions.

The easiest way to picture this is with Lego: a single brick, a row of bricks, a wall of Lego, many walls of Lego stacked together. At every step, you're still working with Lego blocks — just organized differently.

Suppose we have a list of 10 million numbers. Why not just use a Python list? Sure, a Python list *can* hold them — but a PyTorch tensor is purpose-built for this. A PyTorch tensor is designed to be:

- Fast for mathematical operations.
- Memory-efficient.
- Able to run on a GPU, where thousands of calculations happen in parallel.
- Able to track gradients, which is essential for training neural networks.

We'll look at how each of these works. So, to sum up: a tensor is a container for numbers arranged in one or more dimensions.

## PyTorch Commands and Socrates Moments

Now imagine you've mastered PyTorch, but one day Archimedes shows up at 5:00 AM and wakes you up to ask a deep philosophical question:

> "By the immortal levers of Syracuse, pray enlighten me: why must we abandon the steady sands of NumPy to worship at the altar of this PyTorch Tensor?"
>
> $\downarrow$
>
> Simple translation: "Why would you use a PyTorch Tensor instead of a NumPy array?"

Such a deep philosophical question — yet the answer is simple.

Yup, NumPy can do the math just fine — multiplication, statistics, linear algebra, all of it. So why PyTorch Tensors?

NumPy was built for numerical computing. PyTorch was built for machine learning.

Already sounds more interesting — let's see exactly how.

For example: Airi Sezaki is building a hospital AI, so she starts with something simple:

```python
import numpy as np

X = np.array([
    [36.8, 70],
    [39.1, 120],
    [37.0, 82]
])
```

She wants to predict whether someone is:

```
sick

or

healthy
```

So she writes $y = Wx + b$.

Fine — but she also needs gradient descent:

```math
\frac{\partial L}{\partial W}
```

This has to be computed. Every iteration. For every parameter.

So with 2 weights, everything's fine. With 50 weights, still fine (though her hand hurts). With 10 million weights? Not so much.

Doing this all by hand for every parameter, every iteration, would be an absolute nightmare — funny as a joke, not funny in real life.

**1. Memory**

*NumPy's limitation:*

NumPy only sees numbers:

```python
import numpy as np

x = np.array([2.0])
y = x * 5 + 3
```

Its entire job is literally:

```
2.0

*

5

+

3

=

13
```

and that's it — it knows nothing beyond that.

*PyTorch:*

Here's the deal — PyTorch actually remembers what it did:

```
Start

Multiply by 5

Add 3

Answer
```

Why do we care? Because now we can ask: "How much did changing `x` affect `y`?" That's exactly what a derivative measures.

> [!TIP]
> If we compute `5 + 5` in NumPy, the moment we move on, it's completely forgotten. PyTorch remembers, because it stores a **computational graph** (we'll learn about this soon), which lets it compute both $`y`$ and $`\frac{dy}{dx}`$.

**2. The CPU**

NumPy is great at raw calculation, but if we write `x * y` and suddenly hear the computer's fans spinning up like a jet about to take off — that's because NumPy dumps everything onto the CPU and just hopes it doesn't hit a bottleneck. Why might this happen? Because under the hood, the machine might be doing far more operations than expected. Multiplying something 100 times? Fine. A billion times? Scary.

That's not the case with PyTorch, though — with a simple command like:

`x = x.to("cuda")`

PyTorch sends the tensor straight to the GPU.

Why does that matter? Imagine you have two massive tensors of size 10,000 × 10,000 — just multiplying them together could require roughly **1 trillion** multiplications and additions.

**NumPy** — the CPU (the PhD mathematician): it looks at the problem, thinks, and answers after about 10 seconds (limited by how many cores it has).

**PyTorch** — the GPU (thousands of kindergarten teachers): none of them are individually brilliant, but split the multiplication across thousands of them, and they all finish together, far faster than the CPU (since the GPU distributes the work across all its cores). The same operation might finish in 0.20 seconds, or less.

We need PyTorch because a neural network performs billions, even trillions, of operations.

**3. Neural Networks**

Suppose Airi wants a single neural layer.

With NumPy, she'd need to implement:

- initialize weights
- initialize bias
- matrix multiplication
- activation functions
- gradients
- optimizer
- parameter updates

Everything, by hand.

With PyTorch, she can just do:

```
layer = nn.Linear(128, 64)
```

Done.

PyTorch already provides building blocks purpose-built for deep learning.

That doesn't mean we should forget about NumPy — we'll still reach for it any time we want general numerical computing. That's why we should stick with NumPy when we're just doing `2 + 2`.

But anyway... what's actually inside a tensor?

## Tensor From the Inside

What's actually the difference between:

```python
x = [2, 5, 8]
```

which is:

```
Python List

┌──────────────┐
│ 2│ 5│ 8│
└──────────────┘
```

and:

```python
x = torch.tensor([2, 5, 8])
```

which is:

```
PyTorch Tensor

┌──────────────────────────┐
│ Tensor                   │
│                          │
│ 2  5  8                  │
│                          │
│ + extra information      │
└──────────────────────────┘
```

We know a tensor is different — but what does it actually have inside it?

A normal Python list is just a Python list. A tensor is a list that also carries extra information alongside it.

For example, a patient doesn't just have a name — they also have an age, a size, a weight, a height, and so on. They have *attributes*.

PyTorch tensors have attributes too:

```python
x = torch.tensor([2, 5, 8])
```

It has:

- shape
- data type
- device
- number of dimensions
- and more.

Let's look at one for now: `shape`.

```python
print(x.shape)

"""
Output:

torch.Size([3])
"""
```

What does that mean? It tells us how many elements exist along each dimension. A vector has just one dimension, so its shape has just one number.

What if we make it a matrix?

```
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

which is literally:

```
      Columns

      1   2   3
    ┌───┬───┬───┐
Row1│ 1 │ 2 │ 3 │
    ├───┼───┼───┤
Row2│ 4 │ 5 │ 6 │
    └───┴───┴───┘
```

```python
print(A.shape)

"""
Output:

torch.Size([2, 3])
"""
```

2 rows, 3 columns. Simple enough — but worth making sure it's cemented in.

Say you load an image dataset. The first thing you'd probably do:

```python
print(image.shape)

"""
Output:

torch.Size([32, 3, 224, 224])
"""
```

So now you know (and can recognize) exactly what you're working with:

```
32 images
↓
3 color channels (RGB)
↓
224 pixels tall
↓
224 pixels wide
```

Seeing a tensor's shape for the first time can feel intimidating, so let's demystify it. I'll show you how to read the shape of any tensor.

```python
x = torch.tensor([
    [
        [1,2],
        [3,4]
    ],
    [
        [5,6],
        [7,8]
    ]
])
```

What shape does this have? Let's look at it!

How many blocks is it made of?

```
[[
 [Block 0]
]
[ 
 [Block 1]
]]
```

So we already know the first number:

`(2...)`

Now let's look inside block 0:

```python
[
 [1,2],
 [3,4]
]
```

How many rows does it have? 2. How many columns? 2.

Putting it together:

```
torch.Size([2,2,2])
```

> [!TIP]
> To read a tensor's shape, count from the outside in: how many top-level blocks, then how many rows inside each block, then how many columns inside each row.

That's how we get the right shape. But recognizing shapes isn't enough on its own — we need to know how to build them with our bare hands. Let's get to it!

### Building Tensors With Bare Hands

We already know the idea of samples and features: more samples means more rows, more features means more columns.

Just in case that's not fully clear yet (I doubt it, but just in case), here's a small example.

You have a basket, and in it, 3 apples:

```python
basket = torch.tensor([1, 2, 3])

print(basket.shape)

"""
Output:

torch.Size([3])
"""
```

Now imagine you have a shelf holding 3 baskets, where each basket is `[total_pears, total_apples, total_oranges]`. In Python:

```python
basket = torch.tensor([
[3, 8, 2], # Basket 1: has 3 pears, 8 apples, and 2 oranges
[12, 4, 1], # Basket 2: has 12 pears, 4 apples, and 1 orange
[1, 6, 9] # Basket 3: has 1 pear, 6 apples, and 9 oranges
])
```

But what if we go to a warehouse with 3 shelves, and each shelf has 3 baskets of pears, apples, and oranges?

```python
import torch

# Shape: (3, 3, 3) -> 3 Shelves, 3 Baskets per shelf, 3 Fruit types per basket
warehouse = torch.tensor([
    [
        [3, 8, 2],   # Basket 1: 3 pears, 8 apples, 2 oranges
        [12, 4, 1],  # Basket 2: 12 pears, 4 apples, 1 orange
        [1, 6, 9]    # Basket 3: 1 pear, 6 apples, 9 oranges
    ],
    [
        [5, 5, 5],   # Basket 1
        [0, 10, 2],  # Basket 2
        [7, 3, 8]    # Basket 3
    ],
    [
        [8, 2, 1],   # Basket 1
        [15, 0, 4],  # Basket 2
        [2, 9, 6]    # Basket 3
    ]
], dtype=torch.int64)
```

But what if we have 2 warehouses? In Python:

```python
import torch

warehouses = torch.tensor([
    # Warehouse 1
    [
        [[3, 8, 2], [12, 4, 1], [1, 6, 9]], # Shelf 1
        [[5, 5, 5], [0, 10, 2], [7, 3, 8]], # Shelf 2
        [[8, 2, 1], [15, 0, 4], [2, 9, 6]] # Shelf 3
    ],
    
    # Warehouse 2
    [
        [[4, 9, 1], [10, 2, 0], [2, 5, 8]], # Shelf 1
        [[6, 6, 6], [1, 11, 3], [8, 4, 9]], # Shelf 2
        [[9, 3, 2], [16, 1, 5], [3, 10, 7]] # Shelf 3
    ]
], dtype=torch.int64)

print("Shape:", warehouses.shape) 

"""
Output:

torch.Size([2, 3, 3, 3])
"""
```

And I'll say this immediately: don't read `torch.Size([2, 3, 3, 3])` as "two threes and three threes." No — read it in terms of what the data actually represents: "We have 2 warehouses, each containing 3 shelves, each shelf containing 3 baskets, each basket holding 3 features."

## Basic Commands and Ideas

Let's learn some fundamental PyTorch commands, since we'll see them constantly — and without understanding them, the rest will be hard to follow. (No worries though, it'll still be a bit hard even *with* these commands.)

### `torch.tensor`

We've basically already met this one — it just builds a tensor, nothing too special. But it does have a few nuances worth knowing:

```python
shelf = [
    [3, 8, 2],
    [12, 4, 1],
    [1, 6, 9]
]

x = torch.tensor(shelf)
```

We just converted a Python list into a tensor! PyTorch figures out the structure on its own from this list:

```
Shelf

Basket 1
Basket 2
Basket 3
```

and builds the matching tensor. But what about higher dimensions?

```python
warehouse = [
    [
        [3, 8, 2],
        [12, 4, 1],
        [1, 6, 9]
    ],
    [
        [5, 5, 5],
        [0, 10, 2],
        [7, 3, 8]
    ]
]

x = torch.tensor(warehouse)
```

It builds another tensor from this too — PyTorch doesn't care whether it's a vector, a matrix, or a 7-dimensional tensor. It just builds the matching structure.

There are some rules, though, as usual — the shape has to be consistent, which makes sense:

```python
x = torch.tensor([
[1, 2, 3],
[4, 5]
])
```

That's a clear red flag, and PyTorch will throw:

```Python
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

Which translates to: "Dude, I expected every single row to have the exact same number of items! You gave me 3 items in the first row, so I locked in that length for the whole dimension — then you handed me 2 items in the next row!"

That's PyTorch's version of a stern warning to us.

### `torch.zeros`, `torch.ones`

You probably already know what these mean — we can basically read off what they do.

```python
torch.zeros()
```

This is literally our beloved `np.zeros()` — it creates a tensor filled with 0s, across as many rows and columns as we want.

For example:

```python
import torch

warehouse = torch.zeros(3, 3)
print(warehouse)

"""
Output:

tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
"""
# The shape is (3, 3)
```

What if we want a 3D tensor? Just do:

```python
x = torch.zeros(2, 3, 4)
```

and we get 2 shelves, each with 3 baskets, each holding 4 features.

Where do we actually use these? Really often, for things like:

- Initializing an image
- An empty matrix
- Creating predictions
- Reserving memory
- Starting an accumulator

What about `torch.ones`? We use it more rarely, but it does the exact same thing — just with 1s instead:

```python
x = torch.ones(2,3)

print(x)

"""
Output:

tensor([[1., 1., 1.],
        [1., 1., 1.]])
"""
# Shape: (2, 3)
```

Where do we use it?

- Masks
- Scaling values
- Multiplication identities
- Initial values

### `torch.rand`, `torch.randn`, `torch.arange`

Here's where the RNG starts — but sadly, the "not-so-RNG" doesn't range from -infinity to infinity, because that would be a mess nobody could actually use. That's why these all come with limits.

Let's start with `torch.rand`.

What's better than random numbers stuck consistently between 0 and 1? It gives you random numbers within exactly that range. Look:

```python
import torch

w = torch.rand(6, 3)

"""
Output (will be totally different next time I run the code):

tensor([[0.2217, 0.2059, 0.4587],
        [0.4020, 0.8286, 0.6479],
        [0.8227, 0.5383, 0.1917],
        [0.7997, 0.5655, 0.7805],
        [0.2903, 0.3888, 0.7785],
        [0.2915, 0.1363, 0.5748]])
"""
```

Now we have some casual weights, but the output is different every time. You'll usually want to do this instead:

```python
import torch

torch.manual_seed(9) # We pick a random number
w = torch.rand(6, 3)
"""
Output (always identical now, since the randomness is pinned down):

tensor([[0.2217, 0.2059, 0.4587],
        [0.4020, 0.8286, 0.6479],
        [0.8227, 0.5383, 0.1917],
        [0.7997, 0.5655, 0.7805],
        [0.2903, 0.3888, 0.7785],
        [0.2915, 0.1363, 0.5748]])
"""
```

But as we've noticed, these outputs (as I've now said 20 times) are uniformly distributed between 0 and 1 — meaning every number in that range has an equal chance of showing up. That can actually be pretty annoying. Another reason it bugs people: there are no negative numbers at all, which introduces some nuances that can quietly break a model.

That's where `torch.randn()` comes in — it doesn't stay strictly positive, it can go negative too. It uses an idea you're probably already familiar with: the **normal (Gaussian) distribution**. Here's roughly how the values are distributed:

- Within ±1: about 68.27% of generated values fall between -1 and 1.
- Within ±2: about 95.45% fall between -2 and 2.
- Within ±3: about 99.73% fall between -3 and 3.

Here's the same example from before, so you can see the difference:

```python
import torch

torch.manual_seed(9) # We pick a random number
w = torch.randn(10, 5)

"""
Output:

tensor([[ 2.1348, -0.1058,  0.3694,  1.5215, -0.0518],
        [-0.0327, -0.5396, -0.9374, -0.3384,  0.7098],
        [ 0.4642, -0.1623, -1.6702,  0.7307, -0.3857],
        [-1.2325,  2.3819, -0.3380,  0.7563, -0.7850],
        [-0.8291,  0.8074, -0.0245, -0.7167,  0.8508],
        [-1.2331,  0.7439,  0.7015,  0.6960,  0.5133],
        [-1.4411, -0.2088, -1.2412, -0.1886,  1.2032],
        [-0.0644,  0.0294,  0.7495, -2.2645,  0.2902],
        [ 0.2220,  1.2037,  1.4792, -1.8700, -0.6055],
        [ 0.1064, -0.4964,  0.8080, -1.0390,  1.6026]])
"""
```

What about `torch.arange`? It works exactly like Python's `range`, except instead of building a plain list, it builds a tensor:

```
torch.arange(start, end, step)
```

We pick a starting number, an ending number, and the distance between steps. (Think of it as: "count from start to end, stopping every `x` steps.")

For example:

```python
import torch

# x = torch.arange(start, end, step)
x = torch.arange(0, 10, 2)
print(x)

"""
Output:

tensor([0, 2, 4, 6, 8])
"""
```

Another way to use it (start from x, stop *before* y):

```python
import torch 

# x = torch.arange(x, y)
x = torch.arange(2, 8)
print(x)

"""
Output:

tensor([2, 3, 4, 5, 6, 7])
"""
```

And another way:

```python
import torch

# start defaults to 0 if omitted, and step defaults to 1
x = torch.arange(5)
print(x)

"""
Output:

tensor([0, 1, 2, 3, 4])
"""
```

What if we want floats? Simple:

```python
import torch

x = torch.arange(0, 1, 0.2)
print(x)

"""
Output:

tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])
"""
# Remember, the end value is never included
```

And negative steps:

```python
import torch

x = torch.arange(5, 0, -1)
print(x)

"""
Output:

tensor([5, 4, 3, 2, 1])
"""
```

What's all of this useful for?

- looping over tensors
- creating labels
- generating positions
- building masks
- positional encodings in transformers
- plenty of other tensor operations

## Data Types of Tensors

Now let's talk about the beautiful `dtype` you've already seen in NumPy, which I never actually explained (because I'm a little silly). Let's fix that, and figure out what to use, and when.

### `torch.int64`

What does this even mean? Let's break it into pieces:

- **int** — just an integer (remember, an integer is a whole number. If you try to store `3.14` in a tensor with dtype `int64`, PyTorch will automatically chop off the decimal part and store it as `3`).
- **64** — the number of bits used to store a single value. In this case, `64 bits = 8 bytes of RAM per number`. It can store an astronomically large range of numbers — roughly 18 quintillion distinct values, spanning from $`-2^{63}`$ to $`2^{63}-1`$.

`torch.int64` can be positive or negative, but it can never hold a decimal.

### `torch.float32`

Let's break this one down too:

- **float** — this lets us keep the decimals that get chopped off with `int`, which makes it essential in practice. When we're working with random weights or biases, `int` won't cut it, since there's no guarantee the number will be whole. That's why we'll use `torch.float32` most of the time.
- **32** — this tells us a single number uses 4 bytes. It can represent numbers as small (in magnitude) as $`1.4 \times 10^{-45}`$ and as large as $`3.4 \times 10^{38}`$, in either sign.

Now you might ask: why use `float32` when we could just use `float64`? Great question — but here's the reality check: `float64` (double precision) is significantly slower on most consumer GPUs.

Say we have 1 billion parameters. Which is the better choice?

- **`float32`** — a great option. It only takes up 4 bytes of memory per number, keeping VRAM usage low and computation relatively fast.
- **`float64`** — not a great option. It takes 8 bytes per number, and with 1 billion parameters already in play, that doubles our VRAM usage and slows everything down considerably.

That's the big difference between the two. So the more weights we're working with, the smaller a type we'll typically reach for — all the way down to `float16`, `int4`, and beyond.

But for our neural networks, we'll be spamming `float32` most of the time.

### `torch.bfloat16`

This one's another fan favorite, used for training large models. Even with half the precision, it's still reliable, and by far the most-used type once parameter counts get large.

Why not just use `float16` instead of `bfloat16`? Because `float16` is too weak for big models. It has 1 sign bit, 5 exponent bits, and 10 mantissa bits — and on top of that, it only works reliably for numbers smaller than 65,504. So as soon as numbers in training get too large, everything crashes.

What about `bfloat16`? It has 1 sign bit, 8 exponent bits, and 7 mantissa bits. Most importantly, it can reach a massive range of numbers (up to $`3.4 \times 10^{38}`$) — you'll basically never hit "Infinity/NaN" with it.

So here's the final comparison:

| Type | Exponent (Range) | Mantissa (Precision) | Best For |
| --- | --- | --- | --- |
| **`float32`** | 8 | 23 | Precision-critical small models |
| **`float16`** | 5 | 10 | Inference (running models quickly) |
| **`bfloat16`** | 8 | 7 | Large-scale training (stability & speed) |

That's how it works.

### `torch.bool`

We reach for this one mainly when building masks. While `float32` and `bfloat16` are all about values and data, `bool` is about logic. It's the perfect fit for boolean operations, which is exactly why it shows up whenever we do things like:

```python
data = torch.tensor([
    [15, 20, 0],
    [5,  0,  0]
])

# Create a boolean mask: Where is the data NOT zero?
mask = (data != 0)

"""
Output:

tensor([[ True,  True, False],
        [ True, False, False]])
"""
```

That's how simple the `bool` idea is.

## Tensor Indexing

This is an important idea (don't worry, I'll keep repeating it until it sticks), because we'll use it constantly — in graph neural networks, transformers, images, and more.

Imagine a warehouse of boxes, where each box holds a number. That's just a vector.

### 1D

```
Warehouse

┌────┬────┬────┬────┐
│ 12 │  8 │ 15 │ 20 │
└────┴────┴────┴────┘
```

```python
import torch

warehouse = torch.tensor([12, 8, 15, 20])
```

Say we need to grab the number 20 specifically — we need the box that strictly holds 20. What do we do? We just index it:

```python
mask = warehouse[3]
print(mask)

"""
Output:

tensor(20)
"""
```

But why did we write 3? Isn't 20 the fourth number? Yes — but sadly, Python counts from 0.

```
Boxes

Value
┌────┬────┬────┬────┐
│12  │ 8  │15  │20  │
└────┴────┴────┴────┘

Index
┌────┬────┬────┬────┐
│ 0  │ 1  │ 2  │ 3  │
└────┴────┴────┴────┘
```

> [!TIP]
> If you want the *N*th item counting normally (1st, 2nd, 3rd...), the index you actually use is *N* − 1, since indexing starts at 0.

Python can also count backward (apparently counting forward by index is hard, but counting backward isn't). We'll use that sometimes too, for example:

```python
print(warehouse[-1])

"""
Output:

tensor(20)
"""
# It always returns the last number.
```

But what do we do when a monster of a matrix shows up? How do we index that?

### 2D

Say a new shelf appears in our warehouse:

```
Shelf 0
12   8   15

Shelf 1
20   9   11

Shelf 2
30   7   14
```

We count from zero here too — we're trying to become one with Python. (Joking. Please don't, unless you have strange tastes.)

```
warehouse = torch.tensor([
    [12, 8, 15],
    [20, 9, 11],
    [30, 7, 14]
])
```

Say we want just the value 9.

```python
print(warehouse[1, 1])

"""
Output:

tensor(9)
"""
```

As usual, Python counts both rows and columns from 0:

```
       0    1    2
    ┌────┬────┬────┐
0   │12  │ 8  │15  │
    ├────┼────┼────┤
1   │20  │ 9  │11  │
    ├────┼────┼────┤
2   │30  │ 7  │14  │
    └────┴────┴────┘
```

That's how Python sees our matrix.

What if we need the column `[8, 9, 7]`? We index it the same way:

```python
print(warehouse[:,1])

"""
Output:

tensor([8,9,7])
"""
```

What is `:` here? Think of it as "give me everything from there." Since we put it in the row slot, we're saying "give me every row, from this column..."

What if we want every column from just the first two rows? We do this:

```python
print(warehouse[:2, :])

"""
Output:

tensor([[12,  8, 15],
        [20,  9, 11]])
"""
```

We used `[:2, :]` to say: "give me every row up to (but not including) index 2, and every column."

There are other ways to index too:

`[:1, 2:]` means: "give me every row up to (not including) index 1, and every column from index 2 onward":

```python
print(warehouse[:1, 2:])

"""
output:

tensor([[15]])
"""
```

Let's use a new warehouse to explain this better:

```python
import torch

warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]])

print(warehouse[:1, 3:])
# This means: give me just row 0 (the first row), and exclude the first 3 columns (so just the last one).

"""
output:

tensor([[35]])
"""
```

```python
print(warehouse[2:, 2:]) 
# "Exclude the first two rows, and exclude the first two columns"

"""
output:

tensor([[ 90,  95],
        [120, 125]])
"""
```

Now say we want an exact row — like `60, 65`:

```python
import torch

warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]])

print(warehouse[[1], 2:])
# "Give me just row index 1 (the second row), and exclude the first 2 columns"

"""
Output:

tensor([[60, 65]])
"""
```

Another example:

```python
print(warehouse[[0, 3], :2])
# "Give me the first 2 columns (stop before index 2) of row 0 and row 3"

"""
Output:

tensor([[ 10,  20],
        [100, 110]])
"""
```

```python
print(warehouse[[2], [2]])

"""
Output:

tensor([90])
"""
```

And the boolean version — using that same warehouse:

```python
warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]
])

print(warehouse > 60)

"""
Output:

tensor([[False, False, False, False],
        [False, False, False,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]])
"""
```

This one is by far one of the most useful indexing tricks around.

We'll cover the other types of indexing and slicing a few chapters from now, since there's no point showing them off without a real use for them yet. I'll bring them back up exactly when we need them.

## Tensor Operations

This part is way too easy for us at this point, so I'll give it a quick sketch and move on.

"But if it's so easy, why even bother showing us?"

Because with tensors, you won't always be reaching for `torch.mean()` or `torch.sum()` explicitly — you'll see why in a moment.

---

Let's start with the full basics:

**1. Element-wise operations** — the tensors need to have the same shape, and the operation happens between each pair of matching elements:

```python
import torch

A = torch.tensor([[1, 2], 
                  [3, 4]])

B = torch.tensor([[5, 6], 
                  [7, 8]])

# 1. Addition (+)
print(A + B)
# [[1+5, 2+6],   -> [[ 6,  8],
#  [3+7, 4+8]]       [10, 12]]

# 2. Subtraction (-)
print(B - A)
# [[5-1, 6-2],   -> [[4, 4],
#  [7-3, 8-4]]       [4, 4]]

# 3. Multiplication (*)
print(A * B)
# [[1*5, 2*6],   -> [[ 5, 12],
#  [3*7, 4*8]]       [21, 32]]

# 4. Division (/)
print(B / A)
# [[5/1, 6/2],   -> [[5.0, 3.0],
#  [7/3, 8/4]]       [2.3, 2.0]]
```

---

**2. Matrix multiplication**

You can write `torch.matmul()`, or — as we usually do — just `@`.

```python
print(A @ B)

"""
output:

tensor([[19, 22],
        [43, 50]])
"""
```

The only rule we have to respect here is broadcasting.

---

**3. Reduction operations (`sum()`, `mean()`, `max()`)**

In NumPy, we always had to write `np.mean()`, `np.sum()`, and so on. In PyTorch, we do:

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

# 1. sum()
print(x.sum())   # tensor(21.)

# 2. mean()
print(x.mean())  # tensor(3.5000)

# 3. max()
print(x.max())   # tensor(6.)
```

You just write `variable.sum()` directly — as long as `variable` is a tensor (which means you need to have imported `torch`).

---

**4. Reducing across a specific axis (`dim=`)**

Say we have this bigger matrix:

```python
import torch

A = torch.tensor([
[1, 21, 3, 4, 52, 68],
[71, 8, 19, 199, 11, 9],
[100, 11, 15, 16, 57, 18],
[19, 65, 92, 2, 23, 10]
])
```

What if we want to check by column only, or by row only?

```python
# dim = 0 checks only the columns
print(A.max(dim=0))

"""
torch.return_types.max(
values=tensor([100,  65,  92, 199,  57,  68]),
indices=tensor([2, 3, 3, 1, 2, 0]))
"""
# It checked each column and pulled the biggest number from each.
# The indices part tells us which row each maximum was found in.

# dim = 1 checks only the rows
print(A.max(dim=1))

"""
torch.return_types.max(
values=tensor([ 68, 199, 100,  92]),
indices=tensor([5, 3, 0, 2]))
"""
# It checked each row and printed only the biggest number.
# The indices part tells us which column each maximum was found in.
```

---

Now we're entering a dangerous topic — so get ready, grab some napkins, because it's about to be boring *and* painful.

## Reshaping Tensors

Sounds scary, but this is our bread and butter, so let's dive straight in. (I'll try to make this feel less intimidating.)

### `reshape()`

Imagine Airi has 20 Lego pieces. What shapes can she build?

- 10 rows of 2: `10x2` — 10 rows, 2 columns
- 2 rows of 10: `2x10` — 2 rows, 10 columns
- 2 blocks of 5 rows and 2 columns: `2x5x2` — 2 blocks, 5 rows, 2 columns
- 5 blocks of 2 rows and 2 columns: `5x2x2` — 5 blocks, 2 rows, 2 columns

To check whether a shape is even possible, just multiply every part of the shape together (rows × columns, block × rows × columns, and so on) — the product has to equal the original total. Let's try it in Python!

```python
import torch

A = torch.arange(20)

print(A)

"""
Output:

tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19])
"""
```

Say I want 2 blocks of 2 rows, 5 columns each. I'll do:

```python
A = A.reshape(2,2,5)

# Quick sanity check that we can actually do this:
# 2 * 2 * 5 = 20... we can!

"""
Output:

tensor([[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9]],

        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]]])
"""
```

Now we've reshaped it exactly as we wanted! But what if we want 2 groups, with 2 blocks each, and 5 rows of just 1 value?

```python
A = A.reshape(2, 2, 5, 1)

# Quick sanity check:
# 2 * 2 * 5 * 1 = 20... we can!

"""
Output:

tensor([[[[ 0],
          [ 1],
          [ 2],
          [ 3],
          [ 4]],

         [[ 5],
          [ 6],
          [ 7],
          [ 8],
          [ 9]]],


        [[[10],
          [11],
          [12],
          [13],
          [14]],

         [[15],
          [16],
          [17],
          [18],
          [19]]]])
"""
```

That's how `.reshape()` reshapes our tensor!

There's one catch, though: `.reshape()` copies the data in VRAM when the tensor is **non-contiguous** (for example, after being changed by `.t()` or `.permute()`), which means it can quietly eat up a lot of memory without you noticing. But it won't copy anything if the tensor **is** contiguous.

So it breaks down like this:

| Condition of Tensor | Does `.reshape()` copy data? | What does it return? |
| --- | --- | --- |
| Contiguous | No | A view (shares memory) |
| Non-contiguous | Yes | A copy (brand-new memory) |

That's why the next tool exists — as a helper.

### `.view()`

This is identical to `reshape()` in what it produces — so why use it instead?

`.view()` immediately crashes if used on a non-contiguous tensor (for example, right after a `.t()` or `.permute()`). But why would we *want* that instead of just letting reshape handle it quietly? Because, as mentioned, `.reshape()` silently copies the entire tensor to a new spot in memory to force it into the new shape.

- If your tensor is a 2 GB batch of high-resolution images, `.reshape()` will silently allocate *another* 2 GB of VRAM and spend time copying all that data.
- If this happens inside your training loop (running maybe 1,000 times a minute), your GPU will either run out of memory instantly (an OOM error) or your training speed will crawl to a halt.

Using `.view()` acts like a smoke detector. If your memory layout is messy, it screams (crashes) immediately, forcing you to fix the underlying issue right away instead of quietly limping along at terrible speed.

That's why professionals default to `.view()`, and if it crashes, they switch to `.contiguous().view()` instead — we'll cover that soon. For now, stick with `.view()`.

Now let's learn another concept.

### `flatten()`

Suppose we have a matrix like:

```math
\begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \\ 8 & 9 & 10 & 11 \\ 12 & 13 & 14 & 15 \\ 16 & 17 & 18 & 19
\end{bmatrix}
```

As we go further, we'll often need to work with a vector instead of a matrix. Breaking it apart by hand is a pain, so instead, we use `.flatten()`:

```python
A = torch.tensor([
[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15],
[16, 17, 18, 19]
])

A = A.flatten()
print(A)

"""
Output:

tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19])
"""
```

As we can see, it's now a simple vector. Before you dismiss this as useless, let me tell you exactly why it matters: it's the bridge between CNNs and linear layers.

In CNNs (convolutional neural networks, used in computer vision), your image starts out as 4D — but sadly, the classifier layer only accepts a 2D tensor: `[Batch_Size, Features]`. That's why we use `flatten()` to crush the tensor down into a 1D vector. But since we still need to keep a 2D result, we use `start_dim=1`, which tells PyTorch: "leave the batch size alone, but crush the channels, height, and width into one long line."

```python
import torch

# A batch of 32 images, 3 color channels, 28x28 pixels
cnn_output = torch.randn(32, 3, 28, 28)

# Crush everything EXCEPT the batch dimension (dim 0)
flat_features = torch.flatten(cnn_output, start_dim=1)

print(flat_features.shape)
# Output: torch.Size([32, 2352])  <- (3 * 28 * 28 = 2352)
```

That's exactly why this is so useful. Now let's move on to something a bit more mind-melting — because life is hard, and making it harder is always better.

### `unsqueeze()`

This part's tricky, because `unsqueeze()` alters a tensor's dimensions by inserting a fake one. Why would we ever want that? Because PyTorch is genuinely strict about dimension rules.

Imagine you've trained a neural network on batches of images. The model expects a 4D tensor shaped `[Batch_Size, Channels, Height, Width]`. If you try to pass in a single image (no batch dimension at all), it'll only have 3 dimensions: `[3, 28, 28]`. Feed that straight in, and your model crashes, complaining the dimensions don't match.

That's why we add a fake dimension.

For example, suppose we have a plain 2D tensor (matrix):

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
```

There's not much we can do with it directly — but say we want to treat it like a single image. Can we? Yes.

```python
import torch

A = torch.tensor([[1, 2], [3, 4]])

print(A)
print("")
print(f"The shape of A: {A.shape}")

"""
Output:

tensor([[1, 2],
        [3, 4]])
 
The shape of A: torch.Size([2, 2])
"""

---------------------------------------------------------------------------------
# Scenario 1: We used `unsqueeze(0)`
A = A.unsqueeze(0)
print(A)
print("")
print(f"The shape of A: {A.shape}")

"""
Output:

tensor([[[1, 2],
         [3, 4]]])
 
The shape of A: torch.Size([1, 2, 2])
"""
# As we can see, it added a dimension to the front — it added a "shelf."

---------------------------------------------------------------------------------
# Scenario 2: We used `unsqueeze(1)`
A = torch.tensor([[1, 2], [3, 4]])
A = A.unsqueeze(1)
print(A)
print("")
print(f"The shape of A: {A.shape}")

"""
Output:

tensor([[[1, 2]],

        [[3, 4]]])

The shape of A: torch.Size([2, 1, 2])
"""
# The original 2 rows each became their own block, and a new singleton "row"
# dimension got inserted in between.
# As we can see, it added a dimension in the middle.

---------------------------------------------------------------------------------
# Scenario 3: We used `unsqueeze(2)`
A = torch.tensor([[1, 2], [3, 4]])
A = A.unsqueeze(2)
print(A)
print("")
print(f"The shape of A: {A.shape}")

"""
Output:

tensor([[[1],
         [2]],

        [[3],
         [4]]])

The shape of A: torch.Size([2, 2, 1])
"""
# The original 2 rows became 2 blocks, each row became its own row again, and
# now each of those has just a single column.
```

The result looks like:

![Visualizing unsqueeze adding a new dimension](Pasted image 20260720191403.png)

We'll use this a lot — you'll see plenty of it going forward.

### `transpose()` and `permute()`

Using `transpose()`, you just swap the places of two dimensions. For example:

Say Airi has 2 baskets, each holding 20 apples — but she wants 20 baskets holding 2 apples each instead. That's exactly what `transpose()` is for:

```python
import torch

A = torch.tensor([
[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
[21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
]) # We have a shape of (2, 20) -> 2 baskets and 20 apples

# Swap dimension 0 (rows) and dimension 1 (columns)
A_t = A.transpose(0, 1)

print(A_t.shape) 
# Shortcut specifically for 2D matrices --->  x.t() 
print(A_t)

"""
Output:

torch.Size([20, 2])

tensor([[ 1, 21],
        [ 2, 22],
        [ 3, 23],
        [ 4, 24],
        [ 5, 25],
        [ 6, 26],
        [ 7, 27],
        [ 8, 28],
        [ 9, 29],
        [10, 30],
        [11, 31],
        [12, 32],
        [13, 33],
        [14, 34],
        [15, 35],
        [16, 36],
        [17, 37],
        [18, 38],
        [19, 39],
        [20, 40]])
"""

# We just swapped the places of the columns and the rows.
```

Now say we have 3 warehouses with 5 baskets each, but only 2 apples per basket — and we want to reorganize it so the warehouse dimension and the apple dimension trade places:

```python
import torch

  
A = torch.tensor([
[ # Warehouse 1
[ 1, 2], # Basket 1
[ 3, 4], # Basket 2
[ 5, 6], # Basket 3
[ 7, 8], # Basket 4
[ 9, 10] # Basket 5
],

[ # Warehouse 2
[11, 12],
[13, 14],
[15, 16],
[17, 18],
[19, 20]
],
  
[ # Warehouse 3
[21, 22],
[23, 24],
[25, 26],
[27, 28],
[29, 30]
]
])

# Swap dimension 0 (warehouses) and dimension 2 (apples)
A_t = A.transpose(0, 2)

print(A_t.shape)
print("")
print(A_t)

"""
Output:

torch.Size([2, 5, 3])

tensor([[[ 1, 11, 21],
         [ 3, 13, 23],
         [ 5, 15, 25],
         [ 7, 17, 27],
         [ 9, 19, 29]],

        [[ 2, 12, 22],
         [ 4, 14, 24],
         [ 6, 16, 26],
         [ 8, 18, 28],
         [10, 20, 30]]])
"""
# We swapped the warehouse dimension and the apple dimension, while leaving the basket dimension untouched.
```

What about `permute()`?

We use `permute()` because working with 3D, 4D, 5D+ tensors can get messy fast — it lets us rearrange every dimension at once, indexing-style, instead of swapping two dimensions at a time over and over:

```python
import torch
# A single PyTorch RGB image: [Channels, Height, Width]
image = torch.randn(3, 224, 224)
# Original indices:  0,   1,   2

# We want new order: Height (1), Width (2), Channels (0)
image_display = image.permute(1, 2, 0)

print(image_display.shape)

"""
Output:

torch.Size([224, 224, 3])
"""
```

As we can see, this lets us pick the new axis order in a much cleaner way.

But both of these come with some nuances worth remembering:

- Both make the resulting tensor non-contiguous.
- Calling `.view()` immediately after `transpose()` or `permute()` will crash (as we already know).

So be careful with them.

## CPU vs GPU: `cuda`, `device`, and `.to()`

Before we get into this topic, let's clarify the difference between a CPU and a GPU.

**1. CPU (Central Processing Unit)** — think of it as a group of 4 to 16 genius mathematicians. They're great at fast clock speeds, complex decision-making, conditional logic (`if`/`else`), and running the operating system. But as always, there's a catch: they do everything sequentially — one, or just a few, actions at a time.

**2. GPU (Graphics Processing Unit)** — think of it as 5,000 kindergarten kids. None of them can handle complex calculus, but every single one can do simple math (like `2 + 2` or `3 * 5`) at the exact same instant. (This is exactly why the computational graph breaks everything into small pieces, as we saw earlier.)

That's the big difference. But why does it matter to us? Because deep learning is packed with matrix multiplication (`@`) and element-wise math (`+`, `*`, and so on).

So imagine this: you want to add two vectors of 10,000 numbers each.

- A CPU runs a loop: adds item 1, then item 2, then item 3... 10,000 sequential steps.
- A GPU assigns each number to one of 10,000 tiny cores and adds them all in a single step.

> [!NOTE]
> Your machine has two separate memory pools that can't directly see each other: **system RAM** (attached to the CPU) and **VRAM** (dedicated memory glued to the GPU). The CPU can't directly reach into VRAM, and the GPU can't directly reach into system RAM.

System RAM is fast compared to a hard drive or SSD, but slow compared to VRAM.

System RAM typically comes in sizes like 8 GB, 16 GB, 32 GB, and so on. VRAM is usually smaller than system RAM (think 6 GB, 8 GB, 12 GB, or 24 GB).

Now let's build on that.

### `device`

Here's a question worth starting with: where does our tensor actually live?

Every tensor has a device, and it's typically either `cpu` or `cuda:0`. How do we check? Simple:

```python
x = torch.tensor([1,2,3])

print(x.device)

"""
Output:

cpu
"""
```

That literally means these numbers live in your computer's RAM.

But there's a key difference — the GPU has its own separate memory:

```
CPU
──────────────
RAM
│
│
└────Tensor A

GPU
──────────────
VRAM
│
│
└────Tensor B
```

As we can see, data stored in RAM is invisible to the GPU, and data stored in VRAM is invisible to the CPU. In other words, the CPU can't directly access GPU memory, and the GPU can't directly access system RAM.

### CUDA

Imagine you just bought a beautiful graphics card — an NVIDIA GPU.

Now ask yourself: how does Python even talk to it? Because Python has absolutely no idea how to communicate with a GPU on its own.

There needs to be a shared language — something like a universal translator. Right now, Python speaks "Python-ish," while the GPU speaks "GPU-ish." That translator is **CUDA**. It helps Python communicate with the GPU, working through a chain like:

```
Python
    ↓
PyTorch
    ↓
CUDA
    ↓
NVIDIA GPU
```

Think of CUDA as an operating system for your GPU. Your CPU runs Windows or Linux; your GPU needs its own software to receive jobs, schedule work, manage GPU memory, launch thousands of threads, and return results. CUDA provides all of that.

So what actually happens when PyTorch adds a million numbers?

```python
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")

z = x + y
```

Does Python add a million numbers itself? No. The process actually looks like:

```
You
↓
PyTorch
↓
CUDA
↓
NVIDIA Driver
↓
GPU
↓
Thousands of GPU cores
```

But here's the unfortunate part: only NVIDIA GPUs can use CUDA, since CUDA belongs to NVIDIA. So unlucky folks with AMD (that's me) or Intel hardware have to use something else:

```
NVIDIA → CUDA

AMD → ROCm / HIP

Apple → Metal

Intel → oneAPI
```

For AMD, the flow looks the same, just swapped out:

```
Your Code
     │
     ▼
PyTorch
     │
     ▼
ROCm
     │
     ▼
AMD GPU
```

But why does any of this matter to us? You're about to see why.

### `to()`

Say you want to change where a tensor lives — because by default, it's usually here:

```
┌─────────────────────┐
│      CPU Room       │
│                     │
│  📦 Tensor A        │
└─────────────────────┘


┌─────────────────────┐
│      GPU Room       │
│                     │
│                     │
└─────────────────────┘
```

When we do:

```python
x = torch.tensor([1, 2, 3])
```

it automatically lands in the CPU room. But what if we want it in the GPU room instead? Someone has to carry it over — that's where `.to()` comes in:

```python
x = x.to("cuda")
```

Even with AMD, `.to("cuda")` is still the command you use — PyTorch's ROCm build maps the `"cuda"` string onto AMD hardware behind the scenes, specifically so the same code works unchanged on both vendors. Apple Silicon is the one real exception here, since it uses `.to("mps")` instead, as we'll see shortly.

We can also use `torch.cuda.is_available()` to check whether a GPU is available at all — it returns `True` if so, and `False` if not.

For example:

```python
import torch

# Works identically on NVIDIA and AMD GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move a tensor to the GPU
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)

print(tensor.device)  # Returns 'cuda:0' if it was sent to the GPU
```

And so, the data moves like this:

```
            .to()

CPU RAM                    GPU VRAM
┌───────────┐             ┌───────────┐
│ 1  2  3   │ ─────────▶  │ 1  2  3   │
└───────────┘             └───────────┘

Same tensor values.
Different physical memory.
```

## Computation Graph

We already know the concept, but PyTorch handles it a bit differently — let's picture how.

Suppose you're building a normal neural network, and you have:

```python
y = ((x * w1) + b1) * w2
```

so your forward step looks like:

```
x
│
▼
× w1
│
▼
+ b1
│
▼
× w2
│
▼
Output
```

But there's a problem: backpropagation. Now you're forced to find:

```
dy/dw1

dy/db1

dy/dw2
```

So what can you do? With a small number of parameters, you'd be fine. But when you're building an AI with, say, 5+ billion parameters? Doing all of this by hand is basically unthinkable. That's why PyTorch remembers every step for us.

PyTorch is like a diligent student — you keep doing the math, and it writes down every step you take.

Imagine baking a cake:

```
Eggs
↓
Add Flour
↓
Mix
↓
Bake
↓
Cake
```

PyTorch writes it down as:

```
Step 1
↓
Step 2
↓
Step 3
↓
Step 4
```

So every time you do:

```
y = x * 3
```

PyTorch remembers: `multiply`.

Every time you do:

```
z = y + 5
```

PyTorch remembers: `add`.

So it notes down:

```
multiply
↓
add
↓
...
```

and so on.

But why do we care about any of this? Because once we reach the loss, we need to ask "how did we get here?" — and PyTorch already knows, since it doesn't have to struggle through manual backpropagation the way we would. It simply walks its own steps backward and applies the chain rule. That's exactly what makes it so useful.

Here's what that ability looks like in Python:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 5
loss = z ** 2
```

`requires_grad` tells PyTorch: "Hey, track this computation — I'll need gradients from it later."

Before frameworks like PyTorch became popular, researchers often had to derive and implement gradients manually for every new model. That was slow and error-prone.

With PyTorch, you can just do:

```python
loss.backward()
```

This way, we only describe the forward step, and PyTorch handles the rest of the backpropagation.

But there's a catch — what if we want to pause tracking for a bit? Maybe we just need to do some quick element-wise math that doesn't need gradients. Letting PyTorch track absolutely everything can lead to a memory crash — imagine tracking every single step while building an 18-billion-parameter GPT. Track everything? Absolutely not.

That's why we use `torch.no_grad()`, like this:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 5 # Let's say we want to shrink z a bit:
with torch.no_grad():
	z = z - 4
loss = z ** 2
```

Now, nothing inside `with torch.no_grad():` gets recorded.

Now let's continue with `.backward()`!

## `.backward()` and `.grad`

I already talked about `.backward()`, but I haven't shown you how to actually use it — let's fix that.

Imagine Airi is building a 2-layer model, but she doesn't want to derive the gradients by hand — her hand hurts. So what does she do? She reaches for PyTorch's special ability: `.backward()`.

```
 Input

   x
   │
   ▼
┌─────────┐
│ Hidden  │
└─────────┘
   │
   ▼
┌─────────┐
│ Output  │
└─────────┘
   │
   ▼
 Prediction
```

We use this:

```python
z1 = x * w1 + b1
a1 = torch.sigmoid(z1) # I'll do what scientists did back in 2010, spamming sigmoid.

z2 = a1 * w2 + b2
a2 = torch.sigmoid(z2)

loss = (a2 - target) ** 2  # MSE, practically (error**2)
```

Now Airi has to derive all of this — but she "accidentally" caught her hand in a door, so she can't. That's why she'll simply use `.backward()`.

The graph remembers:

```
x

↓

× w1

↓

+ b1

↓

Sigmoid

↓

× w2

↓

+ b2

↓

Sigmoid

↓

Loss
```

So now we just write:

```python
loss.backward()
```

The process plays out exactly like this:

```
Loss

↓

depends on   < --- The activation, in our case it was sigmoid

↓

a2

↓

depends on

↓

z2

↓

depends on

↓

w2
```

and eventually it computes:

```
dLoss/dw2
```

and the same for:

```
dLoss/db2
dLoss/dw1
dLoss/db1
```

Together, these are the gradients.

Now we can do:

```python
print(w1.grad)
print(b1.grad)

print(w2.grad)
print(b2.grad)
```

This hands us the exact value we need for every weight and bias — no manual backpropagation required.

So the whole cycle looks like:

```
          Forward Pass
────────────────────────────────

Input

↓

Layer 1

↓

Activation

↓

Layer 2

↓

Prediction

↓

Loss


          Backward Pass
────────────────────────────────

Loss

↑

Compute gradients

↑

Store gradients in parameters


          Optimizer
────────────────────────────────

Read gradients

↓ 

Update weights

↓

Better model
```

`.grad` literally stores the gradient of a weight or bias. For example, if we do:

```python
w1 = torch.tensor(0.5, requires_grad=True)
```

right now we have:

```
w1
├── Value = 0.5
└── grad  = None
```

But after `loss.backward()`, if we write:

```python
print(w1.grad)
```

we get:

```
w1
├── Value = 0.5
└── grad  = -0.18
```

As we've noticed, PyTorch doesn't change the weights automatically — it just tells us: "if you want to reduce the loss, moving this weight in this direction would help."

All of the topics ahead are pretty pivotal. Let's start with `Normalize`.

## Normalize

We'll often have small-scale features (practice hours, kilograms lost last month, how many times you eat per day, and so on), and sometimes really large-scale features (house price, car price, annual salary...).

Suppose we have:

```

Hours fishing      = 5
Water temperature  = 23
Annual salary      = $112,000,000
```

We can immediately notice something: one feature is millions of times larger than the others.

Now imagine doing:

```math
y=w_1x_1+w_2x_2+w_3x_3
```

Here's the problem — say all the weights start at `0.1`:

```
0.1 × 5          =          0.5
0.1 × 23         =          2.3
0.1 × 112,000,000 = 11,200,000

Total ≈ 11,200,002.8
```

As we can see, the fishing hours and water temperature basically vanish — out of `11,200,000`, they're only worth `2.8` combined. The model will conclude these features are nearly useless and focus almost entirely on the annual salary.

During training, gradient descent is essentially asking: "which weight should I change?" Since the salary feature produces much larger values, its gradients tend to be much larger too:

```
Hours gradient       ▏
Temperature gradient ▎
Salary gradient      ███████████████████████████
```

So the optimizer pours nearly all of its effort into adjusting the salary weight, and barely touches the hours or temperature weights.

That's why we don't feed in raw values — we normalize them so they land in a similar range. For example:

```
Hours       = 0.42
Temperature = 0.58
Salary      = 0.63
```

Now the optimizer can actually learn from **all** of them, instead of one feature drowning out the rest.

Let's see this in code, since a concept without code isn't really understood yet.

```python
import torch

# [Hours Fishing, Water Temperature, Annual Salary]
X = torch.tensor([
    [2.0, 18.0, 45_000.0],
    [4.0, 20.0, 60_000.0],
    [6.0, 22.0, 85_000.0],
    [8.0, 24.0,120_000.0],
], dtype=torch.float32)
```

We can see the min and max of each column:

```
Hours       :      2 → 8
Temperature :     18 → 24
Salary      : 45,000 → 120,000
```

Salary clearly dominates. So we'll use this formula:

```math
x_{new} = \frac{x - min(x)}{max(x) - min(x)}
```

This squashes every feature into the range 0 to 1. In Python:

```python
min_x = X.min(dim=0).values
max_x = X.max(dim=0).values

X_new = (X - min_x)/(max_x - min_x)
```

That's the baby-steps version. There's another, much more commonly used formula:

```math
x_{new} = \frac{x - \mu}{\sigma}
```

We write it as:

```python
X_mean = X.mean(dim=0)
std = X.std(dim=0)

X_standard = (X - X_mean)/std
```

The output looks like:

```
tensor([
[-1.16, -1.16, -0.99],
[-0.39, -0.39, -0.53],
[ 0.39,  0.39,  0.23],
[ 1.16,  1.16,  1.30]
])
```

Now, instead of values ranging from 0 to 1, we have values centered around 0.

Now let's move on to a harder lesson — we're going to write a custom `autograd.Function`.

## `autograd.Function`

Before we start, I want to explain what autograd actually is, since we can't move forward without it. Autograd is PyTorch's automatic differentiation engine — and I'm sure that sentence alone explained nothing, so let me give you an example.

Building a small neural network in PyTorch goes through these steps:

```
You
 │
 │ create tensors
 ▼
PyTorch
 │
 │ records every mathematical operation
 ▼
Autograd
 │
 │ builds a computation graph
 ▼
Loss
 │
 │
backward()
 │
 ▼
Autograd computes gradients
 │
 ▼
.grad fields are filled
```

Now imagine we have:

```python
x = torch.tensor(2.0)

w = torch.tensor(3.0, requires_grad=True)

y = x * w
z = y + 4
loss = z ** 2
```

When you run this, Autograd is quietly watching every step, recording:

- which operation happened
- which tensors were involved
- how to differentiate that operation

Then comes the backpropagation part:

```python
loss.backward()
```

Now PyTorch tells Autograd: "please compute all the derivatives." Autograd walks back through the graph and applies the chain rule at every single operation.

So we can't say Autograd = backpropagation. Backpropagation is the algorithm; Autograd is the system that carries it out automatically. Autograd quietly watches, then computes everything on its own.

But why would we ever need a *custom* Autograd function? Because, sadly, Autograd doesn't automatically know how to handle everything. Say we call some function in the middle of our forward pass. Autograd adds it to the graph — but when it reaches that step, what's it supposed to do? Can it differentiate:

```python
cute_function_that_Airi_made(x):
	# Airi called a Julia library
```

It has no idea what that even is. Even after adding it to the graph, Autograd still doesn't know how to differentiate it — so it fails.

But if we write this instead:

```python
class SquareOfTwo(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        local_derivative = 2 * x
        grad_input = grad_output * local_derivative
        return grad_input
```

I'm sure that explained absolutely nothing on its own — so let's break down every piece:

- `class SquareOfTwo(torch.autograd.Function)` — this is just a normal class, but it tells PyTorch: "I'm defining a new differentiable operation." We're declaring a brand-new mathematical operation it doesn't already know about. Just like PyTorch already has `addition`, `subtraction`, `multiplication`, `division`, and so on, we're defining a new one called `SquareOfTwo`.
- `@staticmethod` — why do we need this? Because we don't need `self`. As we already know, when we build a normal class:

```python
class Dog():
	def bark(self):
		print("Woof, woof")  
```

we understand that when we do:

```python
dog = Dog()

dog.bark()

"""
Internally, this happens:
Dog.bark(dog)
""" 
```

But we don't need that here, because we never actually instantiate an object. We'll never write `cute = SquareOfTwo()` — we'll write `SquareOfTwo.apply(...)` instead, which works differently.

- `ctx` — imagine the forward step runs first, and the backward step doesn't run until much later. The backward step needs information from the forward pass, so we use `ctx` — short for **context** — as a shared backpack. We stash things in it during `forward`, and later hand that same backpack to `backward()`, which unpacks it to understand what happened. Meanwhile, `x` is just our input.
- `ctx.save_for_backward(x)` — say `x = 3`, so we compute `3 ** 2 = 9`. But `backward` needs to know `x` too, since it'll need it during backpropagation, so it wants to be ready ahead of time. That's why we save it — we tuck it into the backpack, so `backward` can open it later and retrieve `x`.
- `def backward(ctx, grad_output)` — we hand `ctx` (the backpack) to `backward`, but what's `grad_output`? It's the incoming gradient of the loss with respect to this function's output — something like:

  ```math
  \frac{\partial L}{\partial (\text{our function's output})}
  ```

  In plain terms: "if I nudge our function's output slightly, how much does the loss go up or down?" That's the whole story behind `grad_output`.
- `x, = ctx.saved_tensors` — this has `backward` open the backpack and look inside, finding our `x`. Why the odd `x,` syntax? Because Python always stores these as a tuple, even with a single value. If we'd saved multiple things earlier, like `ctx.save_for_backward(a, b, c)`, we'd unpack them as `a, b, c = ctx.saved_tensors`.
- `local_derivative = 2 * x` — that's just our derivative (aww, they thought they could escape the math).
- `grad_input` — this is the chain rule in action.
- `return grad_input` — we return the result, but be careful: whatever we return here has to line up one-to-one with the inputs of `forward`.

We won't go much deeper into this right now, since there's not much use for it yet — we'll dig further into it in months 4-5.

For now, let's cover two harder, but genuinely useful, concepts: optimizers and initializations.

But before that, let's learn `nn` — that's the smarter move first.

# Chapter 2

## `nn.Module`

Before explaining what `nn.Module` is, let's get there the fast way.

Say Airi wants to build a neural network — easy, right? Sure. She could build it from scratch:

```python
import torch

x = torch.tensor([2.0, 3.0])

W = torch.tensor([[1.5, -0.5]])
b = torch.tensor([0.2])

y = W @ x + b

print(y)
```

Easy enough. But now imagine she wants a network with many layers — say, 20+. She'd be stuck writing:

```python
W1 = ...
W2 = ...
W3 = ...
W4 = ...
...
W22 = ...
```

and probably a matching pile of biases too. That would be hellish.

That's exactly why `nn.Module` exists — think of it as a toolbox built for neural networks:

```
torch.nn
├── Linear
├── Conv2d
├── ReLU
├── Dropout
├── Embedding
├── LSTM
├── Transformer
└── ...
```

and plenty more. We use it to build as many weights as we want, without writing them all by hand. We import it as:

```python
import torch.nn as nn
```

The most common piece is `nn.Linear` — literally our linear equation:

```math
y = Wx + b
```

Forget matrices and calculus for a second. Imagine you're a doctor looking at **3 patient stats**:

1. Temperature (39.0)
2. Coughs per minute (15.0)
3. Days sick (4.0)

and you want to compute 2 scores: flu risk and cold risk. We want to feed in three features and get back just two numbers. How does that happen?

A single patient walks into the room, and there are 32 doctors waiting. The first doctor takes the patient's numbers, multiplies them by whatever weights they think are right, and adds their own bias. The second doctor takes the exact same numbers, applies *their own* set of weights, and produces a second number — and so on, through doctor number 32.

By the end of layer 1, you have 32 features. If we want more layers, we do:

```python
import torch.nn as nn

# Layer 1: Takes 4 patient features -> Outputs 32 hidden features
layer1 = nn.Linear(in_features=4, out_features=32)

# Layer 2: MUST take 32 features in, since layer 1 output 32 columns and the shapes have to line up -> Outputs 16 features
layer2 = nn.Linear(in_features=32, out_features=16)

# Layer 3 (Output): Takes 16 features in -> Outputs 1 final decision score
layer3 = nn.Linear(in_features=16, out_features=1)
```

Without this, we'd have had to write:

```python
W1 = torch.randn((4, 32), requires_grad=True)
b1 = torch.zeros(32, requires_grad=True) 
W2 = torch.randn((32, 16), requires_grad=True)
b2 = torch.zeros(16, requires_grad=True)
W3 = torch.randn((16, 1 ), requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)
```

and even that's the simplified version.

`nn.Linear` automatically creates its own weights and biases. Where does it get the values from? It uses **He (Kaiming) initialization** — our next topic.

For now, let's see it in action. Airi went fishing, and she's noticed that some factors seem to affect her catch:

```python
import torch
import torch.nn as nn

# Features:
# [Hours Fishing, Water Temperature (°C), Worms Used]
X = torch.tensor(
    [
        [1.0, 18.0, 2.0],  # Morning trip 1
        [2.0, 20.0, 3.0],  # Morning trip 2
        [3.0, 21.0, 5.0],  # Morning trip 3
        [4.0, 22.0, 5.0],  # Morning trip 4
        [5.0, 23.0, 7.0],  # Morning trip 5
        [6.0, 24.0, 8.0],  # Morning trip 6
        [7.0, 25.0, 9.0],  # Morning trip 7
        [8.0, 26.0, 10.0],  # Morning trip 8
    ],
    dtype=torch.float32,
)

# Target: Total fish caught
y = torch.tensor(
    [[2.0], [4.0], [6.0], [8.0], [11.0], [13.0], [15.0], [18.0]], dtype=torch.float32
)

x_mean = X.mean(dim=0)
std = X.std(dim=0)
x_norm = (X - x_mean) / std

# Now we will use nn
class FishingPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(in_features = 3, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=8)
        self.out = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x


model = FishingPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    prediction = model(x_norm)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Loop {epoch}, current loss: {loss}")

# prints the weight of the first layer
print("FC1 Weights:\n", model.fc1.weight)
print("FC1 Bias:\n", model.fc1.bias)

# prints the weight of the output layer
print("Out Weights:\n", model.out.weight)
print("Out Bias:\n", model.out.bias)

"""
Output:

Loop 0, current loss: 119.17611694335938
Loop 100, current loss: 0.4675063192844391
Loop 200, current loss: 0.03997226059436798
Loop 300, current loss: 0.037955187261104584
Loop 400, current loss: 0.03754353150725365
FC1 Weights:
 Parameter containing:
tensor([[-0.4034, -0.6806,  0.1853],
        [ 0.2933,  0.0851,  0.6993],
        [ 0.6669,  0.6234, -0.2840],
        [ 0.0354, -0.7087,  0.0216],
        [ 0.3548, -0.2439, -0.2731],
        [-0.5145,  0.4749,  0.2096],
        [-0.1646,  0.3272, -0.2300],
        [ 0.0149,  0.3767,  0.6364],
        [-0.6687, -0.1604, -0.1056],
        [ 0.0613,  0.7329, -0.3069],
        [-0.3375,  0.4067, -0.2905],
        [ 0.1195,  0.6321,  0.6457],
        [ 0.4223, -0.1450,  0.2715],
        [ 0.0429, -0.1012, -0.1148],
        [ 0.0687,  0.3507,  0.2892],
        [ 0.4159,  0.8836,  0.8269]], requires_grad=True)
FC1 Bias:
 Parameter containing:
tensor([ 0.0087,  0.2606,  0.5894, -0.0782,  0.9613,  0.6187, -0.2514,  1.2168,
         0.1200,  1.0267,  1.0186,  1.1820,  0.9321,  0.9260,  0.8966,  0.1792],
       requires_grad=True)
Out Weights:
 Parameter containing:
tensor([[ 0.5070, -0.1406,  0.0392, -0.0577,  0.6202, -0.5363,  0.1732,  0.6354]],
       requires_grad=True)
Out Bias:
 Parameter containing:
tensor([0.3290], requires_grad=True
"""
```

That's how we use `nn.Module`. It becomes the standard for pretty much all of our code from here on.

Now let's start with initializations.

## Initializations

Setting the initial weights is easy, right? Just set them to 0, or use `torch.randn()`. Done. Thanks for reading, bye.

Not so fast. This is one of the most important concepts in all of AI and machine learning — let me show you the two traps almost everyone falls into at first (we were all a little naive here).

**1. Setting all the weights to 0**

This one's tempting, but way too dangerous to even try. Imagine a layer with 1,000 neurons, all initialized with the exact same weights (all zeros). Since every neuron gets the same input and has the same weights, they all produce the exact same output. During backpropagation, they also receive identical gradients, so they get updated in exactly the same way. Those 1,000 neurons end up as perfect doppelgangers of each other. We don't want that — this is called the **symmetry** problem.

For example:

```
Neuron A

w = [0,0,0]

Neuron B

w = [0,0,0]
```

Say the input is:

```python
x = [2, 5, 1]
```

and we compute:

```python
z = x @ w
```

Output for both neurons? 0.

Now say backpropagation runs. Both weights update — and both land on 0.7 (a random illustrative number). What does that mean? The weights are still perfectly identical. Next iteration? Still identical. Next? Still identical. So really, we've built one neuron that just gets copied 1,000 times.

**2. Random initialization**

This is a far more standard choice in modern deep learning. But, as expected, it has its own problem. If the weights start too large, each layer amplifies the previous one, and we get **exploding gradients** (numbers that blow up to `inf`) or exploding activations. If the weights start too small, each layer gradually shrinks the signal, and we end up with **vanishing gradients** (values collapsing toward zero) — also not great.

For example:

```
Neuron 1

0.13

Neuron 2

-0.44

Neuron 3

0.08
```

These are all different! Problem solved, right? Not so fast — imagine we have 100 layers:

```
Input

5

↓

Weight

8

↓

40
```

```
40

×

7

↓

280
```

and so on, until you slam straight into the wall of infinity. Same problem, different flavor.

That's why we have two well-known solutions: Glorot initialization and Kaiming initialization.

### Glorot (Xavier) Initialization

This one was introduced in 2010 by Xavier Glorot and Yoshua Bengio.

Instead of just saying "pick a random number," they asked: "how large should these random numbers be so the signal neither explodes nor vanishes?"

The answer depends on the number of inputs and outputs:

```
Number of inputs

(fan_in)

Number of outputs

(fan_out)
```

The more connections a neuron has, the smaller each initial weight generally needs to be:

```math
w \sim U\left(-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}\right)
```

For now, just take away the *idea* behind this — we'll almost never implement it by hand, except with genuinely deep networks. We'll get to that later.

### He (Kaiming) Initialization

Later on, Kaiming noticed that ReLU throws away every negative number, zeroing them out completely. That made Glorot initialization suboptimal, since Glorot was only designed around keeping values centered near zero — a concept ReLU completely disregards.

So Kaiming built a new initialization specifically for ReLU, Leaky ReLU, GELU, and similar activations:

```math
w \sim \mathcal{N}\left(0, \frac{2}{fan_{in}}\right)
```

Before we just memorize the formula, let's understand the logic. Imagine 1,000 people spread across 10 rooms, but some of them are "sad" (negative values), and an inspector at each doorway won't let sad people through (negative numbers become 0). By the time you reach room 10, roughly half the people have been filtered out along the way. But what if we told the surviving half to be twice as "happy"? We'd end up with the same overall energy as the original 1,000 — just as 500 doubly-happy people instead. That doubling is exactly what the factor of 2 in Kaiming's formula compensates for.

We'll learn how to apply this by hand once the time comes — for now, that time isn't quite here yet.

## Optimization

This is another massively important step, because without it, everything falls apart no matter how hard we try elsewhere. Imagine having 20+ weights and biases — imagine how absurd it'd be to write, by hand, at the end of every step:

```python
w1 -= dw1 * lr
b1 -= db1 * lr
w2 -= db2 * lr
...
dw26 -= dw26 * lr 
```

That's why we have `torch.optim` to save us.

It's basically PyTorch's automatic parameter handler. You simply write:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

`model.parameters()` hands PyTorch references to every tensor saved inside our class — `fc1`, `fc2`, `fc3`, and so on.

For example, we'd do:

```python
for epoch in range(500):
	prediction = model(X_norm)
	loss = criterion(prediction, y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

Let's break down anything unclear here:

- `loss = criterion(prediction, y)` — we'll cover why we use this in the next chapter.
- `optimizer.zero_grad()` — by default, PyTorch *accumulates* gradients instead of overwriting them. Without resetting, the second epoch's gradients would stack on top of the first epoch's, the third on top of the second, and so on forever. `optimizer.zero_grad()` goes through every `.grad` field and resets it to `None` (effectively 0).
- `optimizer.step()` — now PyTorch walks through every parameter tensor and updates it.

But as we noticed, we started with `torch.optim.Adam()` specifically — why? There are several optimizer types, so let's cover the most famous ones.

**1. SGD (Stochastic Gradient Descent)**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

This is the "dumb" classic approach — it applies the exact same learning rate to every parameter:

```math
w_{new} = w_{old} - grad \times lr
```

So if it runs into vanishing gradients, the learning rate effectively crawls to a halt.

**2. Adam (Adaptive Moment Estimation)**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

This is the standard choice for most modern deep learning prototypes. Instead of blindly applying just the learning rate, Adam tracks two extra statistics for every single weight:

- **Momentum (first moment)** — a moving average of past gradients. If a weight has been moving in the same direction for a while, Adam speeds it up in that direction, like a heavy ball rolling downhill.
- **Adaptive scaling (second moment)** — a moving average of squared gradients. If a parameter's gradient updates are huge, Adam lowers its effective learning rate to keep things stable; if the updates are tiny, Adam raises it. Either way, things stay stable.

**3. AdamW (Adaptive Moment Estimation with Weight Decay)**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Before explaining why we'd use this, let's cover what weight decay actually is.

Imagine you're cleaning your room (let's say the room *is* our neural network), and you own 500 things (weights) — but only 10 of them are actually useful, and the other 490 aren't. So each day, you throw away a tiny percentage of the clutter. If something isn't useful enough to earn its keep, it slowly shrinks away. That's exactly what weight decay does: "if this weight isn't pulling its weight, slowly make it smaller."

In real life, it'd look something like: Airi studies 7 hours a day, while Hinako studies just 30 minutes. Airi's mom comes home and, not caring that Hinako doesn't even live there, forces *both* of them to clean the room.

With plain Adam, the optimizer would essentially say: "Since Airi studies so much, I'll make Hinako clean 90% of the room, and Airi just 10%." That's a problem — studying $`\neq`$ cleaning. They're completely unrelated, but Adam ties them together anyway.

AdamW handles this differently. First, it does the same thing Adam does — move the weights based on gradients. Then, separately, it applies a flat "tax": it shaves off a fixed 1% from every weight, no exceptions, since PyTorch groups weights and biases together and taxes both equally.

AdamW shows up literally everywhere: GNNs, Transformers, MLPs, general deep learning, you name it. It's simply the better version of Adam. There's a subtlety we'll cover later, though: why we generally shouldn't decay biases, and should never touch normalization parameters at all.

## Criterion

Maybe you've seen code like:

```python
loss = nn.MSELoss()
```

and thought: why bother with this when I can just flex my own skills and write:

```python
loss = (error ** 2).mean()
```

Wouldn't the output be the same? Maybe — but PyTorch's built-in loss functions are still much better than rolling your own formula.

Think of it like making pizza dough. You *can* make it yourself:

```
Flour
Water
Yeast
Salt
Mix
Wait
Knead
Wait
Bake
```

or you could just buy ready-made dough from a professional, and get the same result — except the version you buy was made by someone who's done it a thousand times.

Why should we prefer PyTorch's built-in loss functions?

1. **Fewer bugs** — imagine accidentally writing `criterion = (error**2)` and forgetting the `.mean()`.
2. **Easier to read** — someone reading `criterion = MSELoss()` understands it faster than someone parsing a raw formula.
3. **Edge cases handled** — without us having to think through every one manually.
4. **Optimized** — with 100 million parameters, instead of spawning a pile of temporary tensors and hammering memory, PyTorch often fuses operations internally, giving us extra speed and lower memory use.
5. **Autograd-friendly** — even a solid manual version is riskier than the PyTorch version, which has been battle-tested by millions of researchers.

Let's see the difference in code. Airi's been feeling pretty rough lately, so she goes to get checked out by a bunch of different doctors.

```python
import torch
# ==== INPUTS ====

# Preparing the 'send' button for the tensors
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Features: [Temperature (°C), Coughs/min, Fatigue Level (1-10), Days Sick]
# She goes to get a check up by different doctors everyday, since she feels bad.
X = torch.tensor(
    [
        [36.5, 1.0, 1.0, 1.0],  # Doctor 1
        [38.8, 18.0, 8.0, 3.0],  # Doctor 2
        [37.0, 3.0, 2.0, 2.0],  # Doctor 3
        [39.5, 25.0, 9.0, 5.0],  # Doctor 4
        [36.6, 0.0, 1.0, 1.0],  # Doctor 5
        [38.2, 12.0, 7.0, 4.0],  # Doctor 6
        [37.3, 5.0, 4.0, 3.0],  # Doctor 7
        [38.9, 20.0, 8.0, 6.0],  # Doctor 8
        [36.8, 2.0, 2.0, 2.0],  # Doctor 9
        [39.1, 15.0, 9.0, 4.0],  # Doctor 10
    ],
    dtype=torch.float32, 
).to(device)

Y = torch.tensor(
    [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
    dtype=torch.float32,
).to(device)

W1 = (torch.randn(4, 32).to(device) * (2.0/4.0) ** 0.5).requires_grad_() # We use He init
b1 = torch.zeros(32).to(device).requires_grad_()
W2 = (torch.randn(32, 16,).to(device) * (2.0/32.0) ** 0.5).requires_grad_()
b2 = torch.zeros(16).to(device).requires_grad_()
W3 = (torch.randn(16, 1 ).to(device) * (2.0/4.0) ** 0.5).requires_grad_()
b3 = torch.zeros(1).to(device).requires_grad_()

# IMPORTANT VARIABLES
lr = 0.001
epsilon = 1e-15
optimizer = torch.optim.Adam([W1, b1, W2, b2, W3, b3], lr=lr)


# ==== TRAINING STEP ====
for epoch in range(1000):
    optimizer.zero_grad()
    Z1 = X @ W1 + b1
    A1 = Z1.relu()
    Z2 = A1 @ W2 + b2
    A2 = Z2.relu()
    Z3 = A2 @ W3 + b3
    A3 = Z3.sigmoid()

    # LOSS FUNCTION (BCE)
    A3_clipped = torch.clip(A3, epsilon, 1 - epsilon)
    loss = -torch.mean(Y * torch.log(A3_clipped) + (1 - Y) * torch.log(1 - A3_clipped))

    # BACKPROPAGATION STEP 
    loss.backward()
    optimizer.step()

    if epoch %500 == 0:
        print(f"Loops {epoch}, the loss is: {loss: 5f}")


print(f"The loss of the model is of: {loss}")
weights = [W1, W2, W3]

for num, weight in enumerate(weights, start=1):
    print(f"W{num}: \n{weight.grad}\n")


"""
Output:

Loops 0, the loss is:  1.010335
Loops 500, the loss is:   nan
The loss of the model is of: nan
W1: 
tensor([[nan, nan, 0., nan, nan, 0., nan, nan, nan, 0., 0., 0., nan, nan, nan nan... 
...
...
...
"""
```

I did this specifically to show the impact of normalization.

```python
import torch
# ==== INPUTS ====

# Preparing the 'send' button for the tensors
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Features: [Temperature (°C), Coughs/min, Fatigue Level (1-10), Days Sick]
# She goes to get a check up by different doctors everyday, since she feels bad.
X = torch.tensor(
    [
        [36.5, 1.0, 1.0, 1.0],  # Doctor 1
        [38.8, 18.0, 8.0, 3.0],  # Doctor 2
        [37.0, 3.0, 2.0, 2.0],  # Doctor 3
        [39.5, 25.0, 9.0, 5.0],  # Doctor 4
        [36.6, 0.0, 1.0, 1.0],  # Doctor 5
        [38.2, 12.0, 7.0, 4.0],  # Doctor 6
        [37.3, 5.0, 4.0, 3.0],  # Doctor 7
        [38.9, 20.0, 8.0, 6.0],  # Doctor 8
        [36.8, 2.0, 2.0, 2.0],  # Doctor 9
        [39.1, 15.0, 9.0, 4.0],  # Doctor 10
    ],
    dtype=torch.float32, 
).to(device)

X_mean = X.mean(dim=0)
std = X.std(dim=0)
X_norm = ((X - X_mean)/std).to(device)

Y = torch.tensor(
    [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
    dtype=torch.float32,
).to(device)

W1 = (torch.randn(4, 32).to(device) * (2.0/4.0) ** 0.5).requires_grad_() # We use He init
b1 = torch.zeros(32).to(device).requires_grad_()
W2 = (torch.randn(32, 16,).to(device) * (2.0/32.0) ** 0.5).requires_grad_()
b2 = torch.zeros(16).to(device).requires_grad_()
W3 = (torch.randn(16, 1 ).to(device) * (2.0/4.0) ** 0.5).requires_grad_()
b3 = torch.zeros(1).to(device).requires_grad_()

# IMPORTANT VARIABLES
lr = 0.001
epsilon = 1e-15
optimizer = torch.optim.Adam([W1, b1, W2, b2, W3, b3], lr=lr)


# ==== TRAINING STEP ====
for epoch in range(1000):
    optimizer.zero_grad()
    Z1 = X_norm @ W1 + b1
    A1 = Z1.relu()
    Z2 = A1 @ W2 + b2
    A2 = Z2.relu()
    Z3 = A2 @ W3 + b3
    A3 = Z3.sigmoid()

    # LOSS FUNCTION (BCE)
    A3_clipped = torch.clip(A3, epsilon, 1 - epsilon)
    loss = -torch.mean(Y * torch.log(A3_clipped) + (1 - Y) * torch.log(1 - A3_clipped))

    # BACKPROPAGATION STEP 
    loss.backward()
    optimizer.step()

    if epoch %500 == 0:
        print(f"Loops {epoch}, the loss is: {loss: 5f}")


print(f"The loss of the model is of: {loss}")
weights = [W1, W2, W3]

for num, weight in enumerate(weights, start=1):
    print(f"W{num}: \n{weight.grad}\n")
    
"""
Output:

Loops 0, the loss is:  0.976106
Loops 500, the loss is:  0.001610
The loss of the model is of: 0.0003151005948893726
W1: 
tensor([[-8.7316e-05, -7.1696e-05,  4.2447e-05,  4.2198e-05,  3.7277e-06,
         -5.8452e-05, -1.0668e-05, -5.1022e-05, -1.1400e-05, -1.5064e-07,
          9.3346e-05,  3.7314e-05,  1.7785e-06,  3.3709e-05,  8.5372e-06,
         -6.7757e-05,  2.0548e-06,  3.9300e-05, -4.6315e-05,  4.3105e-05,
         .....
         .....
"""
```

That's the gigantic difference normalization makes. What if we rewrote this using the full `nn.Module` version instead?

Let's use `nn`, and everything we've learned so far!

```python
import torch
import torch.nn as nn
from torchinfo import summary

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

X = torch.tensor(
    [
        [36.5, 1.0, 1.0, 1.0],  # Doctor 1
        [38.8, 18.0, 8.0, 3.0],  # Doctor 2
        [37.0, 3.0, 2.0, 2.0],  # Doctor 3
        [39.5, 25.0, 9.0, 5.0],  # Doctor 4
        [36.6, 0.0, 1.0, 1.0],  # Doctor 5
        [38.2, 12.0, 7.0, 4.0],  # Doctor 6
        [37.3, 5.0, 4.0, 3.0],  # Doctor 7
        [38.9, 20.0, 8.0, 6.0],  # Doctor 8
        [36.8, 2.0, 2.0, 2.0],  # Doctor 9
        [39.1, 15.0, 9.0, 4.0],  # Doctor 10
    ],
    dtype=torch.float32,
)

Y = torch.tensor(
    [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
    dtype=torch.float32,
)

x_mean = X.mean(dim=0)
std = X.std(dim=0)
X_norm = ((X - x_mean)/std)

X_norm = X_norm.to(device)
Y = Y.to(device)

class DoctorCheckUp(nn.Module):
    def __init__(self, feature_in):
        super().__init__()

        self.fc1 = nn.Linear(feature_in, out_features=32)
        self.fc2 = nn.Linear(32, out_features=16)
        self.fc3 = nn.Linear(16, out_features=8)
        self.out = nn.Linear(8, out_features=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)

        return torch.sigmoid(x)

model = DoctorCheckUp(feature_in=4).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for epoch in range(2500):
    prediction = model(X_norm)
    loss = criterion(prediction, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(f"Loop {epoch}, loss: {loss: .6f}")

print(f"Current loss: {loss}")
print("")
summary(model, input_size=X_norm.shape)

"""
Output:

Loop 0, loss:  0.718449
Loop 250, loss:  0.227422
Loop 500, loss:  0.023005
Loop 750, loss:  0.006907
Loop 1000, loss:  0.003201
Loop 1250, loss:  0.001808
Loop 1500, loss:  0.001141
Loop 1750, loss:  0.000772
Loop 2000, loss:  0.000549
Loop 2250, loss:  0.000405
Current loss: 0.00030676223104819655

=================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
=================================================================================
DoctorCheckUp                            [10, 1]                   --
├─Linear: 1-1                            [10, 32]                  160
├─Linear: 1-2                            [10, 16]                  528
├─Linear: 1-3                            [10, 8]                   136
├─Linear: 1-4                            [10, 1]                   9
=================================================================================
Total params: 833
Trainable params: 833
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.01
=================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.01
=================================================================================
"""
```

Let me explain `summary` — it's basically the medical report for our model. It tells us:

- The name
- How many layers it has
- How big those layers are
- How many neurons it contains
- How many values it has to learn
- How much memory it uses

1. `DoctorCheckUp` — the name of our class.
2. `[10, 1]` — the final output shape of our model.
3. `├─Linear: 1-1   [10,32]   160`:
   - `├─Linear: 1-1` — tells us this is a linear layer, and it's the first one (hence `1-1`).
   - `[10, 32]` — 10 patients, each mapped to 32 neurons.
   - `Param: 160` — the model has 160 numbers to learn here. Where from? Our first layer is `[4, 32]`, so using the linear formula (`X * W + b`): `4 * 32 + 32 = 160`.
4. `Total params: 833` — the total number of parameters gradient descent updates.
5. `Non-trainable params: 0` — since we never froze anything (freezing means setting `.requires_grad = False`).

That's the output — and it's genuinely useful.

So now let's learn something great: mixed precision training.

## Mixed Precision Training (`torch.amp`)

What is this, and why was it invented?

Imagine Hinako bought an RTX 5090. (I considered writing "imagine *you* bought one," but then remembered you're probably as broke as me, so not even in our dreams.)

It costs a fortune and packs a ton of CUDA cores — yet your model might only be using half of what the GPU is actually capable of. Why? Because you're feeding it the wrong type of numbers.

Everything in deep learning is just numbers. Our neural network is millions (or billions) of values like:

```python
Weight = 0.2356
Bias   = -1.734
Input  = 7.25
Output = 0.823
Gradient = -0.00412
```

But our machine doesn't "understand" numbers the way we do — it stores bits, like:

```
0000011111000111000111000000111101101
```

(That's random filler, don't try to read meaning into it — I'm not a compiler.)

And generally: more bits means better precision, but more memory usage. The most common types:

| Type | Bits | Bytes | Precision |
| ---- | ---: | ----: | --------- |
| FP64 |   64 |     8 | Very high |
| FP32 |   32 |     4 | High      |
| BF16 |   16 |     2 | Medium    |
| FP16 |   16 |     2 | Medium    |

Think of your GPU memory as a warehouse, and each floating-point number as a box. Store them as FP32, and the boxes are relatively big. Store them as FP16, and the boxes shrink.

Say your model has 100 million numbers, and you use FP32:

```
100,000,000 * 4 (bytes) = 400,000,000 bytes -> 400 MB
```

Now say you use FP16 instead:

```
100,000,000 * 2 (bytes) = 200,000,000 bytes -> 200 MB
```

You roughly cut your memory usage in half. And FP16 is often faster than FP32 too, because GPUs like the RTX 5090 have something called **Tensor cores** — highly specialized workers.

- CUDA cores → general-purpose workers.
- Tensor cores → specialists, built specifically to multiply matrices extremely fast — exactly what neural networks do constantly.

Tensor cores are primarily designed around BF16 and FP16, so sticking with FP32 means you're leaving performance on the table.

So why doesn't everybody just always use FP16? Because of the precision tradeoff shown in the table above:

FP32: `0.123456789`

FP16: `0.12345`

The neural network is usually pretty tolerant of this small amount of rounding. But there's still a problem: what if a gradient is as small as `0.00000018`, or even smaller, like `0.00000000002`? In FP16, that could round straight down to 0 — a problem called **underflow**.

But remember the name of this whole topic: *mixed* precision training. Instead of "everything must be FP16," PyTorch's approach is "let's only use FP16 where it's safe." That's where `autocast` comes in.

A normal (FP32) forward pass:

```python
x = self.linear1(x)
x = self.relu(x)
x = self.linear2(x)
loss = criterion(x, y)
```

Using `autocast` instead:

```python
with torch.autocast(device_type="cuda"):
    prediction = model(x)
    loss = criterion(prediction, y)
```

Here, we're telling PyTorch: "you decide." It automatically checks, operation by operation, whether it's safe to use FP16 or not.

Now let's bring in the `scaler`:

```python
scaler = torch.amp.GradScaler("cuda")
```

Rather than just memorizing that line, let's understand what it does.

Recall what happens during training: say our model predicts `0.87`, the real answer is `1`, so we're off by `0.13`. We call `loss.backward()`, and say one gradient comes back as `0.4`, another as `0.003`, another as `0.0000000009`. Some gradients are big, some are tiny. If we're using FP16 everywhere, the tiny ones can underflow to 0, and learning either slows way down or stalls entirely for those parameters.

So what if, instead of keeping these numbers so small, we scale them up — multiply them by 1,000 (or some other factor) just to keep them representable in FP16? Think of it like looking at an ant from far away — hard to make out any detail — but zoom in 1,000x, and it's the same information, just clearer.

But wait — doesn't that change the math? Yes, temporarily — but that's fine, because PyTorch multiplies by that factor purely to store the value safely in FP16, and then divides back by the same factor once it's time to actually update the weights, landing back at the original scale.

That's exactly what `GradScaler` does. In code, you'll see this pattern constantly:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Put together in full:

```python
optimizer.zero_grad()

with torch.autocast("cuda"):
    prediction = model(x)
    loss = criterion(prediction, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

One more thing worth adding: we'll often reach for BF16 instead, since it has a much wider numerical range. The general rule:

- **FP16** → usually pair `autocast` **with** `GradScaler`.
- **BF16** → usually just `autocast` alone is enough.

BF16 support shows up on newer architectures, like NVIDIA's Ampere, Hopper, and Ada generations.

The next topic is one we'll always be using: Dataset!

## `DataLoader` + `Dataset`

Imagine we have a tiny medical dataset:

| Age | Weight | Has Disease |
| --: | --: | --: |
| 20 | 65 | 0 |
| 35 | 82 | 1 |
| 41 | 70 | 0 |
| 52 | 91 | 1 |

We could recreate this in Python easily enough:

```python
X = torch.tensor([
    [20, 65],
    [35, 82],
    [41, 70],
    [52, 91]
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [0],
    [1]
], dtype=torch.float32)
```

But here's the sad reality: you'll never get handed just 4 rows of data with a boss saying "use this for your model." You'll get millions of rows, sitting in a file or a SQL database — so you can't just write:

```python
prediction = model(All_the_data_pretty_please)
```

Do that, and you'll get a beautiful OOM error.

So the first key idea: **don't load everything at once.**

If you're reading a 1,000-page book, do you memorize all 1,000 pages in one sitting? No — you read page 1, then page 2, and so on. PyTorch does the same thing, just with batches:

```
batch 1
↓
batch 2
↓
batch 3
↓
...
batch n
```

Say we have 1,000 samples and set `batch_size = 100`. PyTorch creates:

```
Batch 1
Samples 1-100
------------
Batch 2
Samples 101-200
------------
...
------------
Batch 10
Samples 901-1000
```

Instead of swallowing a whole piece of meat in one bite, we eat it in 10 smaller ones. Why is that better? Say one sample is 4 KB — with 1,000 samples, that's `1000 * 4 KB = 4 MB`, totally fine. But with 100,000,000 samples? Impossible to load all of that onto a GPU at once.

Let's start with `Dataset`.

Think of `Dataset` as a giant drawer. If we write:

```python
dataset[42] # This opens a file on our machine, reads a small chunk, tokenizes it, and returns one training example
```

it returns `(features, label)`. That's how we'd inspect any specific example.

If our raw data contains something like `name`, that's a useless feature for training — and `Dataset` helps here too, since it can automate a pipeline like:

```
Read row
↓
Remove "Name"
↓
Normalize Height
↓
Normalize Weight
↓
Convert to Tensor
↓
Return
```

every single time, automatically.

There's another helper here too: `__getitem__()`, which lets us support indexing and slicing on our own custom objects.

Now say we have a file called `Airi_grades.csv`. Let's see whether this next approach is professional or not:

```python
import pandas as pd
import torch

# Read the ENTIRE CSV into RAM.
# If the CSV is 20 GB and your laptop has 16 GB RAM... Congratulations, your program just died.
df = pd.read_csv("Airi_grades.csv")

X = df[["Math", "Physics", "Chemistry", "Hours_Studied"]]

y = df["Passed"]

# Convert everything into tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# Feed the WHOLE dataset into the model
prediction = model(X)
loss = criterion(prediction, y)
```

Cute? Yes. Professional? Not really. You just dumped 20 GB into your laptop's memory in one shot. Dangerous.

But this is a good jumping-off point to explain `__len__()`, `__getitem__()`, and `Dataset` properly:

```python
import torch
import polars as pl
from torch.utils.data import Dataset

class ExamLogDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pl.read_csv(csv_file).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

dataset = ExamLogDataset("random_grades.csv")

total_rows = len(dataset)

print(f"The file has {total_rows} rows")

```

Let's understand what this does:

1. `def __len__(self): return len(self.data)` — say our CSV has 20 rows; `len(self.data)` tells us how many rows the file holds (`self.data` is the variable holding our file's contents).
2. `def __getitem__(self, index)` — this lets us use indexing and slicing on our custom object.
3. `return torch.from_numpy(self.data[index]).float()`, broken down further:
   - `self.data[index]` — since `self.data` is already a NumPy array, we grab the exact row with standard NumPy indexing.
   - `.float()` — converts the datatype to Float32, our go-to standard for neural networks.

Now let's look at this from another angle. A professional would want to feed data in small pieces:

```
Student 1-50
↓
GPU
-------------
Student 51-100
↓
GPU
-------------
Student 101-150
↓
GPU
```

As mentioned, that's what we call batches. `Dataset`'s job here is genuinely simple — it just answers two questions:

```
Question 1
How many samples exist?
↓
__len__()
```

and

```
Question 2
Give me sample #17.
↓
__getitem__()
```

But we still need batching, since loading everything at once is bad, and training one sample at a time is also bad — if we trained with a for-loop over each individual student, it'd take 1,000,000 separate updates to get through everything. That's why we never do that, and use `DataLoader` instead:

```python
loader = DataLoader(
    dataset,
    batch_size=50,
    shuffle=True
)
```

Here we pass in our dataset, set 50 students per batch, and set `shuffle=True`. Why shuffle? Because without it, we'd get:

```
# shuffle=False
Epoch 1: Batch 0 -> [Item 0, Item 1, Item 2, Item 3]
Epoch 2: Batch 0 -> [Item 0, Item 1, Item 2, Item 3]  # Same every time

# shuffle=True
Epoch 1: Batch 0 -> [Item 87, Item 12, Item 304, Item 5]
Epoch 2: Batch 0 -> [Item 42, Item 991, Item 3, Item 18] # Different each time
```

Leaving shuffle off during training is a bad idea — imagine 1,000 cat images followed by 1,000 dog images. With shuffling off, the model would see every cat image first, then every dog image — a recipe for bad training.

Here's what happens inside our `DataLoader`:

- On the first iteration (batch size 50), the `DataLoader` says: "Dataset! Give me 50 students, now." The dataset runs:

```python
__getitem__(0)
__getitem__(1)
__getitem__(2)
...
__getitem__(49)
```

  and instead of handing back 50 tiny separate tensors, it stitches them into one big tensor called `X_batch`, shaped `(50, <number of features>)`. The labels become their own tensor too, called `y_batch`.

I'll call the next section "projects," since I like the sound of that — and I'll write some code using a few new ideas I haven't covered yet, explaining them as we go. One of the new ideas here is genuinely important, so let's see if it clicks.

## Projects of Chapter 1 and Chapter 2

As mentioned, I'll lean on a few niche ideas we haven't discussed yet, but I'll do my best to explain them along the way.

```python
import math
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.manual_seed(9)


# ================== DATASET ==================
class StudentsDatabase(Dataset):

    def __init__(self, csv_file, mean=None, std=None):
        df = pl.read_csv(csv_file)
        useful_features = [
            "hours_studied",
            "sleep_hours_night_before",
            "practice_tests_taken",
            "teacher_rating",
        ]
        label_col = "score"

        raw_X = df.select(useful_features).to_numpy().astype("float32")

        self.mean = mean if mean is not None else raw_X.mean(axis=0)
        self.std = std if std is not None else raw_X.std(axis=0)

        normalized_X = (raw_X - self.mean) / (self.std + 1e-8)

        self.X = torch.from_numpy(normalized_X)
        self.y = torch.from_numpy(
            df.select(label_col).to_numpy().astype("float32")
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ================== MODEL ==================
class ScoreEvaluation(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


# ================== TRAINING ==================
database = StudentsDatabase("exam_attempts_log.csv")
loader = DataLoader(database, batch_size=50, shuffle=True)

model = ScoreEvaluation().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = nn.MSELoss()

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

for epoch in range(30):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_epoch_loss = running_loss / len(loader)

    if epoch % 2 == 0:
        print(
            f"Epoch {epoch + 1}: Avg MSE = {avg_epoch_loss:.4f} | RMSE = {math.sqrt(avg_epoch_loss):.4f} points"
        )

"""
Output:

Epoch 1: Avg MSE = 2358.9833 | RMSE = 48.5694 points
Epoch 3: Avg MSE = 96.8511 | RMSE = 9.8413 points
Epoch 5: Avg MSE = 75.5645 | RMSE = 8.6928 points
Epoch 7: Avg MSE = 73.7411 | RMSE = 8.5873 points
.......
Epoch 25: Avg MSE = 72.4339 | RMSE = 8.5108 points
Epoch 27: Avg MSE = 72.4810 | RMSE = 8.5136 points
Epoch 29: Avg MSE = 72.9445 | RMSE = 8.5408 points
"""
```

Before we go further — **the code itself is completely fine and works great, it's just that I used a messy CSV file** (that particular file had some pretty messed-up data in it).

Let me explain the parts you probably haven't seen yet:

1. `self.network = nn.Sequential(...)` — before anything else, `self.network` is just a variable name, nothing more. We use `nn.Sequential` because without it, we'd have to define every layer individually and manually pass data through each one step by step, the way we did earlier:

```python
def __init__(self, in_features):
	super().__init__()
	self.fc1 = nn.Linear(in_features, 32)
	self.fc2 = nn.Linear(32, 16)
	self.out = nn.Linear(16, 1)
	
def forward(self, x):
	x = torch.relu(self.fc1(x))
	x = torch.relu(self.fc2(x))
	x = self.out(x)
	return x
```

   With `nn.Sequential`, the output of each layer automatically becomes the input to the next one.

2. `for epoch in range(30)` — we're using a small number here on purpose, instead of the usual 1000+, because we're also accounting for batch size. Each run updates on 50 samples, and with 1,500 rows total: `1500 / 50 = 30` batches per full pass through the data.
3. `model.train()` — always call this before training. When we evaluate the model later instead, we'll call `model.eval()`.
4. `running_loss = 0.0` — when we compute `loss = criterion(prediction, y_batch)`, PyTorch hands back a tensor holding the loss for just that one 50-student batch. So to track the loss across *all* batches, we reset this accumulator at the start of every epoch.
5. `loss.item()` — since PyTorch remembers computation history, `.item()` strips away the computational graph and gives us back the plain loss value.
6. `running_loss += loss.item()` — we accumulate the loss across every batch, then divide by the number of batches to get the average loss for the epoch.

Since we'll rarely write everything out by hand like this going forward, let's move on to the next topics. (Keep what you've learned about PyTorch fresh in your mind — we'll circle back to it plenty.)

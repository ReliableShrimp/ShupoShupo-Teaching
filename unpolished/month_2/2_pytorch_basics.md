
That a really important topic, y'know? Because we will use it too many times, that why...

We will not learn only basics, because basics are boring, we will go from basics to intermediate or an advanced level. Soooo, we will... suffer? Idk, I come back at the end of the topic and tell if it was hell or no. (If there is no message in bold, that means that I forgot, Claude will add if I forgot or no)

anyway, I opened my beautiful study plan... and saw spectral graph theory there - first opinion... was doom, yet I will make it alive, because if I don't understand the spectral Graph, then month 4 and 5 will demolish me. Wish me luck (I wish you luck, reader)

The first idea we want to start with is...

# Chapter 1

## 1) Tensor

Anyway, what is a Tensor? 
Imagine this...

on python we may have:
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

But what are they? They are all Tensors - why?

Because Tensor is simply a general name for data organized into one or more dimensions.

So, basically the easiest idea is talking about LEGO, because imagine having a Lego, a row of bricks (made out of Lego), a Lego wall, many walls of Lego stacked together. 

Why? Because at every step you are still working with Lego blocks, just organized in a different way.

Presume we have a list of 10M numbers, why wouldn't we use a python list? 
Yeah, a python list can hold them for sure, yet the problem is that a PyTorch tensor was made for this, a PyTorch tensor is designed to:

- Fast for mathematical operations.
- Memory-efficient.
- Able to run on a GPU, where thousands of calculations happen in parallel.
- Able to track gradients, which is essential for training neural networks.

But we will see how to do each part.

So we understood that a tensor is a container for numbers arranged in one dimension or more.

## 2. PyTorch commands and Socrates moments

And now imagine that you mastered PyTorch, but one day Archimedes comes at the 5:00 AM and wake you up, asking you a deep philosophical question: 

"By the immortal levers of Syracuse, pray enlighten me: why must we abandon the steady sands of NumPy to worship at the altar of this PyTorch Tensor?"
$\downarrow$ 
Simple translation: "Why would you use PyTorch Tensor instead of NumPy arrays"

Such a deep philosophical question, yet the answer is simple.

Yup, NumPy can do the right math... It can do multiply, do statistics, do linear algebra, and so on. But why PyTorch Tensors?

NumPy was built for numerical computing.
PyTorch was built for machine learning.

Already sounds more interesting, but let's see how exactly.

For example:
Airi Sezaki is building a hospital AI.
So she does something simple as:

```python
import numpy as np

X = np.array([
    [36.8, 70],
    [39.1, 120],
    [37.0, 82]
])
```

But she wants to predict who is:

```
sick

or

healthy
```

So she writes: $y = Wx + b$ 

Right, but she needs the gradient descent:
$$\frac{\partial{L}}{\partial{W}}$$
This:
must be computed.
Every iteration.
For every parameter.

So, till Airi has 2 weights - everything is fine.
Till Airi has 50 weights - everything is fine (Yet her hand hurts)
Till Airi has 10 Millions weights - it is not daijoubu

Because you will explode to do all by hand! To compute every iteration for every parameter by hand is laughable as joke, but not funny in real life.

while 

1. The memory


- NumPy Limitations

NumPy see just numbers:

```python
import numpy as np

x = np.array([2.0])
y = x * 5 + 3
```

Its literal job is literally to do:

```
2.0

*

5

+

3

=

13
```

and that it, it knows nothing beside it

- PyTorch

Now here is the deal, because PyTorch actually remembers what it did, as:

```
Start

Multiply by 5

Add 3

Answer
```

Why do we even care? 
Because now we can ask:
"How much did changing x affect y?"

That exactly what the derivative measures.

For example:
If we calculate what is 5 + 5 on NumPy, as soon as we leave the program, it literally forgets what is 5 + 5, while PyTorch remembers them - this will be really useful, because it stores a "Computational Graph" (We will learn it later) and because it can actually calculate $y$ and $\frac{dy}{dx}$

2. The CPU

NumPy is great at making calculations, but imagine if we write `x * y`, we may think that nothing happened, but then we hear the computer make strange noises and the fans making the sound of an airplane that is about to take off. Since NumPy throws everything on the CPU and it prays to don't hit a memory bottleneck. But why this may happen?

It happens because the machine may do too many operations under hood, for example it will be fine my multimpying 100 times something, but how will it feel to make 1 Billion operations? Scary.

But that not the case for PyTorch, because with a simple command as:

`x = x.to("cuda")`

PyTorch will send the tensor to the GPU.

But why do we care?
Imagine you have 2 massive arrays (tensor) of size 10_000 x 10_000, just to multiply them, you may have to perform 1 billion multiplications and additions.

Numpy -
The CPU (The PhD mathematician) :
It will look at the problem, think, and then answer, after 10 seconds (The CPU is limited to the numbers of cores)

PyTorch -
The GPU (The thousands of kindergarten teachers):
The kindergarten teachers are not really smart, yet if we split the multiplication phase to those thousands, they all will finish it together and much faster than the CPU (Because the GPU distributes all the operation to all the cores). The operation may be solved in 0.20 seconds or faster.

We will need to use PyTorch because a neural network makes billions and trillions of operations.

3. Neural Network

Presume Airi wants one neural layer.

With NumPy:

She will need to implement:

- initialize weights
- initialize bias
- matrix multiplication
- activation functions
- gradients
- optimizer
- parameter updates

everything.

While with PyTorch she can just do:

```
layer = nn.Linear(128, 64)
````

Done.

PyTorch already provides building blocks designed for deep learning.

That doesn't mean we have to forget about NumPy, since we can use it every time we want general numerical computing. That why we should use PyTorch when we do just `2 + 2`



But anyway... what is inside a tensor?

## 3. Tensor from inside

We may think, what is the difference between:

```python
x = [2, 5, 8]
```
Which is:
```
Python List

┌──────────────┐
│10│20│30│
└──────────────┘
```

and

```python
x = torch.tensor([2, 5, 8])
```
Which is:
```
PyTorch Tensor

┌──────────────────────────┐
│ Tensor                   │
│                          │
│ 10 20 30                 │
│                          │
│ + extra information       │
└──────────────────────────┘
```

We know the difference of a tensor... but what does it has inside?

A normal python list is a python list

While a tensor is a list that has even other information in it.

For example, a patient doesn't have just a name, it has even an age, a size, a weight, a height, and so on... - it has attributes

PyTorch has attributes too:

```python
x = torch.tensor([2, 5, 8])
```

it has:

- shape
- data type
- device
- number of dimensions
- and more.

Let's take one as example for now: `shape`

```python
print(x.shape)

"""
Output:

torch.Size([3])
"""
```
What does that mean?

It actually just tells us how many elements exist in each dimension.
A vector has just a dimension, so the shape has just one number.

What if we make it a matrix?

```
A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])
```

Which is literally:

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

2 rows and 3 columns. That the easy idea that we already knew, but it was worth making you remember.

Let's say you upload an image dataset. The first thing you will do is:

```python
print(image.shape)

"""
Output:

torch.Size([32, 3, 224, 224])
"""
```

So you will notice (and know with what you are working):
```
32 images
↓
3 color channels (RGB)
↓
224 pixels tall
↓
224 pixels wide
```

Imagine seeing tensor for the first time, must be scary, that why I want to scare you, but even to un-scary it. So I will teach you how to see the perfect shape of tensors.

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

What shape it has?
Let's look at it!

Of how many blocks is it composed?
```
[[
 [Block 0]
]
[ 
 [Block 1]
]]
```

So we already write:
(2...)

Let's go to block 0:

```python
[
 [1,2],
 [3,4]
]
```

how many rows it has?
2.
how many columns?
2

We add them:

```
torch.Size([2,2,2])
```

That how we got the right shape.
But there is a problem, just recognizing them is not enough, we need to know how to build them with our bare hands. 
So let's start!

### 1. Building Tensors with bare hands

We already know the idea of samples and features:
More samples (more rows)
More features (more columns)

If not really (I doubt it), I will give a small example:

You have a basket, and in the basket you have 3 apples
```python
basket = torch.tensor([1, 2, 3])

print(basket.shape)

"""
Output:

torch.Size([3])
"""
```

Now imagine that you have a shelf with 3 baskets, each basket has 
`basket = [total_pears, total_apples, total_oranges]`

On python this information looks like:
```python
basket = torch.tensor([
[3, 8, 2], # Basket 1: has 3 pears, 8 apples, and 2 oranges
[12, 4, 1], # Basket 2: has 12 pears, 4 apples, and 1 orange
[1, 6, 9] # Basket 3: has 1 pear, 6 apples, and 9 oranges
])
```

But what if... we go to a warehouse that have 3 shelves, each shelf has 3 baskets, and they have ... pears, apples, orange

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

But what if we have 2 warehouses??

On python it will be like:
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

and i'll tell immediately, don't read `torch.Size([2, 3, 3, 3])` as:
"Two threes and three threes", no. Read it after the information you have, as:
"We have 2 warehouses that contain 3 shelves and each shelve contains 3 baskets with 3 features."

## 4. Basic commands and ideas

Now we will learn some PyTorch basic commands, because we will see them many times, and if I don't explain them, it would be hard to understand the rest (No worries, it will be hard even with this commands).

### 1. torch.tensor

We already know this one pretty much, it will just make a tensor, so there is nothing special in it, yet it has some special ideas, as:

```python
shelf = [
    [3, 8, 2],
    [12, 4, 1],
    [1, 6, 9]
]

x = torch.tensor(shelf)
```

We just converted a python list into a tensor! PyTorch understands by himself by this list:

```
Shelf

Basket 1
Basket 2
Basket 3
```

and builds the corresponding tensor. 
But what if we have a higher dimension? 

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

It will make from it another tensor, because PyTorch doesn't care whether it's a vector, matrix, or 7-dimensional tensor. 
It simply builds the matching tensor.

And beside it, we have some rules - as usual.

The shape must be identical, and that normal:

```python
x = torch.tensor([
[1, 2, 3],
[4, 5]
])
```

That  clearly a red flag, and PyTorch will throw at you:

```Python
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

Which means: 
"Dude, I expected every single row to have the exact same number of items! You gave me 3 items in the first row, so I locked in that length for the entire dimension. Then you handed me 2 items in the next row!"

Sooo, that his bro message to us. 

### 2. torch.zeros, torch.ones

I guess you already know what they mean. Because we can already read what they can do.

```python
torch.zeros()
```

this is literally our beautiful `np.zeros()`, it will create a tensor with 0 as value in as many rows and in as many columns we want.

for example:
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
# The shape is of (3, 3)
```
What if we want a 3D tensor?

We will just do:
```python
x = torch.zeros(2, 3, 4)
```

and so we will get 2 shelves with 3 baskets and 4 features per basket.

But where we will use them?
We will use them really often for:

- Initialize an image
- Empty matrix
- Create predictions
- Reserve memory
- Start an accumulator

What about `torch.ones`?
We will use it more rarelly, but it does the exact same stuff, just with ones:

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

Where will we use it? We will use it in:

- Masks
- Scaling values
- Multiplication identities
- Initial values

### 3. torch.rand, torch.randn, torch.arange

Yatta, here the RNG start, but sadly the "Not-So-RNG" is not betwen -inf and inf, because that would messed up and nobody would ever use it. That why we have limits for them.

We start from `torch.rand`:

What is better than some random numbers that are constantly stuck between 0-1. Because it will give random numbers just between those two numbers. Look:

```python
import torch

w = torch.rand(6, 3)

"""
Output (Will change totally next time I run the code):

tensor([[0.2217, 0.2059, 0.4587],
        [0.4020, 0.8286, 0.6479],
        [0.8227, 0.5383, 0.1917],
        [0.7997, 0.5655, 0.7805],
        [0.2903, 0.3888, 0.7785],
        [0.2915, 0.1363, 0.5748]])
"""
```

Now we have a casual weight, and the outcome is always different, but you will better do it as:

```python
import torch

torch.manual_seed(9) # We choose a random number
w = torch.rand(6, 3)
"""
Output (Will always be the same, since all the random became constant):

tensor([[0.2217, 0.2059, 0.4587],
        [0.4020, 0.8286, 0.6479],
        [0.8227, 0.5383, 0.1917],
        [0.7997, 0.5655, 0.7805],
        [0.2903, 0.3888, 0.7785],
        [0.2915, 0.1363, 0.5748]])
"""
```

But as we already noticed... The outputs (as already said 20 times) they are between 0-1 in a uniform way (The chances are identical for each number between 0-1). And that may be really annoying. Another stuff why it annoys everybody, is that because they have no negative number. So it has some nuances that may break our model.

And here comes... `torch.randn()`, which doesn't stay always positive, it can be even negative. And it actually uses a idea that I suppose you already understand - the normal distribution (Gaussian distribution). It will give have chances as:

- Within ±1: About 68.27% of the generated values fall between -1 and 1.
- Within ±2: About 95.45% of the generated values fall between -2 and 2.
- Within ±3: About 99.73% of the generated values fall between -3 and 3.

So that is how it goes, for example (The same from above, but now we will show the difference):

```python
import torch

torch.manual_seed(9) # We choose a random number
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

What about `torch.arange`? They are exactly like range, but instead of making a normal list, it will make a tensor, for example:
```
torch.arange(start, end, step)
```

We will choose a starting number, a number which will be the last and the the distance between them (Think about it as: "Choose a number from start to end and tell me to stop after every 'x' steps")

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


another ways of making it (Start from x and stop before y)
```python
import torch 

# x = torch.arrange(x, y)
x = torch.arange(2, 8)
print(x)

"""
Output:

tensor([2, 3, 4, 5, 6, 7])
"""
```

another way
```python
import torch

# The default of start will be 0 if not added and the default of steps will be 1
x = torch.arange(5)
print(x)

"""
Output:

tensor([0, 1, 2, 3, 4])
"""
```

What if we want floats? What will we do?
We will simply write:

```python
import torch

x = torch.arange(0, 1, 0.2)
print(x)

"""
Output:

tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000])
"""
# Remember that the end number is nuver included
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

What are they useful for?
They will help us in:
- looping over tensors
- creating labels
- generating positions
- building masks
- positional encodings in transformers
- many other tensor operations

## 4. Data types of tensors

Now we will discus about the beautiful `dtype` you saw on `numpy`, that I never explained (Because I am a tad silly). That why I will explain them to you, and make you understand what to use when.

We will start with the

### 1. `torch.int64`

What does it even mean? Let's break it piece by piece:

- int - that just an integer (Remember that an integer is a whole number. Because If you try to store `3.14` in a tensor with the data type of `int64`, PyTorch will automatically delete the decimal part and store it as `3`.)
- 64 - this are the numbers of bits used to store a single number. In our case that `64 bits = 8 bytes of RAM per number`. It can store a astronomical amount of numbers, because it can hold $2^{63}$ distinct numbers.

`torch.int64` can be positive and also can be even negative, but it can't contain any decimal.

### 2. `torch.float32`

We will break this one to pieces too:

- float - this let us save the decimals that get deleted when using `int`, so they are really useful in real environment, because when we do random weights or biases, we can't use `int`, because there is no way that the number will be whole. That why we will use `torch.float32` most of the times.
- 32 - this tell us that a single number will use 4 bytes. It's going to be able to store numbers that go in negative till $1.4^{-45}$ and positive numbers up to $3.4^{38}$

But now we have a question... Why to use `float32` when we can choose `float64`? And that a cute question, yet I'll give that question a reality check; float64 (Double Precision) is significantly slower on most consumer GPUs. 
Let's say that we have 1 billion parameters, so which is better to use?

- `float32` -  that a really great option, it consumes 4 bytes per number and because of that it will consume just 4 VRAM and get the result relatively fast...
- `float64` - not a good option, because it would take 8 bytes per number, and since we already have 1 billion parameters, it would consume 8 VRAM and be really slow...

So that their biggest difference. So the more weights we have the smaller type we will use... down to `float16`,  `int4`, and so on...

But for our neural network, we will spam `float32`

### 3. torch.bfloat16

This is another fan favorite. It is used to train large models. Even if the precision is halved, it is still reliable and by far the most used when parameters gets too big.

But why wouldn't we use `float16` instead of `bfloat16`? Because `float16` is too weak for big models. It has 1 sign bit, 5 for exponent, and 10 for mantissa... and beside this, it works till the number is smaller than 65,504. That why as soon that in our training the numbers get too big, this will make everything crash.

What about `bfloat16`?
It has 1 sign bit, 8 bits for exponent, and 7 for mantissa.
And the most important... it can reach a massive range of numbers (up to $3.4 \times 10^{38}$). You effectively never get the "Infinity/NaN"


So the final comparison is:

| Type           | Exponent (Range) | Mantissa (Precision) | Best For                                 |
| -------------- | ---------------- | -------------------- | ---------------------------------------- |
| **`float32`**  | 8                | 23                   | Precision-critical small models          |
| **`float16`**  | 5                | 10                   | Inference (Running models quickly)       |
| **`bfloat16`** | 8                | 7                    | Large-scale Training (Stability & Speed) |

That how it works.

### 4. torch.bool

We use this buddy boy only when doing mask, because while float32, bfloat16 are concerned for our value and data... bool is concerned about the logic. It is a perfect choice for boolean logic, that why it always appears when we do:

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

That how simple the bool idea works.

## 5. Tensor indexing

This is an important idea (Don't worry, I'll keep saying it and eventually the topic will be useful), because we will do that constantly in graph neural network, transformers, images, and so on....

Imagine we have a warehouse of boxes, and each box contain a number (we will have simply a vector).

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

But let's say that we need to get the number 20 a photo, because we need boxes that are strictly 20 (warehouse = 20).

What will we do?
We will just index it:
```python
mask = warehouse[3]
print(mask)

"""
Output:

tensor(20)
"""
```

But why did we write 3??? Isn't it the forth number? Yes it is, but sadly python counts from 0.

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

So we will always write the equivalent: 
$\text{Index number we want} - 1$

But python can even count backwards (Because apparently getting the number by its index number is hard, but counting backwards it is not.)
But we will use even that operations sometimes, for example:
```python
print(warehouse[-1])

"""
Output:

tensor(20)
"""
# It always return the last number.
```

But what do we do now? What if a monster of a matrix appear? How we index it?

### 2D
Let's say that to our warehouse new shelve appear.

```
Shelf 0
12   8   15

Shelf 1
20   9   11

Shelf 2
30   7   14
```

We count from zero too... because we are trying to become python, to fuse with it (Jokes, don't... at least if you don't have strange tastes)

```
warehouse = torch.tensor([
    [12, 8, 15],
    [20, 9, 11],
    [30, 7, 14]
])
```

let's say we want just 9.

```python
print(warehouse[1, 1])

"""
Output:

tensor(9)
"""
```

As usual, python counts rows and columns from 0:

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

So that how python sees the reality of our matrix.

what if we need the column `[8, 9, 7]`? We will simply index again:

```python
print(warehouse[:,1])

"""
Output:

tensor([8,9,7])
"""
```

What is `:` even? 
Think about it as "Gimme everything from there!"
Since we placed it on the row spot, we said "Gimme all the rows from column..."

What if we want all the columns from the first two rows? We can simply do this:

```python
print(warehouse[:2, :])

"""
Output:

tensor([[12,  8, 15],
        [20,  9, 11]])
"""
```

We used `(:2, :)` to say - "Give me all the rows and stop before 2 and all the columns".

Yet other ways of indexing exist:

`(:1, 2:)` - this means - "give me all the rows and stop before 1 and exclude all the columns before this number":
```python
print(warehouse[:1, 2:])

"""
output:

tensor([[15]])
"""
```

Now we will use another warehouse, to explain them better:

```python
import torch

warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]])

print(warehouse[:1, 3:])
# That means give me just row 1 and exclude the first 3 rows (Yeah, we count from 1, not 0...)

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

Now let's say we want an exact line...
like the `60, 65`:

```python
import torch

warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]])

print(warehouse[[1], 2:])
# "Give me just the row 2 (We count again from 0), and exclude the first 2 columns"

"""
Output:

tensor([[60, 65]])
"""
```

another examples:

```python
print(warehouse[[0, 3], :2])
# "Give me the first 2 columns (stop before 2) of row 0 (counting from 0)and row 3"

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

and the boolean one...
We will use the same warehouse:

```python
warehouse = torch.tensor([
[10, 20, 30, 35],
[40, 50, 60, 65],
[70, 80, 90, 95],
[100, 110, 120, 125]
])

print(x > 60)

"""
Output:

tensor([[False, False, False, False],
        [False, False, False,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]])
"""
```

This one is by far one of the most useful.

We will learn about the other types of indexing/slicing after some chapters, since right now it is useless to just show them and then not using them. So I will show them when we will need them.


## 6. Tensor operations

This part is way too easy for us, that why I will show a fast sketch and go to the next. 
"But if it is so easy, why would you even show us?"

I will show you, because on tensor you will not have to write always `torch.mean()` or `torch.sum()`, that why you will see soon~

---
Now we start with full basics, as:

1. Element wise - The matrices have to have the same shape and the will do the operation with the corresponding element that faces them:

```
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
2. Matrix multiplication

Here you can write `torch.matmul()` or - as we usually do - `@`. 

```python
print(A @ B)

"""
output:

tensor([[19, 22],
        [43, 50]])
"""
```

Now the only rule we will go by is the broadcasting one...

---
3.  Reduction Operations (`sum()`, `mean()`, `max()`)

On numpy we had to write always `np.mean()`, `np.sum`, and so on...
But on PyTorch we do:

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

You just write: `variable.sum()` < - It has to be tensor. (You have to have tensor imported.)

---
4. Reducing across specific axis (`dim=`)

Now let's say we have this big matrix!

```python
import torch

A = torch.tensor([
[1, 21, 3, 4, 52, 68],
[71, 8, 19, 199, 11, 9],
[100, 11, 15, 16, 57, 18],
[19, 65, 92, 2, 23, 10]
])
```

But what if we want to check by column only or row.

```python
# dim = 0 will check only the columns
print(A.max(dim=0))

"""
torch.return_types.max(
values=tensor([100,  65,  92, 199,  57,  68]),
indices=tensor([2, 3, 3, 1, 2, 0]))
"""
# It checked each collumn and got the biggest numbers from each.
# The indicies part is just saying at which row it found each number

# dim = 1 will check only the rows
print(A.max(dim=1))

"""
torch.return_types.max(
values=tensor([ 68, 199, 100,  92]),
indices=tensor([5, 3, 0, 2]))
"""
# It checked each row and printed only the biggest numbers
# The indicies part is just saying at which column it found each number 
```

---

Now we enter a dangerous topic... so be ready, prepare napkins, cuz it will be boring + painful.

## 7. Reshape tensors

Scary, yet this is our bread and butter. That why we will start directly. (I will try to make this concepts less scary)

### 1. `reshape()`

Imagine that Airi has 20 pieces of lego... what shapes can she do?

She can do 10 rows  of 2:
`10x2` - 10 rows, 2 columns

She can do 2 rows of 10:
`2x10` - 2 rows, 10 columns

She can do 2 blocks of 5 rows and 2 columns:
`2x5x2` - 2 blocks, 5 rows, 2 columns

She can do 5 blocks of 2 rows and 2 columns:
`5x2x2` - 5 blocks, 2 rows, 2 columns

To check if the shape is possible to be done, just multiply all the parts of the shape (rows with columns... block with rows and with columns) and in the end the product must be equal to the original
Now let's try it on python!

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

Now let's say I want 2 blocks of 2 rows and 5 columns per row... I'll do:

```python
A = A.reshape(2,2,5)

# Little check if we can do it:
# 2 * 2 * 5 = 20... we can!

"""
Output:

tensor([[[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9]],

        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]]])
"""
```

Now we reshaped as we wanted!... but what if we have 2 shops... with 2 blocks... and 5 rows of 1?

```python


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

That how `.reshape()` reshapes our tensor!

The problem is one... that reshape copies the data in the VRAM when the tensor is non-contiguous (was changed by using `.t()` or `permute()`), so it may actually use all of our memory without us knowing it.
But it will not copy the data if the tensor is not contiguous.

So it looks like this:

| Condition of Tensor | Does .reshape() copy data? | What does it return?      |
| ------------------- | -------------------------- | ------------------------- |
| Contiguous          | No                         | A View (shares memory)    |
| Non-Contiguous      | Yes                        | A Copy (brand new memory) |

That why the next one is a helper.
### 2. `.view()`

This is identical to reshape... but why would we use this instead of reshape?

View instantly crashes when it is used on non-contiguous tensors (for example, after a transpose `.t()` or a `.permute()`),  (But why would't we use reshape then?)
Right... But reshape as already said... copies the entire tensor to a new location in memory to force it to fit the new shape...

- If your tensor is a 2GB batch of high-resolution images, `.reshape()` will silently allocate another 2GB of VRAM and spend time copying the data.

- If this happens inside your training loop (running 1,000 times per minute), your GPU will instantly run out of memory (OOM error) or your training speed will drop to a crawl.

Using `.view()` acts like a smoke detector. If your memory layout is messy, it screams (crashes), forcing you to fix the underlying issue efficiently instead of letting your code run terribly slow.

That why all the professionals use always `.view()` and if it crashes by view, they will change to `.contiguous().view()` - We will learn that.
That why for now... stick to `view()`.

Now we will learn another concept...

### 3. `flatten()`

Suppose we have a matrix that is:
$$\begin{bmatrix} 0 & 1 & 2 & 3 \\\\ 4 & 5 & 6 & 7 \\\\ 8 & 9 & 10 & 11 \\\\ 12 & 13 & 14 & 15 \\\\ 16 & 17 & 18 & 19
\end{bmatrix}$$

Many times, as we continue... we will do some operations where we will use a vector instead of a matrix. And as you already understand, it is not cute to break it by hand, that why... we will use `.flatten()`

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

As we can see, it became a simple vector. 
And before you think that this is highly useless, I will immediately tell why to use it...
Because it is the bridge between CNNs and Linear Layer.

In CNNs (Convolutional Neural Networks) that is used in Computer vision, your image starts in 4D, but sadly, the classifier layer takes only 2D tensor: `[Batch_Size, Features]`
That why we will use `flatten()` to crush the tensor into a 1D vector, but since we need a 2D one, we will use `start_dim=1`, so this will tell PyTorch: "Leave the batch size alone, but crush the channels, height, and width into one long line"

```python
import torch

# A batch of 32 images, 3 color channels, 28x28 pixels
cnn_output = torch.randn(32, 3, 28, 28)

# Crush everything EXCEPT the batch dimension (dim 0)
flat_features = torch.flatten(cnn_output, start_dim=1)

print(flat_features.shape)
# Output: torch.Size([32, 2352])  <- (3 * 28 * 28 = 2352)
```

So that why this is so useful. Now we will continue... with something that is mind melting. Because life is hard and making it harder is always better.

### 4. `Unsqueeze()`

Now this part is hard... because `unsqueeze()` alters the dimensions, making a fake dimension... why would we ever do this??? 
We would do this because PyTorch is really strict about the dimension rules, because:

Imagine you have trained a neural network to look at batches of images. The model expects a 4D tensor shape: `[Batch_Size, Channels, Height, Width]`.

If you try to pass a single image (No more 'Batch_Size' as feature) into this model, your image will only have 3 dimensions: `[3, 28, 28]`. If you feed this straight in, your model will crash and complain that the dimensions don't match.

So that why, we would add a fake dimension.

for examples:
If we have a normal 2D tensor (matrix) as:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$
We can't do many stuffs with it... but let's say we want to make it into a single image... can we? Yes.

We will just do:

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
# As we can see, it added a dimension to the front... it added a shelf

---------------------------------------------------------------------------------
# Scenario 2: We used `unsqueeze(1)`
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
# We just changed from 2 rows to 1 row and gave this '2' to blocks.
# As we can see, it addedd a dimension in the middle.

---------------------------------------------------------------------------------
# Scenario 3: We used `unsqueeze(0)`
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
# We just gave the two to rows and the rows to block, and now we have 1 column.
```

The result looks like:

![[Pasted image 20260720191403.png|640]]


We will use it many times, in the future you will see how.

### 5. `transpose()` and `permute()`

By using transpose you will just with places two dimensions, for example...

Let's say Airi has 2 baskets and each basket has 20 apples... but she want 20 baskets with 2 apples per each.
That why she would use `transpose()`:

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

# We just changed the places... of the columns with the rows.
```

Let's say we have 3 warehouses with 5 baskets each, but only 2 apples per basket, yet we want 2 warehouses with 5 baskets and each basket shall have 3 apples.

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

# Swap dimension 0 (warehouses) and dimension 2 (columns)
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
# We just changed the places of warehouses with columns, while not touching the rows.
```

What about `permute()`?

We use permute because when we work with 3D, 4D, 5D+ tensors may get really messy, that why... permute let us change them just like indexing, instead of doing two at time everytime:

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

As we could see, we just choose the new position in a much easier way...

But they both have some nuances:
Something that is  worth noting is that:

- Both make the tensor non-contiguous.
- Calling `.view()` immediately after `transpose()` or `permute()` will crash. (As we already know)

That why you should be careful about it.

## 8. Cpu vs Gpu: `cuda`, `device`, and `.to()`

Before starting this topic, I would like to explain the difference between the CPU and GPU

1. CPU (Central Processing Unit):
     -  Think about the CPU as a group of 4 to 16 genius mathematicians.
     They will give you and solve: fast clock speed, complex decision-making, conditional logic (`if/else`), and managing operating systems.
     But... everything has a problem - as always. 
     They do all the actions sequentially (one or just a few actions at time)...

2. GPU (Graphic Processing Unit)
     -  Think about the GPU as 5000 kindergarten kids. None of them can do complex calculus, but every single one can perform simple math (like `2 + 2` or `3 * 5`) at the exact same microsecond. (That why we showed in the computational graph how it breaks it to pieces)

So that the big difference... 
But why do we need to know this?
Because Deep Learning is full of matrix multiplication (`@`) and element-wise math (e.g. `+`, `*`...)

So now imagine...
You want to add two vectors with 10,000 numbers:

- A CPU executes a loop: adds item 1, then item 2, then item 3... taking 10,000 sequential steps.

- A GPU assigns 1 number to 10,000 tiny cores and adds them all in 1 single step.

Your machine has also two main memory pools:

 - System RAM: Attached directly to your CPU.

- VRAM (Video RAM): Dedicated high-speed memory physically glued onto your GPU graphics card.

The System RAM is fast compared to the hard drive or to the SSD, but it is slow compared to the GPU.

The System RAM can be of:
- 8 gb
- 16 gb
- 32 gb...

And so on... 

The VRAM is usually smaller than the RAM (e.g., 6 GB, 8 GB, 12 GB, or 24 GB).

Now we will learn some ideas that will require the knowledge we used.

### 1. `device`

A question i'd like to start with is... where does our Tensor lives?

Every tensor has a device, and it may be `cpu` or `cuda:0`.

But how we discover this? We will simply do this:

```python
x = torch.tensor([1,2,3])

print(x.device)

"""
Output:

cpu
"""
```

The meaning is literally that these numbers are stored in your computer's RAM.

But there is a difference...
GPU has its own memory:

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

As we can see, data stored in RAM is invisible to the GPU, and data stored in VRAM is invisible to the CPU.

In other words, the CPU cannot directly access GPU memory, and the GPU cannot directly access system RAM...

### 2. CUDA

Imagine you buy a beautiful graphic card:

`NVIDIA GPU`

Now ask yourself...

How does python talk to it?
Because python has absolutely no idea how to communicate with a GPU.

There needs to be a language both understand.
Something international like english...

Because python right now is speaking "Python-ish", while the GPU is speaking "GPU-ish"
But there is actually a translator...

It is CUDA...
He will help python communicate to the GPU, because he will do a job like:

```
Python
    ↓
PyTorch
    ↓
CUDA
    ↓
NVIDIA GPU
```

Think about CUDA as an OS (Operating system) for your GPU

Your CPU has Windows or Linux.

The GPU also needs software to:

- receive jobs
- schedule work
- manage GPU memory
- launch thousands of threads
- return results

CUDA provides all of that.

But what happens when PyTorch adds one million numbers?

```python
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")

z = x + y
```

Do you think Python adds one million numbers? No.

The process looks like:

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

But there is a sad part...
Only NVIDIA can use CUDA... because CUDA belongs to them, sooo unlucky people with AMD (That me) and intel will have to use other stuff.

For  example:

```
NVIDIA → CUDA

AMD → ROCm / HIP

Apple → Metal

Intel → oneAPI
```

AMD will do the same stuff, just:

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

But why do we care about it?... Now you will see why.
### 3. `to()`

Let's say that you want to change where the tensor will exist, because it is usually:

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

Because when we do:

```python
x = torch.tensor([1, 2, 3])
```

It will automatically go to the CPU Room...
But what if we want this tensor to don't be in the CPU room, but to be in the GPU room?

Someone has to carry it, that why... we will use `.to()`

```python
x = x.to("cuda")
```

Even if we have AMD, Intel, or so on, `cuda` is the universal command to send the tensor to the GPU, that why we will not do something as: `.to("rocm")`, and so on.

We may use even `torch.cuda.is_available()` to check if we can use it; if it retuns `True`, then it is available, if `False`, then it is not.

For example, we may do:

```python
import torch

# Works identically on NVIDIA and AMD GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move a tensor to the GPU
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)

print(tensor.device)  # Returns 'cuda:0' if it was sended to the GPU
```

And so, the data goes as:

```
            .to()

CPU RAM                    GPU VRAM
┌───────────┐             ┌───────────┐
│ 1  2  3   │ ─────────▶  │ 1  2  3   │
└───────────┘             └───────────┘

Same tensor values.
Different physical memory.
```

## 9. Computation graph

We already know the concept, but pytorch works a bit differently, that why we will imagines something...

Presume you are doing some normal neural network...
(You have):

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

But there is a problem...
Here comes the backpropagation. Where now you are forced to find:

```
dy/dw1

dy/db1

dy/dw2
```

So what can you do? Till you have a small amount - you will be fine... but when you are building an ai? That will, maybe, have 5+ billions of parameters? Doing all by hand seems a tad hard... that why, PyTorch remembers every step we do...

PyTorch is like a student. You keep doing the math, and he will write down every step we do.

Imagine you are baking a cake, you do:

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

Pytorch will write down:

```
Step 1
↓
Step 2
↓
Step 3
↓
Step 4
```

so, everytime you do

```
y = x * 3
```

pytorch remembers:
```
multiply
```

Everytime you do:

```
z = y + 5
```

pytorch remembers:
```
add
```

so he will note:
```
multiply
↓
add
↓
...
```

And so on...

But why do we even care about it?
Because when we reach the loss, we ask ourselves... how did we get here, and pytorch knows how, that why it will not struggle as we do while trying to solve all the backpropagation by hand, it will simply know its step back and apply the chain rule. That why it is so useful.

This ability looks like this on python:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 5
loss = z ** 2
```

`requires_grad` - This will tell python: "Hey, track this computation. I will need gradients later."

Before frameworks like PyTorch became popular, researchers often had to derive and implement gradients manually for new models. That was time-consuming and error-prone.

But on PyTorch you can just do:

```python
loss.backwards()
```

This way we just describe the forward step, while PyTorch will do the rest of the backpropagation.

But there is a problem... what if we want to stop the tracking for a bit? Because we need just to do an element-wise math or something as this. Because letting PyTorch note everything will lead to a memory crash, because imagine tracking all the steps of a GPT you are building, and the GPT has 18 billions parameters. What will you do? Track everything?
No.

That why we will use `x.torch.no_grad`, we will use it as:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
z = y + 5 # Let's say that we want to make z a bit smaller:
with torch.no_grad():
	z = z - 4
loss = z ** 2
```

Now it will not record everything inside `with torch.no_grad()`.
Now we will continue with... backwards! 

## 10. `.backwards()` and `.grad`

I already spoke about backwards, but... I didn't explain how to use it, that why I will demonstrate you how!

imagine Airi is building a model with 2 layers, but she doesn't want to derive them - her hand hurts.
So what will she do? She will use the special ability that PyTorch has - `.backwards()`

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

we will use this:

```python
z1 = x * w1 + b1
a1 = torch.sigmoid(z1) # I will do like scientists back in 2010, spaming sigmoid.

z2 = a1 * w2 + b2
a2 = torch.sigmoid(z2)

loss = (a2 - target) ** 2  # MSE, practically (error**2)
```

Now Airi has to derive them... yet she 'accidentally' caught her hand in the door... so she can't derive them... that why she will simply use `.backwards()`

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

so now we will just write:

```python
loss.backwards()
```

So the process will look exactly like:

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

and eventually it will compute:
```
dLoss/dw2
```

same for:
```
dLoss/db2
dLoss/dw1
dLoss/db1
```

This all together are the gradients.

Now we can do:

```python
print(w1.grad)
print(b1.grad)

print(w2.grad)
print(b2.grad)
```

It gives immediately to us the exact values for each weight and bias we need. That why we just escaped from manual backpropagation-ing. 

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

The `.grad` literally stores the gradients of the weights/bias, for example if we do:

```python
w1 = torch.tensor(0.5, requires_grad=True)
```

for now we have:
```
w1
├── Value = 0.5
└── grad  = None
```

But after `loss.backwards()`, if we write:

```python
print(w1.grad)
```

we will have:
```
w1
├── Value = 0.5
└── grad  = -0.18
```

As we already noticed, PyTorch doesn't change the weights automatically, yet it says to us:
"if you want to reduce the loss, changing it in this direction would help."

Now, all the topics are pretty pivotal.
And we will start from the `Normalize`
## 11. Normalize

Many times we will have small features (e.g. Practice hours, kilograms lost the past month, How many times you eat per day... and so on ), and sometimes we have really big features (e.g. House price, Car price, Annual salary...).

Suppose we have:
```

Hours fishing      = 5
Water temperature  = 23
Annual salary      = $112,000,000
```

We clearly can notice something... One feature is millions of times larger than the others.

And now imagine doing:
$$y=w_1​x_1​+w_2​x_2​+w_3​x_3​$$
Now the problem is one... suppose the weights were equal to `0.1`

```
0.1 × 5          =          0.5
0.1 × 23         =          2.3
0.1 × 112,000,000 = 11,200,000

Total ≈ 11,200,002.8
```

As we already noticed... The Hours fishing, the water temperature are invisible... out of `11,200,000` they are worth just `2.8`. 

The model will think that this features are almost useless, so it will focus mainly on the Annual salary.

So during training the gradient descend asks:

"Which weight should I change?"

Since the salary feature produces much larger values, its gradients are usually much larger too.

```
Hours gradient       ▏
Temperature gradient ▎
Salary gradient      ███████████████████████████
```

So it will put much more effort to change the weights of the Salary, rather than the Hours or Temperature one.

That why we will not use the raw values, but we will normalize them so they will have a similar value.

For example:
```
Hours       = 0.42
Temperature = 0.58
Salary      = 0.63
```

Now the optimizer can learn from **all** of them instead of one feature dominating the others.

Now let's show it in a code, because if you don't know how to code it, you don't know the concept.

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

We can see the minimum and maximum:

```
Hours       :      2 → 8
Temperature :     18 → 24
Salary      : 45,000 → 120,000
```

We can clearly see that salary dominated all of them.
That why we will use this formula:
$$x_{new} = \frac{x - min(x)}{max(x) - min(x)}$$

This formula will concert all the features from a range of 0 to 1.

In python it looks like this:

```python
min_x = x.min(dim=0).values
max_x = x.max(dim=0).values

x_new = (x - min_x)/(max_x - min_x)
```

This is the baby steps.
Because we will use another formula this is much more used, and it looks like:
$$x_{new} = \frac{x - \mu}\sigma$$

We will write it as:

```python
x_mean = x.mean(dim=0)
std = x.std(dim=0)

x_standard = (x - x_mean)/std
```

The output will look like:

```
tensor([
[-1.16, -1.16, -1.09],
[-0.39, -0.39, -0.55],
[ 0.39,  0.39,  0.37],
[ 1.16,  1.16,  1.28]
])
```

Now, instead of having numbers from 0 -> 1, we have numbers centered around 0.

Now
We will start a harder lesson... 
We will write a custom `autograd.Function`.

## 12. `autograd.Function`

Before we start I wanted to explain what is an autograd, since we can't continue without it.
Autograd is is PyTorch's automatic differentiation engine... and I am sure you understood nothing, yet I'll give an example.

To get a small neural network, you do this steps on PyTorch:

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
When you run this code, Autograd is silently staring at each step we do...
It stores

- which operation happened
- which tensors were involved
- how to differentiate that operation

Now the backpropagation part comes:

```python
loss.backwards()
```

Now PyTorch says to Autograd:
"Autograd, please compute all derivatives."

So Autograd now walks through the graph backwards and for every operation it applies the chain rule.

So we can't say Autograd = Backpropagation.

Because Backpropagation is the algorithm. Autograd is the system that performs it automatically.
Autograde will slowly stare and then compute all by itself.

But why would we even do a custom Autograd function?

Because, sadly, Autograd doesn't know all by itself, because let's say we run a function in the middle of our forward step. Autograd will add it to its steps (graph), and when it reaches it? What shall he do? can he derive 

```python
cute_function_that_Airi_made(x):
	# Airi called a Julia library
```

It doesn't even know what is that. Even if it already added it to the graph, he still has no idea how to derive it. That why it will fail.

But if we do:

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

I am sure we understand nothing, but now I'll explain all the steps:

- `class SquareOfTwo(torch.autograd.Function` - This is just a normal class, but it tells to PyTorch: "I am defining a new differentiable operation" - we are just stating that we are doing a single mathematical operation that it doesn't know about. As PyTorch has `addition`, `subtract`, `multiply`, `divide`, and so on... we define a new type, called "SquareOfTwo".
- `@staticmethod` - Why we do this? Because we don't need self. Because as we already know, when we make a class:

```python
class Dog():
	def bark(self):
		print("Woof, woof")  
```

 We understand that when we will do:
 
 ```python
dog = Dog()

dog.bark()

"""
Internally will happen this:
Dog.bark(dog)
""" 
 ```

But we don't need that, because we never create an object. We will never do `Cute = MultiplyByTwo()`, we will do `MultiplyByTwo.apply()` - that is different.

- `ctx` - Imagine that we forward step start first and after some hours the backwards step start... the backwards step needs information, that why we will use `ctx` - Context. We will give to both a backpack, so they store everything there and later we will give this backpack to `backwards()` and he will understand everything. Meanwhile `x` is just out input. 
- `ctx.save_for_backwards(x)` - Suppose `x = 3`, then we will do `3 * 2 = 6`, but as we can understand, backwards needs to know this too, because later it will use it in the backpropagation, so it wants to be prepared. That why we save it! We store it in the backpack, so `backwards` will be able to open the backpack and then get x.
- `def backward(ctx, grad_output)` - we gave `ctx` (the backpack) to `backwards`, but what is grad_output?
  grad_output is the beautiful gradient of our function and loss, something as:
  $$\frac{\partial{L}}{function_{our}}$$

  As we understand: "If I tweak by a bit our function, by how much will our loss raise? fall?"
  That all the story of `grad_output`
- `x, = ctx.saved_tensors` - this will make `backwards` open the backpack and look inside it. So he will find our `x`. Why the strange syntax `x,`? Because python stores them always as tuple. Even if we have just a variable, python will store it as a tuple. If we had before something as: `ctx.save_for_backwards(a, b, c)`, we would do just `a, b, c = ctx.saved_tensors`.
- `local_derivative = 2 * x` -  that just our derivative (Awwww, how cute, they thought that they can escape from math)
- `grad_input` - this is the chain rule.
- `return grad_input` - we just return the value, but be careful. Because what we return must be one-to-one with the inputs of `forward`.

Now we wouldn't go too deep into it, because there is no use from it. That why we will go deeper in month 4/5.

So for now... we will learn two harder concepts, yet really useful. 

Optimizers and initializations.

But before the, we will learn `nn`. Because that the smartest choice.

# Chapter 2
## 2.1 nn.Module

Before I explain what `nn.Modules` are, I will make you understand it in no time.

Imagine that Airi wants to build a neural network - easy, right? Yup, pretty easy...
So she will build it from scratch:

```python
import torch

x = torch.tensor([2.0, 3.0])

W = torch.tensor([[1.5, -0.5]])
b = torch.tensor([0.2])

y = W @ x + b

print(y)
```

Really easy, but what now?
Imagine she wants to make a neural network... with many layers (Suppose 20+ layers).
She will be forced to write down:

```python
W1 = ...
W2 = ...
W3 = ...
W4 = ...
...
W22 = ...
```

and maybe even biases... that would be hellish!

That why `nn.Modules` exist.
Think about it as a toolbox that exists for Neural Network.

It has everything you need:

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

And much more. That why we would use it to build as many weights as we want, without writing them by hand.

We import it by using:
```python
import torch.nn as nn
```

The most common part is:

```python
nn.Linear
```

What is it? This is literally our linear equation:
$$y = Wx + b$$
Imagine we have three features.
Forget matrices and calculus for a second. Imagine you are a doctor looking at **3 patient stats**:

1. Temperature (39.0)
2. Coughs per minute (15.0)
3. Days sick (4.0)

You want to compute 2 scores:

1. Flu Risk
2. Cold Risk
3. ..... till 32

We want to insert three features and get back just 2 of them.
How this happens?
A single patient enters the room, and there a 32 doctors 
The first doctor takes the numbers of the patient and multiply them by the weights he thinks are right and add even add the bias there. 
Then the second doctor takes the exact same numbers, uses _their_ set of weights, and outputs a 2nd number.
and so it continues till the Doctor number 32.

At the end of layer 1, you have 32 features
But if we want more layers, we do:

```python
import torch.nn as nn

# Layer 1: Takes 4 patient features -> Outputs 32 hidden features
layer1 = nn.Linear(in_features=4, out_features=32)

# Layer 2: MUST take 32 features in, because we have 32 collumns, and we have to make the broadcasting work -> Outputs 16 features
layer2 = nn.Linear(in_features=32, out_features=16)

# Layer 3 (Output): Takes 16 features in -> Outputs 1 final decision score
layer3 = nn.Linear(in_features=16, out_features=1)
```

without this we had to do:

```python
W1 = torch.randn((4, 32), requires_grad=True)
b1 = torch.zeros(32, requires_grad=True) 
W2 = torch.randn((32, 16), requires_grad=True)
b2 = torch.zeros(16, requires_grad=True)
W3 = torch.randn((16, 1 ), requires_grad=True)
b3 = torch.zeros(1, requires_grad=True)
```

And our is even half baked.

Because the `nn.Linear` automatically creates the weights and biases. 
From where does he knows what weights to set?
It uses the He (Kaiming) Initialization... We will learn it as next topic.

But let's use it in a random code, to see how it works.

Airi went fishing, and she noticed that some factors make her catch more fishes and some less. 
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

# Target: Total fishes caught
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

that how we use `nn.Modules`. It will become a standard for all of our codes.

Now we will start with initializations:
## 2.2. Initializations

Setting the initial weights is easy, no? Don't you think so? We go there, set the weight to 0 or just use `torch.randn()`. We did it. Thanks for reading. Bye.

Not so fast. Because this is one of the most import concepts in all Ai and Machine learning, because I will show the two traps that we always did (We were childish, that why):

1.  **Setting all the weights to 0**

This one is cute, but sadly too dangerous for even trying. 
imagine a layer with 1000 neurons, all initialized with the exact same weights (all zeros). Since every neuron receives the same input and has the same weights, they all produce the same output.
During backpropagation, they also receive the same gradients, so they are updated in exactly the same way. As a result, those 1000 neurons are doppelgangers. We don't need that! This problem is called symmetry - you can become afraid of that word.  

For example:

```
Neuron A

w = [0,0,0]

Neuron B

w = [0,0,0]
```

Let us say the input is:

```python
x = [2, 5, 1]
```

Now we will do our:
```python
z = x @ w
```

Output for both neurons? 0.

Now presume that the backpropagation happened.

Both weights update... both of them are 0.7 (A random number). What does that mean? That the weights are literally identical. Don't they change?

Next iteration?
Still identical.

Next?
Still identical.

So we get 1 neuron that copies for 1000 times.

2. **Using a random start**

This much more standard as choice in modern deep learning companies and even models. But as expected, there is a problem. If the weights start too big, each layer can amplify the previous layer and we will get a mesh exploding gradients (We will get a way too big number or inf) or exploding activation.

While if the weights start from smaller numbers, eventually each layer shrink and in the end of it  we are going to get a vanishing gradient (Extremely close to zero), and that not so cute.

For example:

```
Neuron 1

0.13

Neuron 2

-0.44

Neuron 3

0.08
```

They are different! Problem solved! 
Not so fast. 

Because imagine that we have 100 layers...

you will do:

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

And so on till hitting the beautiful wall of infinity. 

Again. A problem.

That why we have two beautiful starts...

(Glorot initialization and Kaiming initialization)

### 1. Glorot (Xavier) Initialization

Okay, so here we have glorot initialization. It was invented in 2010 by Xavier Glorot and **Yoshua Bengio.

They didn't say like a random function:

"Choose a random number"

They said:

"How large should the random numbers be so the signal neither explodes nor vanishes?"

The answer depends by the number of inputs and outputs as:

```
Number of inputs

(fan_in)

Number of outputs

(fan_out)
```

The more connections a neuron has, the smaller each initial weight should generally be.

$$w \sim U\left(-\sqrt{\frac{6}{fan_{in} + fan_{out}}}, \sqrt{\frac{6}{fan_{in} + fan_{out}}}\right)$$

But for now I teached you just the idea of it, because we will almost never implement it by hand. Only in case we have deep layers! But for that... we will learn it later.

### 2. He (Kaiming) Initialization.

Later, Kaiming noticed that ReLu threw away all the negatives numbers and made them zero. So using Glorot wasn't the optimal choice, because all that Glorot was looking at was if the number stayed around zero. But ReLu threw this concept in the trash.

That why Kaiming made another initialization that is used only for ReLu, Leaky ReLu, GeLu... 

It looks like:

$$w \sim \mathcal{N}\left(0, \frac{2}{fan_{in}}\right)$$

So before we just memorize it, I'll explain the logic:

Imagine that you are going with 1000 people in 10 rooms, but some of them are sad (negative value), so what the inspector does? Doesn't let them pass (The negative numbers become 0), so by the time you reach room 10, half of the people were sad, so they got stooped (became zero). But what happens if we tell the other half that survived to be twice as happy? We get the same value of those 1000, because there are no 1000 happy people, but 500 happy, happy people. 

We will learn how to use it by hand when the time will come... for now this isn't still the right time

## 2.3 Optimization

This is another mega important step we should care about because without it everything would explode, as much as we wouldn't try.
Because imagine having 20+ weights and biases... imagine how funny it is to literally to write in the end:

```python
w1 -= dw1 * lr
b1 -= db1 * lr
w2 -= db2 * lr
...
dw26 -= dw26 * lr 
```

that why... we have optimization to save us `torch.optim`.

That basically PyTorch automatic parameters handler. You will simply write:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

In `model.parameters()`, PyTorch passes reference to all the tensors that were saved in our class as `fc1, fc2, fc3...`

for example, we will do this:

```python
for epoch in range(500):
	prediction = model(X_norm)
	loss = criterrion(prediction, y)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

Now let's break everything we didn't understand 

- `loss = criterrion(prediction, y)` - We will talk in the next chapter why would we use it.

- `optimizer.zero_grad()` - By default PyTorch accumulates gradients, it doesn't overwrite them. And because of that... the second epoch will be added to the first epoch, the third epoch will be added to the second ... and so on till infinity. That why we use `optimizer.zero_grad()` - it will go through all the `.grad` and set them to None (0).

- `optimizer.step()` - Now PyTorch will iterate through all the parameter tensor and update them.

But as we noticed...
At the start we wrote: `torch.optim.Adam()` - why?
There are many ways types, and now I'll explain the most famous:

1. SGD (Stochastic Gradient Descend)
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

This is the dumb way we used to do... because it uses the same learning rate for all of the parameters. 
$$w_{new} = w_{old} - grad \times lr$$
So if it hits a vanishing gradients, the learning rate becomes so slow that it seems like it is crawling.

2.  Adam (Adaptive moment estimation)
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

This is the standard choice for modern deep learning prototypes.

instead of blindly applying just the learning rate, Adam applies other two stats for every single weight we have:

- Momentum (first moment) - that the moving average of our past gradient. If the weight was moving positively for the next 20 times, it will increase speed in that direction - like a heavy ball rolling in that direction.
- adaptive scaling (second moment) - The moving average of squared gradients. If a parameter receives massive gradient updates, Adam lower its learning rate to keep it stable. If a parameter receives tiny gradient updates, Adam will raise the learning rate. So in both cases everything will be stable

3.  AdamW (Adaptive moment estimation with weight decay)
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Before explaining why to use it, I will explain what is weight decay.

Imagine that you cleaning your room (Let us say that our room is our neural network), and you own 500 things (weights)... but:
10 are actually useful
490 are actually useful

So each day you throw a tiny percentage away. Because if something isn't useful enough to stay big, it slowly disappears.
That's exactly what weight decay does. It says: "If the weight isn't important, slowly make it smaller."

In real life it would be something as:

Airi studies 7 hours per day, while Hinako just 30 minutes. But Airi's mom comes home and doesn't care that Hinako doesn't belong to their house, she forces both to clean the room.

In our case, Adam (The optimizer) would say:

"Since Airi studies so much... I will make Hinako clean 90% of the room, while Airi just 10%"

But that a bad stuff. Because studying $\neq$ cleaning. They are totally unrelated. But Adam tied them together.

But AdamW will act differently in the end. Because firstly it will do the same stuff that Adam did, move the weights, and then it will add some taxes... (Take away 1% from all the weights, there are no preferences, it will take 1%). Because PyTorch lumps both weights and biases together and taxes both of them.

And AdamW is used literally in: GGN, Transformers, MLPs, General deep learning, and so on...
So it is by far the better version of Adam.
But there is a secret... that we will learn later. That sounds like: "Why we shouldn't decay bases and never touch even normalization parameters."

## 2.4 Criterion

Maybe we saw in a code:
```python
loss = nn.MSEloss()
```

And thought... why in this world shall I use this? I can flex and use my skills as:

```python
loss = (error ** 2).mean()
```

wouldn't the output be the same? Yeah... maybe it will be the same...
But I can assure that PyTorch loss functions are much better than any formulas. 

Imagine this... you are making a pizza, and you can make the dough yourself:
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

The problem is that... you could buy a ready-made dough... by a professional and the outcome would be the same. It is just that the dough you buy is free and made by a professional.

But why we should use the PyTorch loss functions?

We should use them, because:

1. They have lesser bugs - Imagine that you accidentally did: `criterion = (error**2)` and forget about the mean.
2. They are easier to read, because somebody who sees `criterion = MSEloss()` will understand it easier than somebody who sees the formula.
3. They already handle edge cases, without us trying to do everything manually
4. They are optimized, so if we have 100 million parameters, instead of letting the machine make several temporary tensors and overwhelm the memory, PyTorch will often fuse operations internally. This will give to use extra speed and less memory consumption.
5. They are perfect for autograd, even if your manual version may be good, the PyTorch version is much safer, since millions of researchers tested it.

So let's see the difference in code.

Airi have been feeling really bad lately, so she goes to get a check up by different doctors.

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

I did this just to show the difference of the normalization...

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
```

That the gigantic difference between normalize and without it... what if we changed our code to full `nn.modules` version?


Let's use nn! and the concepts we learned here!

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
```

Let me explain what is `summary`.

Summary is basically the medical report of our model.
It will tell:

- The name
- How many layers it has
- How big are those layers
- How many neurons it contains.
- How many values it has to learn.
- How much memory it uses.

1. `DoctorCheckUp` - that the name of our class
2. `[10, 1]` - that the final output of our class
3. `├─Linear: 1-1          [10,32]      160`:
	1. `├─Linear: 1-1` - that just tell us that the first layer is a linear layer, and it is the first layer, since it has 1-1
	2. `[10, 32]` - it means that we have 10 patients, and each patient has 32 neurons
	3. Param: `160` - this means that our model has to learn 160 numbers. From where? Since our first layer looks like: `[4, 32]`, we will use the linear formula: X * W + b (In our case that is: 4 * 32 + 32)
4. `Totale params: 833` - That just the total parameters that gradient descent updated
5. `Non-trainable params: 0` - this is because we never froze anything (Freeze - use `.requires_grad = False) 

That the output of this. Which is really good from a side.

So let us learn something really good:
Mixed precision training

## 2.5 Mixed precision training (torch.amp)

What is this and why was it invented?...

Imagine that Hinako bought a RTX 5090 (I thought about writing: "Imagine that you bought...", but then I remembered that you are probably broke as me, so not even in our dreams).

It costs a fortune, it has a lot of CUDA cores, yet... your model uses just half of what your GPU is capable, why? Because you are feeding it the wrong type of numbers.

Everything in deep learning is numbers. Our neural network is just millions (or billions) of numbers as:

```python
Weight = 0.2356
Bias   = -1.734
Input  = 7.25
Output = 0.823
Gradient = -0.00412
```

But, our machine doesn't understand the numbers, it stores bits as:
```
0000011111000111000111000000111101101
```
(That a random spam, so don't really look at it, cuz I am not a compiler to understand that)

And we can understand that the more bits means:
- Better precision
- More memory usage

The most common ones are:

| Type | Bits | Bytes | Precision |
| ---- | ---: | ----: | --------- |
| FP64 |   64 |     8 | Very high |
| FP32 |   32 |     4 | High      |
| BF16 |   16 |     2 | Medium    |
| FP16 |   16 |     2 | Medium    |

Suppose your GPU memory is a warehouse and each floating number is a box.

If you store them in a FP32 - the boxes are relatively big...
if you store them in a FP16 - the boxes are relatively small...

Imagine your model has 100 million numbers -> Use FP32 -> 
```
100,000,000 * 4 (bytes) = 4,000,000 bytes -> 400 MB
```

Now imagine you use FP16 ->
```
1,000,000 * 2 = 2,000,000 bytes -> 200 MB
```

You roughly saved the double of the amount of your MB.
And FP16 is even faster in many cases than FP32, because the RTX 5090 has something called Tensors cores, which are highly specialized workers.

CUDA cores -> Normal worker
Tensor cores -> Specialized worker -> They specialize in multiplying matrices extremely quickly. Exactly what neural networks do all day. 

Because Tensors cores were designed primarily for BF16 and FP16... so if you use FP32... you are not taking full advantage of the cores.

But why doesn't everybody always use FP16?

Because as the table above shows us... The precision difference:

FP32:
```
0.123456789
```

FP16:
```
0.12345
```

But anyway, the neural network is highly tolerant to this small rounding.

Yet, as usual, there is a problem:
what if the gradient is small as:
```
0.00000018
```

or even smaller:
```
0.00000000002
```

If we use FP16, the rounded version could be 0 (underflow - that how this problem is called).

But remember the title: "Mixed precision training".

Instead of saying: "Everything must be FP16."

PyTorch says: "Let's only use FP16 where it's safe."

Here is where we use `autocast`:

A normal neural network
```python
x = self.linear1(x)
x = self.relu(x)
x = self.linear2(x)
loss = criterion(x, y)
```

This one uses FP32 by default.

Meanwhile if we use autocast:
```python
with torch.autocast(device_type="cuda"):
    prediction = model(x)
    loss = criterion(prediction, y)
```

Here we say to PyTorch: "You choose"

That why PyTorch will automatically check if it is safe to put FP16 here or there?

Now we will use the `scaler`, of which formula is:
```python
scaler = torch.amp.GradScaler("cuda")
```

But we will not just memorize it, let us try to understand it.

Let us remember what happens during training:

```
Presume our model predicts -> 0.87
↓
Real answer -> 1
↓
We are off by -> 0.13
↓
Now we call `loss.backward()`

Let's say one gradient recived 0.4, another 0.003, another 0.0000000009. As we noticed, some gradients are big, and some are really small. If we spam FP16 the neural network learning can become really slow or a part of it stops entirely.

But what if... We don't keep so small numbers... we multiply them by 1000? or other numbers, just to keep them afloat, so FP16 can store them.

Think about looking at an ant from a close distance... hard to see something, but what if we zoom by 1000 times? Same information, just clearer.

But there is a problem... Didn't we changed the math? Yes, we did. But no problems, because PyTorch multiplies it by one thousand just to save it in FP16, then, when the weights have to update, it will divide it back by 1000, so we have the same amount as before.
```

That is what GradScaler does.
So in a code we will see many times:

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

So fully it looks like:
```python
optimizer.zero_grad()

with torch.autocast("cuda"):
    prediction = model(x)
    loss = criterion(prediction, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

But I will add: 
Many times we will use BF16, because it has a bigger numerical range.

And the rule is simple:
- **FP16** → usually use `autocast` **plus** `GradScaler`.
- **BF16** → usually use `autocast` alone.

But BF16 is used on newer architectures as NVIDIA Ampere, Hopper, and Ada...

The next topic is the one we will always use.
Dataset!

## 2.6 DataLoader + Dataset

Imagine we have a tiny medical dataset.

|Age|Weight|Has Disease|
|--:|--:|--:|
|20|65|0|
|35|82|1|
|41|70|0|
|52|91|1|

We can recreate it on python, no?

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

But now we get hit with a sad reality...

You will never get just 5 lines of data and the boss saying to you: "Use this for your model".
You will get millions of data, but stored in a beautiful file/SQL. So you will not be able to write:

```python
prediction = model(All_the_data_pretty_please)
```

Because if you do so, you will get a beautiful OOM error.

So the first key idea is: 

1. **Don't load everything.**

Let's say you are reading a book with 1000 pages, do you memorize all 1000 in a try? Nope.
You read:
```
page 1
↓
page 2
↓
...
page 1000
```

That what PyTorch does too, just with batches:

```
batch 1
↓
batch 2
↓
batch3
↓
...
batch n
```

Suppose we have 1000 samples.
We do:

```python
batch_size = 100
```

then PyTorch creates:

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

Instead of eating a piece of meat in a big bite, we ate it in 10 smaller. Why is it better?
Because imagine that a sample is 4 KB. If we have 1000 samples we will get 
```
1000 * 4 KB = 4 MB
```

Fine, but if we have 100_000_000? Impossible to load all of them on a GPU.

Now we will start with the first idea: 
`Dataset`.

Think about Dataset as a giant drawer. 
if we write:
```python
dataset[42] # This will open a file from our machine. Read a small chunk, tokenize it, and return one training example
```

it will return
```python
(features, label)
```

This what we will do to inspect a specific something. 
And beside it, if our data contains `name`, this is an useless feature. that why dataset will help even here, because this is the process it will do:

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
Every time. Automatically.

And we have another saver. `__getitem__()`. This will allow us to support indexing and slicing even on custom objects.

Now imagine we have a file called `Airi_grades.csv`.

Now let's see if it is professional or no:

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

Cute? Yes.
Professional? Not really.
You just fed 20 GB worth of memory on your laptop in one shot. Dangerous.

But before it, let us use this idea to explain better ideas as `__len__()`, `__getitem__()`, and `Dataset`

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

Let us understand what this code does:

1. `def __len__(self): return len(self.data)` - Suppose our csv_file has 20 rows. We will use len(self.data) to count how many rows the file has. (Anyway, self.data is the variable that holds our file)
2. `def __getitem__(self, index)` - This will let us to use slices and indexing in our custom objects.
3. `return torch.from_numpy(self.data[index]).float()` - I will break it in more pieces:
	- `self.data[index]` - Because `self.data` is already a numpy array, we just get the exact row using standard numpy indexing `[index]`.
	- `.float()` - that will change the datatype to Float32, so it becomes a good standard for our beautiful Neural Network. 

But now let us look at another side...
A professional would want:

to feed the data on small pieces as:
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

As already mentioned, this what we call batches.

And here the dataset job is relatively simple. It has two question:
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

But we will use even batches, because loading all at the same time is bad, and training all of them is also bad, that why we will do them on batches, because if we decide to train the neural network with a for loop for each student... it would take 1,000,000 updates to finish it. That why we would never do that and use 
DataLoader.

```python
loader = DataLoader(
    dataset,
    batch_size=50,
    shuffle=True
)
```

Here we choose our dataset, gave the number of students per batch (There will be 50 students per batch), and `shuffle=True`, why? Because if we didn't turn it on the result would become this:

```
# shuffle=False
Epoch 1: Batch 0 -> [Item 0, Item 1, Item 2, Item 3]
Epoch 2: Batch 0 -> [Item 0, Item 1, Item 2, Item 3]  # Normal

# shuffle=True
Epoch 1: Batch 0 -> [Item 87, Item 12, Item 304, Item 5]
Epoch 2: Batch 0 -> [Item 42, Item 991, Item 3, Item 18] # Casual
```

And during training letting it turned off it bad as idea, because imagine we have 1000 of cat images and then 1000 of dog images.
If we let Shuffle off, then it would show firstly all the cat images , and only then the dog images. Bad training.

Let's see what happens in our DataLoader:

- `dataset` - in the first iteration (We have the batch set at 50), the DataLoader will say: "Dataset! GIVE ME 50 STUDENTS, NOW", the dataset will do: 
```python
__getitem__(0)

__getitem__(1)

__getitem__(2)

...

__getitem__(49)
```

   and instead of making 50 small tensors, it makes one big tensor - called `X_batch`, shape - `(50, <numbers_of_feature> -it will give as many columns as numbers of features we have)`
   The label will just become:
```python
[
1,
0,
1,
1
]
```
And it will be called `y_batch`.

I will call this section projects, because I like the idea, and I'll do a random code (There will be new ideas in them, so I will explain them right away! One of the new ideas is a whole important one, so let's see if you understand this concepts.
## PROJECTS OF CHAPTER 1 AND CHAPTER 2.

As I said, I will use niche ideas that we didn't talk about, yet I will try my hardest to explain them.

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

    if (epoch + 1) % 2 == 0:
        print(
            f"Epoch {epoch + 1}: Avg MSE = {avg_epoch_loss:.4f} | RMSE = {math.sqrt(avg_epoch_loss):.4f} points"
        )

"""
Output:

Epoch 1: Average MSE Loss = 2358.9833. The model was off by  48.5694 points
Epoch 3: Average MSE Loss = 96.8511. The model was off by  9.8413 points
Epoch 5: Average MSE Loss = 75.5645. The model was off by  8.6928 points
Epoch 7: Average MSE Loss = 73.7411. The model was off by  8.5873 points
.......
Epoch 25: Average MSE Loss = 72.4339. The model was off by  8.5108 points
Epoch 27: Average MSE Loss = 72.4810. The model was off by  8.5136 points
Epoch 29: Average MSE Loss = 72.9445. The model was off by  8.5408 points
```

Before we continue... THE WHOLE CODE IS TOTALLY FINE AND PEAK FICTION, IT IS JUST THAT I USED A STUPID FILE (That file had some pretty messed up data)

Let me explain all the parts you probably don't understand.

1. `self.network = torch.nn.Sequential` - before we start I can say: `self.network` is just a variable name, so don't think much about it.
   We use `torch.nn.Sequential`, because without it we would have been forced to define layers individually and pass the data thorough each of them step by step... as we did before:
```python
def __init__(self, in_feature):
	super()__init__()
	self.fc1 = nn.Linear(in_features, 32)
	self.fc2 = nn.Linear(32, 16)
	self.out = nn.Linear(16, 1)
	
def forward(self, x):
	x = torch.relu(self.fc1(x))
	x = torch.relu(self.fc2(x))
	x = self.out(x)
	return x
```

   Thanks to `torch.nn.Sequential` the output of the first layer becomes the input of the next layer.

2. `for epoch in range(30)` - I am sure you know what it means, but there is a reason why we put it so small instead of our classic 1000+ ultra pro max. Because now we think even about batch size... because 1 run upgrades 50 samples, we will look at out rows (In my case 1500), I will do: 1500/<batch_size> (In my case 50) = 30.
3. `model.train()` - You will always write it whenever you have to train the model, when we will test it we will write:
   `model.eval()`
4. `running_loss = 0.0` - When we compute `loss = criterion(predictions, y_batch)`, PyTorch gives you a PyTorch Tensor holding the loss value for that single 50-student batch. So to get the loss for every batches, we will reset it every new batch
5. ` loss.item()` - As we know, pytorch remembers... So of course... we strip away the computational graph and hold the loss
6. `running_loss += loss.item()` - We slowly accumulate all the losses of the batches, so  then we divide the loss by all the batches we have (30). So we get the average loss.

Since we will almost never write all by hand and do stuff as this, we will continue with the next topics (Be careful and repeat what you learned about pytorch! Because even so we will come back to it)


Now we will start with the pain... but these are just the basics we need before moving into a more complex topic (Newsflash: even this part will be hard).

Now, let's start with the first topic.

# Chapter 1. Non-Euclidean Geometry Prime

We'll start with the non-Euclidean prerequisites, and then move on to complex numbers and rotation groups.

## 1. What Is a Manifold?

When talking about manifolds, everybody would simply state:

"A manifold is a space that looks flat (local) when you zoom in closely, even though the whole space may be curved (global)."

Cute, but that doesn't provide much information for most people. That is why I will give some examples.

Imagine a 2D space:
```
              y
              ↑
              |
              |
              |       •
              |
──────────────┼──────────────→ x
              |
              |
```

That can be written as $`\mathbb{R}^2`$ (since it's 2D).

If you zoom into any point, it will look like a flat plane.

The distance is measured as:

```math
ds^2 = dx^2 + dy^2
```

This is why the shortest path from point A to point B is simply a straight line.

Sooo, before we say that flat = a line, I will say: "flat" — take a normal sheet of paper.

<img width="500" height="334" alt="image" src="https://github.com/user-attachments/assets/4c3eb02d-bb11-4093-be99-d478f083f825" />


If we draw two points on that sheet of paper, the shortest distance between point A and point B can be described by a straight line.

This is what flat means.

Now we will draw a small neighborhood around A:

```
        ┌───────────┐
        │           │
        │     • A   │
        │           │
        └───────────┘
```

Here you can move left, right, forward, or backward — nothing about the geometry itself changes. Basically, this is Euclidean geometry.

Now imagine taking a basketball. If we put a square sticker on it, we can see something clearly: the basketball is curved, but the sticker (if it's small enough) will look almost like an ordinary flat square, and the geometry inside that tiny region will be approximately like the geometry of a flat sheet.

The word "neighborhood" may seem simple, but it isn't. Imagine that we have a point:
```
            • P
```

Now imagine drawing a circle around it:
```
        .─────────.
      .'           '.
     /               \
    |       • P       |
     \               /
      '.           .'
        '─────────'
```

The circle around our point P is the neighborhood. It doesn't have to be a circle mathematically — I just gave you an idea of what a neighborhood is.

Now imagine our point sitting on the globe:

```
             ______
          .-'      '-.
        .'            '.
       /       • P      \
       \                /
        '.            .'
          '-.______.-'
```

Let's put a really small neighborhood around P:
```
             ______
          .-'      '-.
        .'       P    '.
       /       [•]      \
       \                /
        '.            .'
          '-.______.-'
```

If we make it smaller and smaller, its shape becomes increasingly similar to a piece of a plane. So it's Euclidean space now, right?

No, it is not — because a tiny piece of a sphere can behave like a piece of Euclidean space without the whole sphere being Euclidean space. Think about a long, curved road: if we zoom in on just the next 10 cm, it may look straight, but that doesn't mean the entire road is straight. We're simply looking at such a small section that the curvature is hard to notice.

Imagine you are in Rome. You look around and see:
```
        horizon

────────────────────────
```

Even if the ground beneath you looks flat, the Earth isn't flat at all — it is curved. And that's an important distinction:

```
VERY LOCAL VIEW:

       flat-ish
    ─────────────


GLOBAL VIEW:

          ______
       .-'      '-.
      /            \
      \            /
       '-.______.-'
```

So "local" means you're looking at a small neighborhood, while "global" means you're looking at the whole shape.

So the idea of the manifold is that every point has some sufficiently small neighborhood that behaves like ordinary Euclidean space.

So just because something looks like Euclidean space when you zoom in, it doesn't mean it is Euclidean space.

And there are two types of spaces to consider.

The manifold: we can place a point, place its neighborhood, and it looks like Euclidean space when zoomed in. You can draw lines, make perpendicular lines, make right angles — whatever you want — just like in Euclidean space.

<img width="1021" height="618" alt="image" src="https://github.com/user-attachments/assets/f22c4a75-68e3-4081-86d5-ece586858ea4" />


This is a manifold. As we can notice, we placed the point and the neighborhood, and if we zoom in, it looks like Euclidean space.

The non-manifold: one or more points can't be represented in Euclidean space — these are non-manifolds.

<img width="923" height="780" alt="image" src="https://github.com/user-attachments/assets/1e900635-0364-4480-8b19-c7a3dd8f4f6e" />


As we can see, if we place a point where the parts intersect and then draw its neighborhood, we won't be able to get Euclidean space out of it — so this is a non-manifold.

Why do we call it a manifold?
Because we want a space that we can represent mathematically using just a couple of coordinates (in this example, two).

For example, the surface of the Earth can be described using two coordinates:

```math
(\text{latitude}, \text{longitude})
```

Now imagine a sphere:

<img width="1024" height="739" alt="image" src="https://github.com/user-attachments/assets/2e6c8053-3b22-40c3-843c-2b98a88e7cc2" />


Even though it sits in 3D, we can describe its surface using just 2D coordinates:

```math
S^2 \subset \mathbb{R}^3
```

That's a really interesting concept.

> [!NOTE]
> Representing a curved surface with flat coordinates always comes at a cost — this is exactly why map projections of the Earth always distort something (area, angles, or distance). The manifold idea is what lets us describe a curved surface with flat coordinates *locally*, but it can never do so perfectly *globally*.

So, no matter what, the surface of a manifold can locally be represented in Euclidean space (though this may sometimes distort the bigger picture).

So a manifold is a shape where every point can be represented locally by Euclidean space, while the global structure may be abstract, complex, or curved.

Now we can move on.

## 2. Geodesics

A geodesic is the curved-space analogue of a straight line. That's the whole idea — but as you can imagine, math can spend 20 pages talking about a "line" without getting tired.

But what exactly is a straight line? Imagine we have two points and connect them with a straight line:
```
A •────────────────────• B
```

This line is the shortest path from point A to point B, because if we try to do:
```
A •──────────╮
             │
             │
             ╰──────────• B
```

this path will be longer than a straight line — which is why, in Euclidean geometry:

```math
\text{Straight line} = \text{Shortest distance between points}
```

This already gives us some intuition for geodesics.

Now let's go back to Earth. Airi (A) wants to visit Sheena (B), but since the world isn't flat, they can't travel in a straight line — because that would mean passing straight through the Earth.

```
       A
       ●
        \
         \
          \  ← through the Earth
           \
            ●
            B
```

They have to stay on the surface — this is why we introduce a new concept:

**Geodesic** — the shortest path from one point to another on a curved space.

Let's take the Earth once again — we need to find the local shortest path between two points.

But before we go all-in on the idea that:

```math
\text{Geodesic} = \text{shortest path}
```

> [!IMPORTANT]
> A geodesic is **not** always the globally shortest path — it's only guaranteed to be the *locally* shortest path: $`\text{geodesic} \approx \text{shortest local path}`$, but $`\text{geodesic} \neq \text{shortest global path}`$.

What do I mean by this? Let's take it one idea at a time — first, the "shortest local" part.

1. Geodesic ≈ the shortest local path:

Imagine points A and B are extremely close to each other on the surface. In that case, the geodesic is definitely the shortest path — because within such a small neighborhood (which behaves just like our familiar Euclidean space), a straight line is still the most direct route from one point to another.

2. Geodesic ≠ the shortest global path:

On a curved space, if you keep walking straight along a geodesic, you might end up looping back around or taking the "long way." There's a simple rule here: the globally shortest path must be a geodesic, but not every geodesic is the globally shortest path.

On a sphere, a geodesic is a great circle. So let's say we need to go from Ecuador (A) to Kenya (B), which are roughly 12,000 km apart.

- **Local shortest path**: walk $`12{,}000\text{ km}`$ directly east along the equator.
- **The other geodesic**: start from point A and walk straight *west* along the equator instead — after $`28{,}000\text{ km}`$ you'll still end up at point B. This is a perfectly valid geodesic, even though it's $`16{,}000\text{ km}`$ longer than the direct path.

Both are geodesics. But as we said, we can't take a truly straight line, since that would mean passing through the Earth's core. And the rule holds: there can be multiple geodesics between two points, but not all of them are the shortest path.

<img width="642" height="624" alt="image" src="https://github.com/user-attachments/assets/53f14a13-8117-474a-9b92-bfaad1e16d33" />


This is what the geodesics look like on a sphere (which is still a manifold). We have two geodesics — one longer and one shorter — even though each one simply followed a single direction (one went west, the other east).

What a geodesic looks like depends on the space:

1. In Euclidean space — the geodesic is a straight line.
2. In spherical space — the geodesic is a great circle.
3. In hyperbolic space — geodesics are curves whose shape depends on the specific model being used.

Now let's get into the formulas — putting them off any longer wouldn't be funny.

So the general metric arc-length formula is:

```math
L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt
```

There's only one correct reaction to that: *"Nah, I'm skipping this."* And that's the right one! But not so fast — because if you do skip it, what will you do when we get to:

```math
\text{Einstein}(x_1, \dots, x_N; w) = \frac{\sum_{i=1}^N w_i \gamma_i x_i}{\left(1 + \sqrt{1 - c \left\Vert \sum_{i=1}^N w_i \gamma_i x_i \right\Vert^2}\right) \sum_{i=1}^N w_i \gamma_i} \quad \text{where } \gamma_i = \frac{1}{\sqrt{1 - c\Vert x_i\Vert^2}}
```

So we'd rather start slow and get comfortable with the easy formulas before heading toward that.

Now we will start with the arc-length again. We will break it down into components:

1. $`\gamma(t)`$ — this is the position vector. For example, if $`\gamma(t) = (t^2, 3t)`$, then after two seconds the position becomes $`\gamma(2) = (4, 6)`$ — in 2D space, we moved 4 units right and 6 units up.

$`\gamma(t)`$ tells us where we are at every moment. In a simple 2D space, it would look like:

```math
\gamma(t) = (x(t), y(t))
```

But for now, let's say:

```math
\gamma(t) = (t, t^2)
```

That means:
```
t = 0  →  (0,0)
t = 1  →  (1,1)
t = 2  →  (2,4)
...
```

So we can understand that $`\gamma(t)`$ reveals our position in space over time by tracing out a path.

And here we will see this idea a lot:

```math
a \leq t \leq b
```

Here, $`a`$ is the time we start and $`b`$ is the time our journey ends.

- $`\dot{\gamma}(t)`$ — this is basically $`\frac{d\gamma}{dt}`$. So if our gamma is:

```math
\gamma(t) = (x(t), y(t))
```

then it becomes:

```math
\dot{\gamma}(t) = (x'(t), y'(t))
```

This is what we call our velocity vector.

Let's take our earlier example:

```math
\gamma(t) = (t, t^2)
```

Its derivative becomes:

```math
\dot{\gamma}(t) = (1, 2t)
```

and this is our velocity vector — one attached at each point along the path.

And now, let's talk about the metric $`g`$.

- $`g`$ — for this example, we'll stay in Euclidean space.

Let's take a vector:

```math
v = (3,4)
```

Now we'll find its magnitude:

```math
\|v\| = \sqrt{3^2 + 4^2} = 5
```

But now we will rewrite it using the identity matrix ($`I`$).
So:

```math
v^TIv = \begin{bmatrix} 3 & 4 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\ 4 \end{bmatrix}
```

which gives us:

```math
v^TIv = 25
```

Therefore:

```math
\sqrt{v^TIv} = 5
```

So the Euclidean geometry metric is playing the role of the dot product.

So $`g_{\gamma(t)}`$ is the metric at the current position.
But we have to remember that $`g_{\gamma(t)}`$ changes based on our position — we'll discuss why later.

In the end, I wanted to add that a geodesic isn't only about drawing a straight line, because even if we do, sometimes it may not be a geodesic — it always depends on how flat the space looks locally (first 6 minutes of https://youtu.be/m6WY6VtPYrk?is=JgUE4t21J_XVOPCH).

## 3. Curvature

Now we will ask ourselves... what happens when we have two nearby geodesics? This will lead us to the sign of curvature.

Imagine that two people walk in a straight line next to each other.
```
A ─────────────────────→
B ─────────────────────→
```

What will happen? They will never intersect; their distance will stay constant if they start 1 meter from each other ($`d(t)=1`$). This is why it has no curvature:

```math
K = 0
```

Therefore, this is flat space, and it means that nearby geodesics neither systematically converge nor diverge.

What is $`d(t)`$? Imagine we have two walkers; at time $`t`$, we measure the distance between them, and that distance can change as time passes.
For example:
$`d(0) = 1`$ (they start 1 meter apart from each other)
$`d(2) = 0.8`$ (after 2 seconds, they are 0.8 meters apart)

So $`d(t)`$ is the separation distance between two geodesics.

Now we have another hero — which we will call Mimi, $`\dot{d}(t)`$ — which is exactly:

```math
\dot{d}(t) = \frac{dd}{dt}
```

In words, this means: how quickly does the distance between the walkers change?

Now we have the cooler Mimi — $`\ddot{d}`$, which is exactly:

```math
\ddot{d} = \frac{d^2d}{dt^2}
```

This basically means: how quickly does the *rate* of separation change?

So, basically, we have these steps:
$`d(t)`$ — the distance between the walkers (geodesics)
$`\dot{d}(t)`$ — how fast the distance between the walkers changes
$`\ddot{d}(t)`$ — how fast that rate of change itself changes

Now let us suppose that

```math
d(t) = 10 - t^2
```

its first derivative (our $`\dot{d}(t)`$):

```math
\dot{d}(t) = -2t
```

and its second derivative (our $`\ddot{d}(t)`$):

```math
\ddot{d}(t) = -2
```

So the separation isn't merely decreasing — its rate of decrease is increasing.

For example:
At $`t=1`$:

```math
\dot{d}(1) = -2
```

At $`t=2`$:

```math
\dot{d}(2) = -4
```

The walkers are getting closer faster and faster.
This is what $`\ddot{d}(t) < 0`$ is telling us.

But what is $`K`$? Its symbol is really important — this represents curvature.

This is what it tells us:
```
K > 0  → sphere-like
K = 0  → flat
K < 0  → hyperbolic
```

Now we combine these two ideas:

```math
Kd(t) = \text{curvature} \times \text{current separation}
```

For example, suppose:

```math
K = 2
```

and:

```math
d(t) = 3
```

Then:

```math
Kd(t) = 2(3) = 6
```

So in the end, the sign of $`K`$ determines the direction of the effect.

Now we will assemble the equation:

```math
\ddot{d}(t) + Kd(t) = 0
```

In words:

```math
\text{relative acceleration} + \text{curvature} \times \text{separation} = 0
```

Simplified even further:
- `relative acceleration` — how fast the walkers' rate of separation changes.
- `curvature` — the physical property of the space that forces nearby parallel geodesics to bend together (only if $`K > 0`$), pull apart (only if $`K < 0`$), or stay parallel (only if $`K = 0`$).
- `separation` — the distance between the two walkers.

And now the final formula we will use is:

```math
\ddot{d}(t) = -Kd(t)
```

We have two possible scenarios (plus the trivial $`K=0`$ case):

---
**Scenario №1**: $`K = 0`$

In this case the result will be 0, no matter what.

```math
\ddot{d}(t) = -(0) \times d(t)
```

```math
\ddot{d}(t) = 0
```

This means the acceleration of the separation is zero — the lines stay parallel.
```
A ─────────────────>

B ─────────────────>
```

This is flat space.

---
**Scenario №2**: $`K > 0`$

In this case, the result will be negative, no matter what.

Let us say $`K = 1`$ and $`d(t) = 2`$:

```math
\ddot{d}(t) = -(1) \times 2
```

```math
\ddot{d}(t) = -2
```

Negative relative acceleration means the separation is being pushed toward smaller values.

```math
K > 0 \Rightarrow \text{geodesics tend to converge}
```

This is sphere-like.

---
**Scenario №3**: $`K < 0`$

In this case, the result will be positive, no matter what.

So let us say $`K = -1`$ and $`d(t) = 2`$:

```math
\ddot{d}(t) = -(-1) \times 2
```

```math
\ddot{d}(t) = 2
```

The walkers are being driven apart from each other.

```math
K < 0 \Rightarrow \text{geodesics tend to diverge}
```

This is hyperbolic behavior.

---

Now we will continue with another topic.

## 4. Why Does Negative Curvature Fit Hierarchies?

This is what we will work with a lot... so this question is really important.

Firstly we will start with a graph:

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/d25797b3-3d36-4e11-8973-2fc23c2e4b2a" />


Now we imagine that each node has two nodes more... and so on:

| Depth (d) | Nodes |
| --------: | ----: |
|         0 |     1 |
|         1 |     2 |
|         2 |     4 |
|         3 |     8 |
|         4 |    16 |
|         5 |    32 |

What does this look like? Each new depth, the nodes multiply by 2. That is why we get:

```math
N(d) = 2^d
```

(N = nodes. This means the number of nodes depends on the depth.)
It has exponential growth.
This is why the more depth we add, the more nodes we get.

At depth 10:

```math
2^{10} = 1024
```

At depth 20:

```math
2^{20} = 1048576
```

This means the tree becomes enormous very quickly.

But where are those nodes situated geometrically?
Imagine placing the root in the middle (the first node):
```
                    •
                 •  •  •
              • • •   • • • 
```

Now we have $`d=1`$, $`d=2`$, $`d=3`$... As the depth grows, we need more space — because the deeper we go, the more nodes we need to place around the root.

So... how much space does Euclidean geometry actually give us?

Let's draw a circle around the root:

```math
C(r) = 2\pi r
```

So if we double the radius, we get:

```math
C(2r) = 4\pi r
```

The amount of available boundary grows only *linearly* with $`r`$.

And what happens if we compare the two? Say we add +1 unit of radius for every extra depth... but that's the problem: one grows linearly, the other exponentially. There's a serious mismatch.

Why? Let's say we have 10 levels of depth:

```math
N(d) = 2^d
```

so:

```math
N(10) = 1024
```

Now compare that to the linear growth of the boundary:

```math
C(10) = 62.8
```

So we're trying to place 1,024 objects around a circumference of only about 63?

What about 20 levels of depth?

```math
2^{20} = 1048576
```

while:

```math
2\pi(20) \approx 126
```

Now the difference is ridiculous.

So in the end, we can't fit too many objects into Euclidean space without either:

1. Putting related things too far apart from each other,
2. Putting unrelated things too close together, or
3. Requiring enormous Euclidean distances.

What about placing them in a different kind of space instead — something like hyperbolic space, where the boundary grows like:

```math
C(r) = 2\pi \sinh r
```

What is $`\sinh r`$? By definition:

```math
\sinh r = \frac{e^r - e^{-r}}{2}
```

As $`r`$ grows, $`e^{-r}`$ shrinks toward zero, so it barely matters — which is why, for large $`r`$, we can approximate:

```math
\sinh r \approx \frac{e^r}{2}
```

so:

```math
C(r) \approx 2\pi \cdot \frac{e^r}{2}
```

and now for a bit of high-school-level simplifying (cancel the 2s):

```math
C(r) \approx \pi e^r
```

> [!NOTE]
> $`\sinh r = \frac{e^r - e^{-r}}{2}`$ is the *exact* definition. The step to $`\sinh r \approx \frac{e^r}{2}`$ is only an *approximation* — one that gets more accurate as $`r`$ grows and $`e^{-r}`$ becomes negligible.

So where's the difference? Let's compare:

- Euclidean space: $`C(r) \sim r`$
- Hyperbolic space: $`C(r) \sim e^r`$
- Tree: $`N(d) \sim b^d`$

The hyperbolic boundary and the tree's node count are *both* exponential — they're related!

Suppose the tree has branching factor $`b`$. Then:

```math
N(d) = b^d
```

Hyperbolic space grows approximately like:

```math
C(r) \sim e^r
```

Notice that, because $`e^{\ln x} = x`$, we can rewrite $`b^d`$ as:

```math
b^d = e^{\ln(b^d)}
```

Using the logarithm power rule $`\ln(x^y) = y \ln x`$:

```math
e^{\ln(b^d)} = e^{d \ln b}
```

And applying the exponent rule $`(x^m)^n = x^{mn}`$:

```math
e^{d \ln b} = (e^{\ln b})^d = e^{(\ln b)d}
```

So:

```math
\boxed{N(d) = e^{(\ln b)d}}
```

> [!IMPORTANT]
> This is the key insight: a tree's node count and hyperbolic space's boundary both grow *exponentially*, with the same underlying shape.
>
> - A hierarchy says: "the farther I go from the root, the more branches I need."
> - Hyperbolic geometry says: "the farther I go from the center, the more space becomes available."
>
> The two match perfectly — which is exactly why hyperbolic space is such a natural home for embedding hierarchies (trees).

It looks something like this:
```
                    boundary
             ___________________
          .-'                     '-.
        .'                           '.
       /                               \
      |             ROOT                |
      |               •                 |
      |            /  |  \              |
      |          •    •    •            |
      |        /|\   /|\   /|\          |
      |      • • • • • • • • •         |
       \                               /
        '.                           .'
          '-._____________________.-'
```

This will help us later with Poincaré embeddings.

Now let's move on to complex numbers and rotation groups.

# Chapter 2. Complex Numbers and Rotation Groups

This topic is small, but it's really important — without it, we'll struggle with everything that comes after.

## 1. What Are Complex Numbers?

What even are complex numbers ($`\mathbb{C}`$)?

We already know what real numbers ($`\mathbb{R}`$) are — for example:

```math
5, \quad -3, \quad 0, \quad \frac{1}{2}, \quad \sqrt{2}
```

Imagine I give you this equation:

```math
x^2 = 4
```

What are the solutions? Easy — at a glance we can see that:

```math
2^2 = 4 \quad \text{and} \quad (-2)^2 = 4
```

So $`x`$ can have two values:

```math
2 \quad \text{or} \quad -2
```

That's it, simple.

But what if I give you this equation instead:

```math
x^2 = -1
```

Hmm... let's try some real numbers:

Positive: $`2^2 = 4 \qquad 1^2 = 1 \qquad 0^2 = 0`$

Negative: $`(-1)^2 = 1 \qquad (-2)^2 = 4`$

A bit tricky, right? Notice something? We can never get a negative result:

```math
x^2 \geq 0
```

So... our equation $`x^2 = -1`$ has no real solution. Can we just pack up and go home now? Nope.

Because we can still invent a new number whose square is $`-1`$.

This number is:

```math
i
```

defined by:

```math
i^2 = -1
```

> [!NOTE]
> "Imaginary" doesn't mean "fake." $`i`$ is simply a number that isn't a real number — it was introduced so that equations like $`x^2 = -1`$ can have a solution. It's not positive, negative, or zero. It's simply imaginary.

The solutions to:

```math
x^2 = -1
```

are:

```math
i \quad \text{or} \quad -i
```

because:

```math
(-i)^2 = (-i)(-i) = i^2 = -1
```

But we'll also run into things like:

```math
x = 3 + 2i
```

This leads us straight to the definition of a complex number. The general form of a complex number is:

```math
z = a + bi
```

Let's break down each part:

- $`z`$ — the complex number itself.
- $`a`$ — a real number, called the **real part** of the complex number. We'll often write this as $`Re(z) = a`$, meaning "the real part of $`z`$ is $`a`$."
- $`b`$ — also a real number, but used like this: $`b \times i = bi`$

Let's decode the earlier example:

```math
z = 3 + 2i
```

The real part here is:

```math
Re(z) = 3
```

and the imaginary part is:

```math
Im(z) = 2
```

> [!WARNING]
> The imaginary part is the *coefficient* $`2`$, not the whole term $`2i`$.

But what if we run into something like:

```math
z = 5i
```

We write this as:

```math
z = 0 + 5i
```

So the real part is $`0`$ and the imaginary part is $`5`$. The same idea applies if we only have a real part with no imaginary part.

So we can see that:

```math
\mathbb{R} \subset \mathbb{C}
```

because the real numbers are simply the case where the imaginary part $`b`$ is zero.

But why do we even care about them? What does

```math
z = a + bi
```

really tell us? Here's the surprising part: we can associate it with the pair

```math
(a, b)
```

This means we can draw complex numbers as points on a plane.

We simply set up two axes: one "Real" axis and one "Imaginary" axis. The real axis is horizontal, and the imaginary axis is vertical:

```
          imaginary axis
                ↑
                |
            3i  |
                |
            2i  |         •  3+2i
                |         (3,2)
            i   |
----------------+----------------→ real axis
                | 1   2   3
           -i   |
                |
```

We can't order complex numbers the way we order real numbers, like:

```math
\cdots < -2 < -1 < 0 < 1 < 2 < \cdots
```

because complex numbers don't live on a single one-dimensional line — they live on a plane.

So instead of asking:
- "Is $`i`$ positive or negative?"

we ask:
- "Where is $`i`$ in the complex plane?"

## 2. Modulus and Argument

We already know a complex number can be viewed as a point on the plane:

```math
(a, b)
```

But now we have new questions:

- How far is $`z`$ from the origin?
- In which direction is $`z`$ pointing?

These two questions introduce two new concepts: **Modulus** and **Argument**.

Let's start with the modulus.

### 1. Modulus

Suppose we have the complex number:

```math
z = 3 + 4i
```

Let's place it on the plane.

<img width="532" height="551" alt="image" src="https://github.com/user-attachments/assets/4b1147cc-e806-4e70-9004-08cddd0fb22a" />


The origin sits at $`(0, 0)`$, and we want the distance from $`(0,0)`$ to $`(3, 4)`$ — which is exactly what the Pythagorean theorem gives us:

```math
|z| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5
```

This is simply the distance from the origin to $`z`$ — the **modulus**, or length of the vector (magnitude).

Now we can continue with the argument.

### 2. Argument

Now we know the magnitude is $`5`$ (5 units away) — but where does it *point*? To answer that, we look at the angle between the positive real axis and the line from the origin to $`z`$.

<img width="532" height="554" alt="image" src="https://github.com/user-attachments/assets/3737a33f-cbdd-4466-a52c-711d2ae1cfc0" />


We write this as:

```math
arg(z) = \theta
```

This is the angle of $`z`$, measured from the positive real axis. So for:

```math
z = 3 + 4i
```

we use:

```math
\tan\theta = \frac{\text{opposite}}{\text{adjacent}} = \frac{4}{3}
```

and with a bit of trigonometry:

```math
\theta = \arctan\left(\frac{4}{3}\right) \approx 53.13^\circ
```

But why do we even care about $`|z|`$ and $`arg(z)`$? Because together, they describe exactly where the point is — using **polar coordinates**:

```math
(r, \theta)
```

What about:

```math
z = i
```

That's the same as:

```math
z = 0 + 1i
```

so the point is simply $`(0, 1)`$. Let's find its modulus and argument:

Modulus:

```math
|z| = \sqrt{0^2 + 1^2} = 1
```

Argument:

```math
arg(i) = 90^\circ
```

because it points straight up from the origin. Or in radians:

```math
arg(i) = \frac{\pi}{2}
```

There are 4 special cases worth memorizing, since even though their magnitude is the same, their argument is not:

| $`z`$  | $`\vert z \vert`$ |  $`arg(z)`$   |
| :--: | :-------------: | :---------: |
| $`1`$  |       $`1`$       |  $`0^\circ`$  |
| $`i`$  |       $`1`$       | $`90^\circ`$  |
| $`-1`$ |       $`1`$       | $`180^\circ`$ |
| $`-i`$ |       $`1`$       | $`-90^\circ`$ |

Once you reach $`360^\circ`$, you're pointing the same direction as $`0^\circ`$:

| Angle ($`\theta`$) |     Direction     |
| :--------------: | :---------------: |
|    $`0^\circ`$     |    Point right    |
|    $`90^\circ`$    |     Point up      |
|   $`180^\circ`$    |    Point left     |
|   $`270^\circ`$    |    Point down     |
|   $`360^\circ`$    | Point right again |

This is how angle-wrapping works: once you rotate a full $`360^\circ`$, you end up pointing in exactly the same direction again.

We already have a taste of the polar form — now let's go deeper.

## 3. Polar Form & Euler's Formula

Now that we know how to get the modulus and argument, we can go from:

```math
(a, b) \rightarrow (r, \theta)
```

and also back the other way:

```math
a = r\cos\theta \quad \text{and} \quad b = r\sin\theta
```

So we get:

```math
z = r(\cos\theta + i\sin \theta)
```

But now we'll write it as:

```math
z = re^{i\theta}
```

Why? Let's break it down piece by piece.

What does $`e`$ mean? As we already know, $`e`$ is a special number that shows up in exponential growth and decay. For example:

```math
e^0 = 1, \quad e^1 = e, \quad e^2 = e \times e
```

So $`e^x`$ is just an exponential function — nothing strange.

We already know what $`e^x`$ means when $`x`$ is a real number. But what if we have:

```math
e^{i\theta}
```

Now the exponent is imaginary — but this turns out to have a beautiful geometric meaning. Let's see why.

This is **Euler's formula**:

```math
e^{i\theta} = \cos\theta + i\sin\theta
```

Let's decode it:
- $`e`$ — Euler's number ($`\approx 2.71828`$)
- $`i`$ — our imaginary unit ($`i^2 = -1`$)
- $`\theta`$ — the angle
- $`\cos\theta`$ — the horizontal coordinate of a point on the unit circle
- $`\sin\theta`$ — the vertical coordinate

Let's try this formula with two examples.

If $`\theta = 0`$:

```math
e^{i0} = \cos0 + i\sin0
```

Since $`\cos 0 = 1`$ and $`\sin 0 = 0`$:

```math
e^{i0} = 1 + i(0) = 1
```

Geometrically, $`1`$ corresponds to the point $`(1, 0)`$ — the rightmost point on the unit circle.

Now let's try $`\theta = \frac{\pi}{2}`$:

```math
e^{i \pi/2} = \cos\frac{\pi}{2} + i\sin\frac{\pi}{2} = 0 + i(1) = i
```

Geometrically, that's a rotation of $`90^\circ`$.

So what is $`e^{i\theta}`$, really? Since $`e^{i\theta} = \cos\theta + i\sin\theta`$, we always get $`|e^{i\theta}| = 1`$ — meaning it's always exactly one unit away from the origin. Multiplying by $`e^{i\theta}`$ only ever changes the *direction*, never the distance.

```
                    i
                    ↑
                    |
              •     |     •
                    |
        -1 ←────────+────────→ 1
                    |
              •     |     •
                    |
                    ↓
                   -i
```

Let's return to the polar form we found earlier:

```math
z = r(\cos\theta + i\sin\theta)
```

Now that we know Euler's formula:

```math
e^{i\theta} = \cos\theta + i\sin\theta
```

we can substitute directly:

```math
z = re^{i\theta}
```

The polar form has two important pieces:

$`r`$ — the magnitude ($`r = |z|`$)
$`e^{i\theta}`$ — the direction, at angle $`\theta`$

That's what the polar form tells us.

Let's put this into practice with the complex number we've been using:

```math
z = 3+ 4i
```

We already have everything we need:

```math
|z| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5
```

```math
\theta = \arctan\left(\frac43\right) \approx 53.13^\circ
```

Now we just plug them in:

```math
z = 5e^{i\,53.13^\circ}
```

Now let's tweak the formula slightly. Write it as:

```math
z = re^{i\phi}
```

where:

$`r`$ — its current distance from the origin
$`\phi`$ — its current angle

Now let's multiply it by $`e^{i\theta}`$:

```math
e^{i\theta} \times z = e^{i\theta} \cdot re^{i\phi}
```

Rearranging:

```math
re^{i\theta}e^{i\phi}
```

and using the exponential rule ($`e^ae^b = e^{a + b}`$):

```math
re^{i(\theta + \phi)}
```

So we get:

```math
e^{i\theta}z = re^{i(\theta + \phi)}
```

Either way, the distance $`r`$ stays exactly the same. The only thing that changes is the angle — we started with $`\phi`$ and end up with $`\phi + \theta`$.

Let's walk through an example so this clicks.

Suppose $`z = 2`$, i.e. the point $`(2, 0)`$. In polar form:

```math
z = 2e^{i0}
```

Now let's multiply it by $`e^{i\pi/2}`$ to rotate it:

```math
e^{i\pi/2}z = e^{i\pi/2}(2e^{i0}) = 2e^{i\pi/2}
```

And since:

```math
e^{i\frac{\pi}{2}} = \cos\left(\frac{\pi}{2}\right) + i\sin\left(\frac{\pi}{2}\right) = 0 + i = i
```

we go from:

```math
(2, 0) \rightarrow (0,2)
```

That's how it works.

Now that we've made it through this mess, let's dive into another one — because clearly, all of this was too easy...

## 4. Möbius Transformation Basics

Let's not hit you with the full formula right away — we'll build up to it with a few smaller ideas first. We already know what a transformation is, but just to be safe, let's go over it again.

A transformation is simply a rule that takes an input and returns an output.
For example:

```math
f(x) = x + 2
```

This takes our variable $`x`$ and adds $`2`$ to it. We can do the same thing with complex numbers — but instead of:

```math
x \rightarrow f(x)
```

we write:

```math
z \rightarrow f(z)
```

which simply means taking a point $`z`$ and moving it to another point $`f(z)`$.

Before we get to the scary part, let's look at some simple transformations. Imagine:

```math
f(z) = z + b
```

where $`b`$ is a complex number. So if $`z = 2 + 3i`$ and $`b = 1 + i`$:

```math
f(z) = (2 + 3i) + (1 + i) = 3 + 4i
```

The point moved: $`(2, 3) \rightarrow (3, 4)`$. That's a transformation.

What about scaling and rotating at the same time? We already learned this powerful idea:

```math
e^{i\theta}z
```

which simply means "rotate $`z`$ by $`\theta`$." We can also multiply by a positive real number to scale it. For example:

```math
2z
```

doubles the distance from the origin.

For example, suppose:

```math
a = 2e^{i\pi/2}
```

If we multiply it by our current complex number $`z`$:

```math
az = 2e^{i\pi/2}z
```

this means $`z`$ gets rotated by $`90^\circ`$ *and* scaled by $`2`$.

Now let's combine both transformations ($`az`$ and $`b`$):

```math
az + b
```

Breaking it down piece by piece:
- $`z`$ — our original point
- $`a`$ — a complex number that scales and rotates
- $`b`$ — a complex number that translates (shifts the position)

So in the end, $`az + b`$ scales/rotates $`z`$, then translates it.

Now let's introduce a second part: division.

```math
f(z) = \frac{az + b}{cz + d}
```

This is the **Möbius transformation**. Let's decode every symbol:
- $`z`$ — our starting point in the complex plane.
- $`a, b, c, d`$ — four complex constants. Each one is a complex number, so each can have both a real and an imaginary part.

But why divide at all? Because it lets us turn straight lines into circles, turn the plane inside out, and more (we'll dig into this properly when we cover the full topic later).

> [!CAUTION]
> There's a restriction we must always follow: $`ad - bc \neq 0`$. If $`ad - bc = 0`$, the transformation collapses into something that isn't a proper Möbius transformation at all. This is called the **non-degeneracy condition**. For now, this simple intuition is enough.

Let's try an example:

```math
a=1 \quad b=2 \quad c=0 \quad d=1
```

so:

```math
f(z) = \frac{1z + 2}{0z + 1} = z + 2
```

If $`z = 3 + i`$:

```math
f(z) = 5 + i
```

so:

```math
(3, 1) \rightarrow (5, 1)
```

That was a simple one. Now let's try rotating by $`270^\circ`$ and shrinking to half size.

Let's use $`z = 5 + 3i`$.

To rotate by $`270^\circ`$ (or $`\frac{3\pi}{2}`$ radians), we use Euler's formula:

```math
e^{i \cdot 270^\circ} = \cos(270^\circ) + i\sin(270^\circ) = 0 + i(-1) = -i
```

So we multiply by $`-i`$ to rotate by $`270^\circ`$, and by $`\frac{1}{2}`$ to shrink to half size. Combining both into $`A = -\frac{1}{2}i`$ and applying it:

```math
w_1 = A \cdot z = -\frac{1}{2}i (5 + 3i) = -\frac{5}{2}i - \frac{3}{2}i^2 = \frac{3}{2} - \frac{5}{2}i
```

And that's our result!

Now we can get back to the topics we were supposed to cover all along — for now, we've only learned the basics, and even though they were hard, they were useful.

The next topic will be Minkowski space coordinate geometry and some others... it already sounds frightening, but I'll make it easy to follow.

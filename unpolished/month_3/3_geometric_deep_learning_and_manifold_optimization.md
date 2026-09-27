
Here I will talk about the same ideas, we will just go deeper than before. So as already said... Don't cry if you see myself repeating for the 15th time.

# Chapter 1. Space Definitions (types & invariants)

A small content table before starting:

1. Riemannian Manifold Definition
2. Minkowski Space Coordinates
3. Hyperboloid Domain
4. Poincare Ball Domain
5. Diffeomorphism Operators

## 1. Riemannian Manifold Deiftion

Eventually we will want to represent

- words 
- concepts 
- skills
- nodes in a hierarchy 
- and many others

as a point in the hyperbolic space

For example:
```
                 Mathematics
                /           \
          Algebra           Geometry
          /    \             /    \
       Linear  Abstract   Euclidean  ...
```

We will want an embedding where this hierarchy naturally fits into the geometry, so we will eventually have:
```
                 Mathematics
                      ●
                    /   \
                   /     \
              Algebra   Geometry
               ●            ●
             /   \        /   \
            ●     ●      ●     ●
```

But we can't actually just slam them there and think that we will do normal vector arithmetic.
Because hyperbolic space is not ordinary Euclidean space.

That means operations that are perfectly valid in ordinary $\mathbb R^n$ (This is the standard notation for a n-dimensional Euclidean space, where $\mathbb{R}$ is/are the real number/s and $^n$ is the number of dimensions. In poor words it would mean: "How many real numbers I need to specify a single point?") can produce invalid points in our hyperbolic space.

While we can talk about a normal manifold, we will talk about the Riemannian manifold.

But what is the difference?

Imagine that a manifold tells us about the shape/structure of the space. But imagine we have two points on it...
You can not do any geometry on a normal manifold, yeah, you can use topology to track paths, intersections, and so on, but we care about the points.

But a Riemannian manifold give us a complete set, it tells us even how to measure things locally. Especially, in the set we have even a metric. Which will help us with:

- lengths,
- angles,
- local distances,
- therefore eventually geodesics.

In a notation it looks like:
$$(M,g)$$
Where:
- $M$ = the manifold
- $g$ = the Riemannian metric

Let us bring back the Minkowski space. We understand that the Minkowski space is basically:
$$\mathbb{R}^{d + 1}$$
For those who don't understand it, I will make it clear.

We have to understand that:
$$\text{Dimension of the object} ≠ \text{dimension of the space containing it}$$
Imagine a sphere, we represent it with three coordinates - (x,y,z). it sits in a 3D space ($\mathbb{R}^3$). But if you're standing on Earth's surface, how many independent directions can you move? 

We can move just:
- north/south
- east/west

but we can't move through the Earth while remaining on the surface.
The Earth's surface is a 2-dimensional manifold embedded inside 3-dimensional space.

This is the whole story.

So, I wanted to say that the Minkowski space is basically $\mathbb{R}^{d + 1}$ with a special inner product. But our hyperbolic manifold is just a particular subset of it. 
For a curvature parameter ($c > 0$) our hyperboloid will satisfy:
$$\langle x, x \rangle _M = -\frac{1}c$$

If we don't understand what $c$ is I will explain:

The $c$ is basically the curvature of the manifold itself.
We can get $c$ by dividing the squared radius ($r^2$) by 1.
$$\frac{1}{r^2}$$
We need to understand what $r$ is:

High curvature:
- If a ball is massive (huge radius $R$), its surface looks almost completely flat when we zoom in. 

Low curvature:
- If a ball is tiny (tiny radius $R$), it curves away sharply in every direction. High curvature.

As the radius gets bigger, the curvature gets smaller.

I will give two examples:

---
Big radius: A "Lazy," Mildly Curved Saddle
Let say that we want our hyperbolic space to be gently curved, so we give it a large radius, say $R = 2$.

$$c = \frac{1}{R^2} = \frac{1}{2^2} = \mathbf{0.25}$$ (positive, because $c > 0$).

Now we look at our equation $-\frac{1}c$
$$-\frac{1}{0.25} = \mathbf{-4}$$
Our equation becomes: $\langle x, x \rangle_M = \mathbf{-4}$.

---
Low radius: A Tightly Pinched, Aggressive Saddle
Now suppose we want a hyper-warped space where distances shrink rapidly. We give it a tiny radius, say $R = 0.5$.

$$c = \frac{1}{R^2} = \frac{1}{0.5^2} = \mathbf{4}$$ (still positive, $c > 0$).

Now we look at our equation $-\frac{1}c$:
$$-\frac{1}{4} = \mathbf{-0.25}$$
Our equation becomes: $\langle x, x \rangle_M = \mathbf{-0.25}$.

---

So now we understand why we want a curvature that is positive, because if it was negative it would turn our hyperboloid in a sphere. We want a hyperboloid, not a sphere.

Now we continue with the idea of:
$$\langle x, x \rangle _M = -\frac{1}c$$
This is a tike-like sheet restriction.

```
Minkowski space
     ↓
all possible (d+1)-coordinate vectors
     ↓
apply hyperboloid constraint
     ↓
valid hyperbolic points
```

This is going to be one of the most important invariants we are going to implement in this plan.

Now we may think... What is an invariant?
The definition is really easy - an invariant is simply something that must remain true. For our hyperboloid, the invariant that must remain true for every point is:
$$\langle x, x \rangle _M = -\frac{1}c$$

We will do something as:
```
valid point
     ↓
     x
     │
     │ operation
     ↓
  new point
     │
     ├── satisfies invariant → ✅
     │
     └── violates invariant → ❌
```

Now we will do some Julia implementation... I will not add data for now, just the skeleton.

```julia
abstract type AbstractManifold{T} end

struct Hyperboloid{d, T} <: AbstractManifold{T}
	c::T
end

struct PoincareBall{d, T} <: AbstractManifold{T}
	c::T
end
```

We probably know what it means, since we learned about it in the julia basics module... but if we forgot, I will explain it again.

The abstract type simply means: "Create a type that represent a general category, but don't create objects of this type directly".

For example, in our case we did:
```
AbstractManifold
       │
       ├── Hyperboloid
       │
       └── PoincareBall
```

The AbstractManifold is the parent category, while the Hyperboloid and PoincareBall are the concrete spaces.

The `{T}` is the parametric type... if we take some steps back we may remember that we use a parametric type to don't hardcode the type of everything with `Float64`. This is why we use `{T}`, so it can choose reliably and quickly its own  type.

Now, we maybe noticed even `Hyperboloid{d, T}` which is simply our Lorentz model. But we wrote even `{d}`, what is it?

`{d}` is simply the dimension of the hyperbolic manifold (The independent directions we can move on. For example in a 2D dimension we can move left/right and up/down)

Later we will see how this is related to the $d$ and $d + 1$ idea:
```
Hyperbolic dimension       Coordinates
       d = 2              3 coordinates
       d = 3              4 coordinates
       d = 4              5 coordinates
```

This means that if we choose `d = 2`, the points on our hyperboloid will look like `(x₀, x₁, x₂)`. But don't really worry about it right now, because I am going to make a whole lesson on this.

The second number `{d, T  <--- This lad}`, this is simply as already said; our parametric type, so instead of robbing our freedom of choice with slapping there something as: `(2, Float32)`, we will let the `T`.

What about the `<:` in the `Hyperboloid{d, T} <: AbstractManifold{T}` part? This simply means that `Hyperboloid is a subtype of AbstractManifold`.
The same idea as:
$$\text{Hyperboloid} \subset \text{AbstractManifold}$$

What about `c::T` - this will just be a field that must have type `{T}`

So just this part:
```julia
struct Hyperboloid{d, T} <: AbstractManifold{T}
    c::T
end
```

Means:
Create a concrete manifold type called `Hyperboloid`, parameterized by its dimension d and numerical type T, which belongs to the `AbstractManifold{T}` family, and store its positive curvature parameter c using numerical type T.

For now we added no data, we just made the skeleton.

## 2. Minkowski Space Coordinates

When we see something as:
$$(2, 3)$$
What is it? This question is literally a piece of cake. 
This is a point, where:
$$x_1 = 2 \quad and \quad x_2 = 3$$
This two points are basically our $x$
$$x = (x_1, x_2)$$
And this point lives in 2D:
$$x \in \mathbb{R}^2$$

Let us remember about $\mathbb{R}^d$. 
$\mathbb{R}^d$ is a space where all the points consist of $d$ real numbers. Where a normal generic point is:
$$x=(x_1​,x_2​,…,x_d​)$$
We can understand that:
if $d=2$, then we will have $x=(x_1​,x_2​)$ 
if $d = 5$, then we will have $x=(x_1​,x_2​,x_3​,x_4​,x_5​)$

The dimensions simply means "How many independent coordinates". When we say something as: "2-dimensional", we mean "We need 2 coordinates to describe a point in the space" as $(x, y)$.

But we have a weird part... Our hyperbolic space.

Let us say that our hyperbolic manifold has $d$ dimensions.
And therefore, we may expect something as: "Okay, if our hyperbolic space is $d$-dimensional, then I'll store $d$ numbers", for example if we have $d=4$, we may have $x=(x_1​,x_2​,x_3​,x_4)$, right?

Nope.

Not in our Lorentz Model (Same as Hyperboloid model), because instead of storing $d$, we will store:
$$d + 1$$
This is weird... let us understand why we do it.

Let us take a sphere as example.

What is a sphere? 

A sphere is just a 2-dimensional surface 

![[Pasted image 20260904105051.png|339]]

You can move in north/south, east/west, you cannot dig in it or float, this is why it is 2-dimensional.

But we represent it by using 3 coordinates as $(x, y, z)$, why? Because it sits in the ordinary 3D Euclidean space. The points satisfy:
$$x^2+y^2+z^2=1$$
So we are using $\mathbb{R}^3$ to represent a 2D surface.

So the whole idea is:
```
Ambient space:
R³
   ↓
three coordinates

Constraint:
x² + y² + z² = 1
   ↓
removes one degree of freedom

Result:
2-dimensional sphere
```

So the main idea is:
We can describe a $d$-dimensional curved surface using $d+1$ coordinates if we impose one constraint.

Let me explain the degree of freedom. 
The degree of freedom is simply an independent direction you are allowed to move.

For example:

- If you are a ghost floating freely in an empty room, you can move up/down, left/right, and forward/backward. That is 3 degrees of freedom. Mathematically, you live in $\mathbb{R}^3$.

- If you glue your feet to the floor, you can now only slide left/right and forward/backward. You lost one degree of freedom. You are now restricted to a 2-dimensional surface (the floor).

Let us give as another example a 1D circle in a 2D space.
What you have? 

You have a flat 2D sheet of paper ($\mathbb{R}^2$). It gives you 2 coordinates: $(x, y)$. You can move anywhere on the page.

The constraint: You decide you are only allowed to walk in a circle of radius 1 centered at the origin. Your coordinates must obey:
$$x^2 + y^2 = 1$$
Because of this equation, you are no longer truly free. If you pick an $x$ coordinate (say, $x = 0.6$), your $y$ coordinate is locked ($y = \pm 0.8$). You only have 1 independent choice left.

This is what the Lorentz model do.
Even if we have $d$ dimensions, it give us a constrain:
$$\langle x, x \rangle _M = -\frac{1}c$$

Now let us remember the Minkowski dot product, which was:
$$⟨x,y⟩_M=−x_0y_0+x_1y_1+⋯+x_dy_d$$

Where $x_0$ is the time-like coordinate and the other coordinates are space-like.
Let us unpack the constrain of the lorentz model.

- $x$ is just our point, which is described by coordinates: $x=(x_1​,x_2​,…,x_d​)$

- $⟨x,x⟩_M = −x_0^2​ + x_1^2​ + ⋯ + x_d^2​$ - Since we multiply point $x$ by itself, we will get $x_i^2$.

So if we break it, we get:
$$-x_0^2 + x_1^2 + ... + x_d^2 = -\frac{1}c $$
This will define our hyperboloid.

Let us give an example when $d = 2$:

From this we can understand that the hyperbolic space has 2 dimensions, but we need 3 coordinates to describe a point - $x=(x_0​,x_1,x_2)$.

So we will get:
$$−x_0^2​+x_1^2​+x_2^2​=−\frac{1}c$$
And now let us suppose that our $c =1$
We will get:
$$−x_0^2​+x_1^2​+x_2^2​=−1$$
Seems familiar. This is the equation of the hyperboloid.

![[Pasted image 20260904125457.png]]

As we can see, we kept both sides, even if originally we remove a side ($x > 0$ )

But where is the constrain? 

Imagine our ambient space gives you $d+1$ independent knobs (sliders): $x_0, x_1, x_2, \dots, x_d$

The constrain is that once we twist the $d$ of ($x_1, x_2, \dots, x_d$) to whatever values we like. The exact second you let go of those $d$ knobs, the constraint equation steps in. Because the equation must equal $-\frac{1}{c}$, your first knob ($x_0$) has its value automatically calculated for you.
$$\text{You rearrange the math: } x_0 = \sqrt{\frac{1}{c} + x_1^2 + x_2^2 + \dots + x_d^2}$$

Our poor $x_0$ has no free will, this is why due to this constrain, we will have $d$ dimension for the space and $d + 1$ for the coordinates.

Now the `d` from before makes much more sense:
```julia
struct Hyperboloid{d, T} <: AbstractManifold{T}
    c::T
end
```

So if we will have:
```julia
Hyperboloid{2, Float64}
```

We will get:
```
intrinsic dimension = 2
coordinate dimension = d + 1 = 3
```

Now that we understand this idea, we will implement the Minkowski inner product on Julia.
Firstly the mathematical definition is:
$$\langle x, y \rangle_M = x_0y_0 + \sum^d_{i=1}x_iy_i$$
But on julia it will look like:
```julia
function minkowski_dot(x, y)
	return -x[1] * y[1] + sum(x[2:end] .* y[2: end])
end
```

arrays are usually indexed from 1, this is why our `x[1]` is mathematically equivalent to $x_0$.
And our last part `x[2:end]` is mathematically equivalent to $x_1 + ... + x_d$

So the code is:
```
       mathematical             Julia

       -x₀y₀             →      -x[1] * y[1]

       +x₁y₁+...         →      sum(x[2:end] .* y[2:end])
```

Now we will give it a try, to see if we get the same result computationally:

$$x = (1, 2, 3) \quad \text{and} \quad y= (4, 5, 6)$$
Which will equal to:
$$= -(1)(4) + (2)(5) + (3)(6)$$$$= -4 + 10 + 18$$$$= 24$$
On julia the result is...
```
minkowski_dot([1, 2, 3], [4, 5, 6])
# 24
```

(Just to note something... This points are not part of the hyperboloid.)

Okay, now I will repeat an idea (Once again the constrain idea, but this time I will go deeper.)

## 3. Hyperboloid Domain

From the previous lesson we understood that  a $d$-dimensional hyperbolic space uses $d+1$ coordinates in the hyperboloid model (Lorentz model).

So if our $d = 2$, our coordinates will equal $x = (x_1, x_2, x_3)$.
We understand that:
$$x \in R^3$$
But wait!... If we had something as:
$$(2, 5, 7)$$
Will it be a point on the hyperboloid? 
Nope, just because it has 3 coordinates as required, it doesn't automatically makes it a hyperbolic point.

Because we clearly understand that the Minkowski space is bigger than our manifold. 

Think of it this way:
```
             Minkowski space
              R^(d+1)
                  │
        ┌─────────┴─────────┐
        │                   │
     valid              invalid
   hyperbolic             vectors
     points
```

Minkowski space give us the coordinate container, but the hyperboloid equation tells us  which vectors are actually allowed to be points on our manifold.

The rule that tell us this is:
$$\langle x, x \rangle_M = - \frac{1}c$$
We already know what this mean and how its full form looks:
$$−x_0^2​+x_1^2​+⋯+x_d^2​=−\frac{1}c$$
and let us say as before "$c$ = 1", so we get:
$$−x_0^2​+x_1^2​+⋯+x_d^2​=−1$$

Let us check if some points are inside the hyperboloid.

---
For example... we will take $(1, 0, 0)$:
$$⟨x,x⟩M​=−(1)^2+(0)^2+(0)^2$$
Therefore it equals:
$$=-1$$
And our required value for `c` is also -1, so this point is in the hyperboloid.

---
What about $(2, 0, 0)$? Let us check:
$$⟨x,x⟩M​=−(2)^2+(0)^2+(0)^2$$
Therefore it equals:
$$= -4$$
Since:
$$-4 \not = -1$$
This point is not in the hyperboloid

---

Let us choose a $d = 2$, and see what the geometry does.
We would have 3 coordinates:
$$-x_0^2 + x_1^2 + x_2^2 = -1$$
Now we rearrange it:
$$x_0^2 - x_1^2 - x_2^2 = 1$$
![[Pasted image 20260904193513.png|628]]

And compare it to a sphere:
$$x_0^2 + x_1^2 + x_2^2 = 1$$
![[Pasted image 20260904193416.png|655]]

We already noticed something... We noticed how different a single sign change. 
So, for now we did:
```
Calculate <x,x>_M

       ↓

Compare it to -1/c

       ↓

same? → true
different? → false
```

Let us go back for a bit...
Let us remember what we did... and what `c` means again.
We already introduced `c` in our struct:

```Julia
struct Hyperboloid{d, T} <: AbstractManifold{T}
    c::T
end
```

Here `c` is a positive curvature parameter.
Our convention is:
$$K = -c$$

Where $K$ is the actual sectional curvature 
So:

```
c = 1
→ K = -1

c = 2
→ K = -2

c = 0.5
→ K = -0.5
```

Let me explain what we did in this steps and what they mean.

The `c` is the same as `K`, it is just that `c` is the positive curvature. So if we had something as:
$$K = -4 \quad \text{It would mean that} \quad c = 4$$
We introduce it for convenience. Even if we could write the invariant as:
$$-\frac{1}c \quad \text{or} \quad \frac{1}K$$
This is the same idea, it is just that K will be negative anyway, so it doesn't really matter which side we choose.

But beside this constrain, we want know that the Lorentz model consist only of the upper sheet of the hyperboloid, not of both, so we always add even:
$$x_0 >0$$
So a hyperbolic point must satisfy this two invariants:
$$\langle x, x \rangle_M = -\frac{1}c \quad \text{and} \quad x_0>0$$
This is the domain we want.

Let us make a Julia implementation:
```Julia
function on_hyperboloid(x, c)::Bool
	correct_surface = isapprox(minkowski_dot(x, x), -1 / c; atol=1e-6)
	
	upper_sheet = x[1] > 0
	
	return correct_surface && upper_sheet
end
```

Let me explain what it means firstly.
- `on_hyperboloid(x, c)::Bool` - This simply means that we will give to our function the coordinates of a point and the curvature of the manifold.
- `isapprox(minkowski_dot(x, x), -1 / c` - The first buddy that looks strange is `isapprox` just: `isapprox(a value we want to compare, a target value; atol= how much difference can be between them before returning true`

In our case we are saying: "Check if the Minkowski inner product (`minkowski_dot(x,x)`) is equal to the invariant (`-1 / c`), and I will return true even if there is the difference of at least 0.000001 (`atol=1e-6`)."

We will use this idea with `@assert` (Will make everything explode in case we get False).

So, for now, our full code looks like:
```julia
abstract type AbstractManifold{T} end

struct LorentzModel{d, T} <: AbstractManifold{T}
	c::T
end

struct PoincareModel{d, T} <: AbstractManifold{T}
	c::T
end

function minkowski_dot(x, y)
	return -x[1] * y[1] + sum(x[2:end] .* y[2:end])
end

function on_hyperboloid(x, c)::Bool
	on_surface = isapprox(minkowski_dot(x, x), -1 / c; atol=1e-6)
	
	upper_sheet = x[1] > 0
	
	return on_surface && upper_sheet
end
```

For now we did just this part, now we will slowly add more ideas to it. In the next topic we will talk about the Poincare model domains.

## 4. Poincare Ball Domain

As we already know, the Poincare ball model is another way to represent the hyperbolic space. From the last lessons we learned this whole idea about the lorentz model:
$$\mathbb{H}^d_{L} = \left\{ x \in \mathbb{R}^{d + 1}: \langle x,x \rangle_M = - \frac{1}c,x_0 > 0 \right\}$$
From here we understand that the Lorentz model lives in a $d$-dimensional space but we represent a point with $d + 1$ coordinates. And its invariants are that the Minkowski inner product must be equal to  $-\frac{1}c$ and that the first coordinate has to be positive.

But now we have even the Poincare ball model domain is a bit different. For example, In our Lorentz model the points were described by $d + 1$ coordinates, so if $d = 2$  then we would have $x = (x_0, x_1, x_2)$. But our Poincare model is different, because the points are simply described by $d$ coordinates, so if $d = 2$ then we would have $x = (x_1, x_2)$, so this means:
$$y \in \mathbb{R}^d$$

And the invariant is different, because the Poincare ball model fits in it the points that have a magnitude of 1. So it takes only:
$$||y|| < 1$$
Or, if we add the positive curvature:
$$||y|| < \frac{1}{\sqrt{c}}$$
Where $||y||$ is simply the euclidean magnitude ($||y|| = \sqrt{y_1^2 + y_2^2 + ... + y_d^2}$ ).

Therefore the Poincare ball is represented by:
$$\mathbb{B}^d_P = \left\{ y \in \mathbb{R}^d: ||y|| < \frac{1}{\sqrt{c}} \right\}$$

Let us say that $c$ is simply $1$, we get again:
$$||y|| < 1$$
But why smaller than $1$? 

Because the boundaries are not part of the Poincare ball. So if we get: $||y|| = 1$, this is not part of the Poincare ball. 

In the Poincaré ball, points approaching the boundary correspond to points going extremely far away in hyperbolic distance.
So if we had (While $c = 1$):
$$||y|| = 0 \rightarrow \text{In the center}$$
But as:
$$||y|| \rightarrow 1$$
The hyperbolic distance from the center approaches infinity
So each distant point will get squashed in a mess as:
![[PoincareInf.png|594]]

Now, that we understood the idea of the Poincare ball model, we will implement it in Julia as:

```julia
function in_poincare(y, c)::Bool
	return norm(y) < 1 / sqrt(c) 
end
```

That it, nothing hard.

We noticed the difference for sure in this two formulas:
$$\langle x, x \rangle_M = -\frac{1}c \quad \text{and} \quad \|y\| < \frac{1}{\sqrt{c}}$$
We noticed that the Lorentz model condition is an equality (= something), while the Poincare ball model is an inequality (<, <= or >, >=). Why so?

Because the hyperboloid is a surface (So the point must be exactly on the surface - not boarders!
The red points will be on the boarder, while the green on the surface - not inside - on the skin.)
![[hyperbolicSurfaceAndBoarders.png|658]]

The Poincare ball is the interior of a ball (So the point must be inside the boundaries). 
![[PoincareInside.png|556]]

We will use the Poincare model for the Display/Export.

What do I mean with this? 
I mean that the Lorentz model is perfect for mathematical operations, but the problem is that it is too hard to visualize it! This is why we are going to translate all the information from the Lorentz model to the Poincare Ball model and visualize it.

Display -> Visualize it 
Export -> Save the image in a file or something as this.

This is the whole idea of the Poincare Ball domain and the Poincare use. Now we can advance to the next topic.

## 5. Diffeomorphism Operators

Scary name = bad sign? Nah, the topic is not that hard. 
The whole idea is:
$$\text{Lorentz model} \rightarrow \text{Poincare Ball model}$$

We are going to transform from Lorentz to the Poincare Ball. But... to do this we need an adapter function that translates between them.

But before continuing, I will explain the scary word we read from above... 'Diffeomorphism'.

Diffeomorphism has a really straight idea.
Imagine we have two representations of the same geometry.
```
        Hyperboloid                    Poincaré Ball

       x = (x₀,...,x_d)                 y = (y₁,...,y_d)
              │                                  ▲
              │                                  │
              │       forward conversion         │
              └─────────────────────────────────►│
              │                                  │
              │       inverse conversion         │
              ◄──────────────────────────────────┘
```

Diffeomorphism is a mapping that is:
1. bijective - every valid point has exactly one corresponding point;
2. smooth - there are no jumps or tears; whose inverse is also smooth.

So we are basically saying:
These two models represent the same hyperbolic geometry, and there is a smooth, reversible correspondence between their points.

Now let us write what we have:

Lorentz Model -
$$\mathbb{H}^d_{L} = \left\{ x \in \mathbb{R}^{d + 1}: \langle x,x \rangle_M = - \frac{1}c,x_0 > 0 \right\}$$

Poincare Ball Model -
$$\mathbb{D}^d_P = \left\{ y \in \mathbb{R}^d: ||y|| < \frac{1}{\sqrt{c}} \right\}$$
And now we have two steps.

1. Forward: We will convert the Lorentz model in the Poincare Ball model.
2. Inverse: We will convert the Poincare Ball model in the Lorentz model

We will start with the Lorentz model, because we will go by this step usually:
```
                 TRAINING
                    │
                    ▼
          ┌───────────────────┐
          │    Hyperboloid    │
          │    d + 1 coords   │
          │                   │
          │    COMPUTE HERE   │
          └─────────┬─────────┘
                    │
              conversion
                    │
                    ▼
          ┌───────────────────┐
          │   Poincaré Ball   │
          │     d coords      │
          │                   │
          │  DISPLAY / EXPORT │
          └───────────────────┘
```

---
1. Forward: Lorentz -> Poincare

Since we use the Lorentz model, we will start with:
$$x=(x_0​,x_1​,…,x_d​)$$
But our target is:
$$y=(y_1, y_2 , …, y_d)$$

But to get out target result, we will use a formula. This formula will get us our favorable result.
$$y = \frac{x_{1:d}}{\sqrt{c} \cdot x_0 + 1}$$

Let us see how it works, by using $x = (1, 0, 0)$ - (Usually the origin of our Lorentz model):

Let us check if the point is on the hyperboloid:
$$-(1)^2 + 0^2 + 0^2 = -1$$
$$-1 = -1$$
As we can see, it satisfy the first rule. Now let us check if it is on the upper sheet:
$$1 > 0$$
Therefore it is valid.
Now we convert it to the Poincare:
$$y_1 = \frac{0}{1(1) + 1} = 0$$
$$y_2 = \frac{0}{1(1) + 1} = 0$$

Therefore:
$$\boxed{y = (0, 0)}$$
(We looked for two coordinates because we had only 3 coordinates on the Lorentz model of which one is the $x_0$ in the denominators.)

But how do the points stay inside the Poincare Ball?
 We can clearly understand that we don't need the formula to produce some random points for the Poincare ball. We need the point to respect the invariant:
 $$||y|| < \frac{1}{\sqrt{c}}$$
 But we want that this invariant is respected by all the points. Now, let us prove something... 
 $$-x_0^2 + \Vert{}x_{1:d}\Vert{}^2 = -\frac{1}{c}$$
 This is the invariant of the hyperboloid, since:
$$\Vert{}x_{1:d}\Vert{}^2 = \left( \sqrt{x_1^2 + x_2^2 + \dots + x_d^2} \right)^2 = x_1^2 + x_2^2 + \dots + x_d^2$$
So this is:
$$\langle x, x \rangle_M = -x_0^2 + \Vert{}x_{1:d}\Vert{}^2$$
Now we rearrange it:
$$\Vert{}x_{1:d}\Vert{}^2 =x_0^2 -\frac{1}{c}$$

Now let us remember the conversion formula:
$$y = \frac{x_{1:d}}{\sqrt{c} \cdot x_0 + 1}$$

But let us square the transformation:
$$\Vert{}y\Vert{}^2 = \frac{||x_{1:d}||^2}{(\sqrt{c} x_0 + 1)^2}$$

But we know that $\Vert{}x_{1:d}\Vert{}^2 =x_0^2 -\frac{1}{c}$, so we substitute:
$$\Vert{}y\Vert{}^2 = \frac{x_0^2 - \frac{1}{c}}{(\sqrt{c} x_0 + 1)^2}$$
And through a bit of algebraic factoring, we get:
$$\Vert{}y\Vert{}^2 = \frac{1}{c} \frac{\sqrt{c}x_0 - 1}{\sqrt{c}x_0 + 1}$$
This is the proof that prove why it doesn't matter what number we get, since $||y||^2$ will always respect the invariant to don't be bigger than 1.

Now let us implement the conversion on julia:
```Julia
function lorentz_to_poincare(x, c)
	denom = sqrt(c) * x[1] + 1
	return x[2:end] ./ denom
end
```

But now we have even the Poincare → Lorentz transformation.

This is the Reverse part.

---
To get the reverse part we will use a bit of sweet and ol' algebra.

We know that:
$$y = \frac{x_{1:d}}{\sqrt{c} \cdot x_0 + 1}$$

Let us get rid of the denominator (multiply both sides by $\sqrt{c} \cdot x_0 + 1$):
$$x_{1:d} = y(\sqrt{c} \cdot x_0 + 1)$$

Now we will square both sides to look at norms:
$$(x_{1:d})^2 = y^2(\sqrt{c} \cdot x_0 + 1)^2$$

Now we will change the value again ($\Vert{}x_{1:d}\Vert{}^2 =x_0^2 -\frac{1}{c}$):

$$x_0^2 -\frac{1}{c} = y^2(\sqrt{c} \cdot x_0 + 1)^2$$
And with a bit of algebra, we get:
$$x_0 = \frac{1 + c\Vert{}y\Vert{}^2}{\sqrt{c}(1 - c\Vert{}y\Vert{}^2)}$$
and for whatever else beside $x_0$
$$x_i = \frac{2y_i}{1 - c||y^2||}$$

With this steps, we got the reverse part.

But how did we got:
$$1−c||y||^2$$

Let us see how we got it:
$$||y|| < \frac{1}{\sqrt{c}}$$
Now we square both side:
$$||y||^2 < \frac{1}{c}$$
Now we just multiply both sides by $c$:
$$c||y||^2 < 1$$
Therefore:
$$1 - c||y||^2 > 0$$
On Julia it would look like:
```Julia
function poincare_to_lorentz(y, c)
	y2 = sum(y .* y)
	denom = 1 - c * y2
	
	x0 = (1 + c * y2) ./ (sqrt(c) * denom)
	spatial = 2 .* y ./denom
	
	return [x0; spatial]
end
```

But there is a problem... what if the point doesn't follow the rules of the domain? And we get a point that is off the Lorentz model and then try to do the forward pass with the invalid point? We don't want this to happen! This is why we will add `@assert` at the start of each function.

```Julia
function poincare_to_lorentz(y, c)
	@assert in_poincare(y, c)

	y2 = sum(y .* y)
	denom = 1 - c * y2
	
	x0 = (1 + c * y2) ./ (sqrt(c) * denom)
	spatial = 2 .* y ./denom
	
	return [x0; spatial]
end
```

and 

```Julia
function hyperboloid_to_poincare(x, c)
    @assert on_hyperboloid(x, c)

    denom = sqrt(c) * x[1] + 1
    return x[2:end] ./ denom
end
```

So now we are doing:
```
Hyperboloid
     │
     │ @assert on_hyperboloid
     ▼
hyperboloid_to_poincare
     │
     ▼
Poincaré Ball
     │
     │ @assert in_poincare_ball
     ▼
poincare_to_hyperboloid
     │
     ▼
Hyperboloid
```

This was all we did! Now the full code (Just for now...) looks like:

```Julia
abstract type AbstractManifold{T} end

struct LorentzModel{d, T} <: AbstractManifold{T}
	c::T
end

struct PoincareModel{d, T} <: AbstractManifold{T}
	c::T
end

function minkowski_dot(x, y)
	return -x[1] * y[1] + sum(x[2:end] .* y[2:end])
end

function in_hyperboloid(x, c)::Bool
	on_surface = isapprox(minkowski_dot(x, x), -1 / c; atol=1e-6)
	
	upper_sheet = x[1] > 0
	
	return on_surface && upper_sheet
end

function in_poincare(y, c)
    return norm(y) < 1 / √(c)
end

function hyperboloid_to_poincare(x, c)
    @assert in_hyperboloid(x, c)

    denom = sqrt(c) * x[1] + 1
    return x[2:end] ./ denom
end

function poincare_to_lorentz(y, c)
	@assert in_poincare(y, c)

	y2 = sum(y .* y)
	denom = 1 - c * y2
	
	x0 = (1 + c * y2) ./ (sqrt(c) * denom)
	spatial = 2 .* y ./denom
	
	return [x0; spatial]
end
```

Now we can continue with the next chapter! But before it, I recommend to learn all the formulas and understand all the steps, because it is better to understand the reason behind and know the next step instead of learning by heart and copying everything.

Let us start!

# Chapter 2. Tangent Space & the Move Primitives

A small content table:

1. Tangent Space Vector Projection
2. Exponential Map
3. Logarithmic Map
4. Parallel Transport

## 1. Tangent Space Vector Projection

We already know what is a tangent space, why we need it and so on. But I will try to explain it better this time (Because we will go deeper).

We know that the tangent space is simply all the direction we can move while staying on the manifold.

And suppose we have our Lorentz model:
$$\mathbb{H}^d_L$$
Let us say that we have the point $x$ on it:
$$x \in \mathbb{H}^d_L$$
But what will be the tangent space at $x$? We will write it as:
$$T_x\mathbb{H}^d_L$$
And we read it as the tangent space to the hyperbolic manifold at point $x$.

What kind of vectors does the tangent space takes? Well, it takes directions that are logical.
Imagine staying on a really big sphere. We are going to zoom in, what now? We can walk north/south or west/est. Can we move upward without leaving the surface, or can we dig inside the core of the sphere without leaving the surface? Nope. Because the Tangent space is all the possible flat, surface-hugging directions we can move in from your exact spot $x$.

Now let us remember what our Hyperboloid is:
$$\mathbb{H}^d_{L} = \left\{ x \in \mathbb{R}^{d + 1}: \langle x,x \rangle_M = - \frac{1}c,x_0 > 0 \right\}$$
Now let us put a point again:
$$x \in \mathbb{H}^d_L$$
and let us take a vector:
$$v \in \mathbb{R}^{d + 1}$$
But wait... our vector ($v$) is not automatically a valid tangent vector. 
It must satisfy certain conditions before we can state that $v$ is a valid tangent vector.
This is why we will prove why valid tangent vectors shall respect the orthogonality rule.

Let us do some tricks on our Hyperboloid invariant to stay on the surface:
$$\langle x,x \rangle_M = - \frac{1}c$$

Now, let us walk from $x$ along $v$:
$$x(t) = x + tv$$
What did we do? We defined a movement where $x$ is our starting point and $v$ is our movement, while $t$ is simply how many seconds we will move in that direction, for example:

Let our starting point be $x = (2, \sqrt{3}, 0)$.
Let our movement direction be $v = (\sqrt{3}, 2, 0)$.

At $t = 0$: 
$x(0) = (2, \sqrt{3}, 0) + 0(\sqrt{3}, 2, 0) = (2, \sqrt{3}, 0)$ (We are standing still at $x$).

At $t = 1$: 
$x(1) = (2, \sqrt{3}, 0) + 1(\sqrt{3}, 2, 0) = (2 + \sqrt{3}, \sqrt{3} + 2, 0)$ (We have moved along $v$).

Now we want to prove an idea, that if the derivative at $t = 0$ is something else then 0 - it is wrong. Why is it wrong? Because we must stay on the surface and if the derivative at $t=0$ is 0, we respect the rule. But what happens if we get something beside $0$?

Derivative is Negative (e.g., $-4$):
Our constraint value is instantly dropping. In the context of the hyperboloid, this means our vector is pointing inward, making us sink inside the manifold.
Vector points inward.

Derivative is Positive (e.g., $+4$):
Our constraint value is instantly rising. This means our vector is pointing outward into open ambient space, flying away from the surface like jumping off a cliff.
Vector points off the surface - in the space.

So, to make sure the vector change at $t=0$ is 0, we write the next step as:
$$ \frac{\partial}{\partial t}\langle x(t),x(t) \rangle_M \Bigg{|}_{t=0}= 0$$
which means:
at $t=0$ when you take we initial step, the rate of change ($\frac{\partial}{\partial t}$) of your constraint ($\langle x(t),x(t) \rangle_M$) must equal zero ($=0$),

Since we know that:
$$x(t) = x + tv$$
We can substitute it as: 
$$\langle x + tv, x + tv \rangle_M$$
Now we use bilinearity:
$$\langle x, x \rangle_M + t \langle x, v \rangle_M + t \langle v, x \rangle_M + t^2 \langle v, v \rangle_M$$
and with some other operations (symmetry and differentiation with respect to $t$), we will get the new rule:
$$\langle x, v \rangle _M = 0$$
So this is the Minkowski orthogonality check.

This make us doubt something... our gradient.
Let us remember the what is the gradient.

Firstly we write down the Loss:
$$L(x)$$
Which tells us how far we are from the original result. But to do it, we have to change our $x$, which is:
$$x=(x_0​,x_1​,x_2,...​)$$
But to tell how the loss changes while we tweak the $x$, we need our gradient (which tells us how the loss changes when we change $x$)
We write the gradient as:
$$\nabla L(x)$$
What is our gradient equal to?
Our gradient is equal to:
$$\nabla L(x) = \bigg( \frac{\partial L}{\partial x_0}, \frac{\partial L}{\partial x_1}, \frac{\partial L}{\partial x_2}, ... \bigg )$$
This is a vector which tells us: "If you change the coordinates in this direction, the loss changes in a particular way."

But here is a problem... The gradient has no idea about the invariant as: "$x$ must remain on the surface", so if:
$$g = \nabla L(x)$$
and:
$$g \in \mathbb{R}^{d+1}$$
But even so, this doesn't automatically means that 
$$g \in T_x\mathbb{H}^d_L$$
Because let us logically state what the real idea is.
Even if we get that $g \in \mathbb{R}^{d+1}$

What it means? This means:
```
             R^(d+1)
    ┌────────────────────────┐
    │                        │
    │       hyperboloid      │
    │        ╭────╮          │
    │       ╱      ╲         │
    │      • x      ╲        │
    │                        │
    │                        │
    └────────────────────────┘
```

That our gradient lives in the same space as our Hyperboloid. But sadly, our gradient can point in any direction. But we don't need to move in any direction, we need to move along the Hyperboloid. Because we don't want to fly in the space or become one with the Hyperboloid (enter it).
The only allowed directions we can move are: 
$$T_x\mathbb H^{d}_L$$
This is why even if we have:
$$g = \nabla L(x)$$
We want:
$$g_{tan} \in T_x\mathbb H^{d}_L$$
Therefore:
$$g \longrightarrow g_{tan}$$
Is what the tangent projection (I will explain it in no time) does.

But how do we check if the point is a tangent vector of x? We can easily do it by using the Minkowski orthogonality check:
$$\langle x , g \rangle _M = 0$$
and this means:
$$g \in T_x \mathbb H^d_L$$
But there is a problem... that it is not 100% that the result will be 0, and trust me, most of the times it will not be 0. This is why we need to project it.

But what is the projection?
The projection is like a shadow of the original point we had, but now, that shadow is legal and valid for the tangent space. Because imagine this; we have a raw direction vector $v$ which tells us where to move next.

Because machines work in ordinary euclidean space, that raw vector will almost certainly points off into the empty space (and we don't need that, since we need the vector to be on the surface of out Hyperboloid). This is why, the projection will take this not valid vector ($v$) and adjust it enough to make it legal and certified as a surface-hugging tangent vector.

Let me show an example of how it work:

Let us start with the point:
$$x = (2, \sqrt(3), 0) \quad (\text{with  c}  = 1)$$

We do a quick check, so we check if the point is on the Hyperboloid:
$$-2^2 + (\sqrt{3})^2 + 0^2 = -4 + 3 + 0 = -1$$
So this point is valid, since it passes even the upper sheet test: $x_0 > 0$

We remember that the tangent vector shall satisfy this invariant:
$$\langle x , g \rangle _M = 0$$

Now we take an illegal vector:
$$v = (1, 0, 0)$$
Let us check if it is really "illegal"
(Minkowski orthogonality check)

Now we run the formula for our Projection:
$$P_x(v) = v + \langle x, v \rangle _M x$$

Now we will just stick them in the formula:
$$P_x(v) = (1, 0, 0) + (-2) \cdot (2, \sqrt{3}, 0)$$
and we will get:
$$\mathbf{P_x(v) = (-3, -2\sqrt{3}, 0)}$$
Now let us test the new projection:
$$\langle x, P_x(v) \rangle_M = -(2)(-3) + (\sqrt{3})(-2\sqrt{3}) + (0)(0)$$$$= 6 - 2(3) + 0$$$$ =0$$

It passes! The projection formula successfully took our illegal, random raw vector $(1, 0, 0)$ and made it valid as a certified tangent vector $(-3, -2\sqrt(3), 0)$.

This is the idea behind the projection... but now we have two questions: Does the result and position of our original vector changes? and where we will plot it after this?

This is the idea behind the projection... but now we have a question:
Even if the vector changes magnitude and everything after the projection... will the original idea behind its use remain?

Will the idea behind the vector remain?

Yes. If we wanted to move 4 steps nord and 2 steps west, the projection will try to find the mathematically closest valid vector that lives in the tangent plane.

Now this movement vector will be plotted on the tangent space.
We will try to implement on Julia this formula:
$$P_x(g) = g + c\langle x, g \rangle _M x$$

```julia
# We will use the Minkowski

function minkowski_inner(x, y)
    return -x[1] * y[1] + ∑(x[2:end] .* y[2:end])
end 

function project_tangent(x, g, c)
	return g .+ c * minkowski_inner(x, g) .* x
end
```

later we will do even:
```julia
@assert isapprox(minkowski_dot(x, projected_v), 0.0; atol=1e-9, rtol=1e-9)
```

So, our new step will be:
$$x→L(x)→\nabla L(x)→ProjTx​​(\nabla L)→Exp_x​(\text{projected gradient})→x_{new}$$

Now we can continue to some familiar from a side yet unknown topics.

## 2. Exponential map

We already learned that the Projection will take an invalid vector and make it in a valid tangent vector.

Now we will learn how to plot this change on the manifold (Since it is still on the tangent space).

In ordinary Euclidean optimization, we would take a point and a gradient:
$$x \in \mathbb R^d \quad and \quad g \in \mathbb R^d$$
now we would do like good old times:
$$x_{new} = x_{old} - \eta \times g$$
But remember what? This is not the good old times. 

We need to respect some rules as:
$$\langle x, x \rangle _M = - \frac{1}c  \quad \text{and} \quad x_0 > 0$$
and beside this, we are in a $\mathbb R^{d + 1}$ space, so if $d = 2$, we would live in a 2D space, but we would have $x = (x_0, x_1, x_2)$ as coordinates.

So let us give an example why our beautiful and glorious Euclidean optimization would fail in the hyperbolic space.

We take our origin:
$$x = (1, 0, 0)$$
Little check:
$$\langle x, x \rangle_M = -1^2 + 0^2 + 0^2 = -1 \quad \text{and} \quad x_0 = 1 >0$$
This point is on the Hyperboloid.
Now we choose a tangent vector:
$$g = (0, 1, 0)$$
let us check if it is a valid tangent vector.
$$\langle x, g \rangle_M = -(1)(0) + (0)(1) + (0)(0) = 0.$$
Now we use our goat $x_{new} = x_{old} - \eta \times g$  where $\eta = 1$:
The result is:
$$x_{\text{new}} = (1, 1, 0)$$
Now we check if this point is on the Hyperboloid:
$$\langle x_{\text{new}}, x_{\text{new}} \rangle_M = -1^2 + 1^2 + 0^2 = 0$$
...
$$0 \not = -1$$
So it left the Hyperboloid.
$$(1,1,0) \in H^2$$
But this problem will be solved by the exponential map.

So instead of saying "take a straight line through the ambient space", we will say: "start at $x$ and travel along the manifold in the direction $v$" - this path is called geodesic.

The geodesic in the hyperbolic space is:
$$\gamma(t)$$
where:
$$\gamma(0) = x \quad \text{and} \quad \dot\gamma(0) = v$$
Let us unpack the ideas:
$\gamma(t)$ - this is the point on the path at time/parameter 

The exponential map will take two things:
$$Exp_x(v)$$
and return the point you reach after traveling from $x$ in tangent direction $v$ along the geodesic.
Which conceptually means:
$$\boxed{\text{point} + \text{tangent direction} \rightarrow \text{new point on manifold}}$$
and mathematically looks like:
$$Exp_x(v) =\gamma(1)$$

Sooo, if you didn't understand, I will explain it this way:

- You place a particle at point $x$.
- You kick it so its initial velocity is $v$.
- You let it slide along the surface of the Hyperboloid without turning or braking.

So $\gamma (t)$ is just that particle and the idea is:

- $\gamma(0) = x$ just means: "At time $t=0$, the particle is at the starting line."
- $\dot\gamma(0) = v$ just means: "At time $t=0$, the speed and direction of the kick was $v$."
- $\gamma(1)$ just means: "Where did the particle land after exactly 1 second?"

But now a question...  why is the exponential map a replacement for the current $x_{new} = x_{old} - \eta \times g$ ?

Because we want to stay on the manifold, so now, we our step looks like:
$$x \rightarrow L(x) \rightarrow g \rightarrow v = Proj_{T_x}(g) \rightarrow v_{step} = -\eta v \rightarrow x_{new} = Exp_x(v_{step})$$

So we understand that instead of writing:
```julia  
x = x - lr .* nabla_x  
```  
  
We will do:  
```julia  
direction = project_tangent(grad, x, c)  
step = -lr .* direction   
x_new = exp_map_lorentx(x, step, c)
```

Now I will firstly tell what the exponential map is mathematically and then the julia code.

the geodesics is equal to:
$$\gamma(t) = \cosh(\sqrt{c}\Vert{}v\Vert{}_M t)x + \frac{\sinh(\sqrt{c}\Vert{}v\Vert{}_M t)}{\sqrt{c}\Vert{}v\Vert{}_M}v$$
Therefore, setting $t=1$
$$\text{Exp}_x(v) = \cosh(\sqrt{c}\Vert{}v\Vert{}_M)x + \frac{\sinh(\sqrt{c}\Vert{}v\Vert{}_M)}{\sqrt{c}\Vert{}v\Vert{}_M}v$$
as we see:
$$\text{Exp}_x(v) = \gamma(1)$$
But don't memorize it, we will understand it.
The formula has two main parts. It is composed of:
$$\operatorname{Exp}_x(v) = \underbrace{\cosh(\sqrt{c}\Vert{}v\Vert{}_M)x}_{\text{part of starting point}} + \underbrace{\frac{\sinh(\sqrt{c}\Vert{}v\Vert{}_M)}{\sqrt{c}\Vert{}v\Vert{}_M}v}_{\text{part of tangent direction}}$$
Which is just our starting point $x$ and our tangent point $v$ .
What is the Minkowski magnitude equal to?
$$||v||_M = \sqrt{\langle v, v \rangle_M}$$

Now we will do a Julia implementation!
```Julia
function minkowski_inner(x, c)
	return -x[1] * y[1] + ∑(x[2:end] .* y[2:end])
end

function minkowski_norm(x, c)
	return √(minkowski_inner(x, c))
end

function exp_map_lorentz(x, v, c)
	v_norm = minkwoski_norm(v, c)
	
	if v_norm < 1e-12
		copy(x)
	end
	
	a = √(c) * v_norm
	
	return cosh(a) . * x .+ (sinh(a) ./ a) .* v
end 
```

This is the idea behind the Julia code implementation. 
So we usually do:
```
                 AMBIENT SPACE
                      │
                      │ gradient
                      ▼
                   g ∈ R^(d+1)
                      │
                      │ projection
                      ▼
               tangent vector v
                 v ∈ TₓH
                      │
                      │ exponential map
                      ▼
                new point x'
                 x' ∈ H
```

We will use the exponential map to place the new result of our tangent space back on the manifold.

Now we go with the other idea that confused us many times... logarithmic map.

## 3. Logarithmic map

From the past lesson we learned that we use the projection to make a invalid vector into a valid tangent vector on the tangent space and then we learned that after all the mess we do with that valid tangent vector and our starting position; we will push the new point on the manifold by using the exponential map.

Now we will learn what the logarithmic map does. 
The logarithmic map is basically the inverse of the exponential map, because the exponential map says:

Exponential map:
"Start at point `x` and move in the `v` direction along the hyperbolic geodesic, so we arrive at point `y`"
$$\text{Exp}_x(v) = y $$

While the logarithmic map is:
"Start at point `x`, stare at point `y` (the point we want to reach) and give me the tangent vector `v` that will take us there"
$$\text{Log}_x(y) = v$$

The idea is not that hard, because we easily understood it.

But why would we ever need it?
The logarithmic map is really useful in AI engineering, because imagine having two words embeddings:
$$x = \text{"animal"} \quad \text{and} \quad y = \text{"dog"}$$
They are represented as points in the hyperbolic space. 

Now, using a normal vector space model (Word2Vec or standard cosine distance - which we will learn after a bit) is useful... but they look at the similarity of the words:
$$\text{similarity}(\text{dog}, \text{animal}) =\text{similarity}(\text{animal}, \text{dog})$$But real-world knowledge is asymmetric: Every dog is an animal, but not every animal is a dog.

This is why, we will think in terms of displacement, so we understand that:
Moving from Animal $\rightarrow$ Dog is a step from general to specific (entailment).
Moving from Dog $\rightarrow$ Animal is a step from specific to general.

In a Euclidean space, to get the displacement we would just write:
$$y - x$$
But since the hyperbolic space is not as flat as a paper sheet, we will use the logarithmic map:
$$v = \text{Log}_x(y)$$

So even if in Euclidean space the actions are like:
$$\text{Exp}_x(v) = x + v \quad \text{and} \quad \text{Log}_x(y) = y - x$$
But now a question... how does the Logarithmic map knows which tangent vector give us? There are infinitely many directions we can reach point `y`... but the Logarithmic map is smart, so it doesn't says:
"Give me whatever tangent vector that will take me to our destination (`y`)"

It says:
"What is the geodesic from `x` to `y`"
and then it says:
"Which initial velocity will produce that geodesic"

We already know that the geodesic of the Lorentz model is the lorentz distance:
$$d(x, y) = \frac{1}{\sqrt{c}} \text{acosh}\left(-c \langle x, y \rangle_M\right)$$
And the logarithmic is:
$$\text{Log}_x(y) = \frac{d(x, y)}{\sinh\left(\sqrt{c}\, d(x, y)\right)} \left( y + c \langle x, y \rangle_M x \right)$$
As we noticed from the formula.
The last part is basically our projection formula, because we will project it on the tangent space.

This formula is basically:
$$\text{Log}_x(y) = \text{distance} \times \text{direction}$$
Where:

- `distance` - "which way from `x` we need to travel to reach `y`"
- `direction` - "How far away is `x` from `y`" (where $d(x, y) = ||v||_M$)

So, the complete form is:
$$\text{Log}_x(y) = \text{the distance from 'x' to 'y'} \times \text{the unit direction toward 'y'}$$
As said, the exponential map is the opposite.

The exponential map takes our starting point (`x`) and the tangent vector (`v`) to reach our destination point (`y`), while the logarithmic map will simply take our starting point (`x`) and our destination point (`y`) to get our tangent vector (`v`).

So we would use the exponential map when we have:
$$\text{starting point + direction}$$
and we want:
$$\text{a new point on the manifold}$$
This is why we use it in the Riemannian optimization
$$x_{new} = Exp_x(-\eta \times grad_{R}L)$$
(The gradient is already projected)

The exponential map will take our gradient (which is our beautiful tangent vector) and then update our embedding.

And we use the logarithmic map when we have: 
$$x, y \in \mathcal{M}$$
and we want to understand:
$$\text{How do I get from point 'x' to point 'y'}$$
But before we say: "Lad, I would use the distance and solve the same problem"

Let's not confuse the, because the distance s scalar, so it tells us just "How far are this points".
While the logarithmic map is a vector that gives us how far the points are and in which direction.

More precisely it looks like:
$$d(x, y) = ||\text{Log}_x(y)||_M$$
Another important idea is that they depend on the starting point, because:
$$\text{Log}_x(y) \not = \text{Log}_y(x)$$
since:
$$\text{Log}_x(y) \in T_x\mathcal M \quad and \quad \text{Log}_y(x) \in T_y\mathcal M$$
They both live in a different tangent space.
We cannot simply subtract them as if they're ordinary vectors in one common Euclidean space.

This is why... the next topic will be the Parallel transport.

But before continuing, we will do the Julia implementation.
```julia
function minkowski_inner(x, y)
    return -x[1] * y[1] + sum(x[2:end] .* y[2:end])
end

function distance(x, y, c)
    return 1 / √(c) .* acosh(-c .* minkowski_inner(x, y))
end

function minkowski_norm(u)
    return √(minkowski_inner(u, u))
end

function on_hyperboloid(x, c)::Bool
    on_surface = isapprox(minkowski_inner(x, x), -1 / c; atol=1e-9, rtol=1e-9)
    on_upper_sheet = x[1] > 0

    return on_surface && on_upper_sheet
end

function log_map_lorentz(x, y, c)
    @assert on_hyperboloid(x, c)

    d = distance(x, y, c)

    if d < 1e-12
        return zeros(eltype(x), length(x))
    end

    xy = minkowski_inner(x, y)
    u = y .+ c * xy .* x

    u_norm = minkowski_norm(u)
    return (d / u_norm) .* u
end

log_map = log_map_lorentz([1, 0, 0], [1, 2, -2], 1)

println("To go from point x to point y we need to use the tangent vector: v = $log_map")
```

that the entire idea of the logarithmic map.
Now we continue with the next topic.

## 4. Parallel Transport

Let us remember the Euclidean space. If we had a vector:
$$v = (a, b)$$
You could place it at whatever point you want, because the vector will not change, for example:
$$x=(0,0) \quad \text{or} \quad x = (128, 284)$$
It doesn't matter, because the vector doesn't change. It will still remain:
$$v = (a, b)$$
Why so? Because Euclidean space is flat.
But... guess what, I am going to color you surprised - Our manifold is not flat.

Let us remember the tangent space:
$$T_x\mathbb H^d_L$$
This is all the tangents vectors attached to the point `x`
And at another point `y`:
$$T_y\mathbb H^d_L$$

They are in different tangent spaces, so we can't add a tangent vector from the $T_x\mathbb H^d_L$ space with a tangent vector from the $T_y\mathbb H^d_L$ space. 

Because if `v` is a tangent vector attached to the point x:
$$v \in T_x\mathbb H^d_L$$
It doesn't automatically mean that `v` is attached to `y` too:
$$v \in T_y\mathbb H^d_L$$

Imagine that you are walking on a sphere. 
We are standing at point `x`. We have an arrow that is lying on the surface, so we are trying to walk somewhere else to point `y`. 
But sadly, as we walk to point `y`, the surface has rotated under us. imagine a car having a tangent vector that doesn't tilt and do nothing, we will end up with it pointing in the void or poking the ground. But when we use the parallel transport, the vector will actually keep the direction, while slightly tilting the arrow according to the geometry:

![[Screencast From 2026-09-08 14-59-09.webm]]

If we didn't use the parallel transpose, the arrow would stay in a place and pierce the sphere.
This is what the parallel transport means.

But what does the parallel means? 
The parallel in a Euclidean space means: "maintaining the same direction."

But in the hyperbolic space, we replace it for "no unnecessary turning according to the geometry of the manifold."

So we don't need to understand the deep philosophy, we will have to understand this idea:

"Parallel transport = carry a tangent vector along a path without artificially changing its direction.'

But... why do we even care about it? 
We care about it because we will use the RSGD + momentum and the RGSD Adam. 
Suppose at iteration 1:
$$x_1$$
We calculate the gradient:
$$g_1 \in T_{x_1}\mathbb{H}^d_L$$
But we might want even momentum. 
$$m_1$$
(We usually use momentum to speed up the model training and to stop making it bounce from side to side)
The momentum gathers peace when going in a steady direction and it stop the annoying side-to-side in steep areas.

Then, on the next iteration we will have:
$$x_2$$
We will have another gradient:
$$g_2 \in T_{x_2}\mathbb{H}^d_L$$
But wait... our momentum was made for the first point... so is still in:
$$m_1 \in T_{x_1}\mathbb{H}^d_L$$
While our gradient is:
$$g_2 \in T_{x_2}\mathbb{H}^d_L$$
So they are attached to two different points. 
This means that:
$$m_2 = \beta m_1 + (1 - \beta) g_2$$
Wouldn't work, because the gradient is already attached to the point $x_2$, while our momentum ($m_1$) is attached to $x_1$.
(In case you didn't know:

$\beta \in [0, 1)$ (Momentum Coefficient): 
Typically set to $0.9$. It controls memory retention: $\beta = 0.9$ retains $90\%$ of the previous momentum $m_1$)

But parallel transpose fixes it.
We will transport the old momentum ($m_1$) from:
$$T_{x_1}\mathbb{H}^d_L$$
to:
$$T_{x_2}\mathbb{H}^d_L$$
By writing:
$$PT_{x_1 \rightarrow x_2}(m_1)$$
So now we can basically do:
$$m_2 = \beta PT_{x_1 \rightarrow x_2} (m_1) + (1 - \beta)g_2$$

Now they will be valid, since:
$$g_2, m_2 \in T_{x_2}\mathbb H^d_L$$
But now we have a question, how we will reach from `x` to `y`? There are infinite many paths... but as already once said, we have our savior - the geodesic from `x` to `y`. But who will give us the geodesic? 
Of course our Log and Exp map. 

We will do:
$$v_{xy} = \text{Log}_x(y)$$
This will give us the tangent vector we will use to reach from `x` to `y`.
Then we will do:
$$\gamma(t) = \text{Exp}_x(tv_{xy}), \quad \quad 0 \leq t \leq 1$$
So:
$$\gamma(0) = x$$
and:
$$\gamma(1) = y$$
So the Parallel transpose will carry `v` along the geodesic.
So the formula we will get in the end is:
$$PT_{x \rightarrow y} (v) = v + \frac{c\langle y, v \rangle _M}{1 - c\langle x, y \rangle_M}(x + y)$$

So, we have: 
$$v \in T_x\mathbb{H}^d_L$$
Therefore:
$$\langle x, v \rangle _M = 0$$
But we want to construct a new vector:
$$v_y$$
Therefore we want:
$$v_y \in T_y \mathbb H^d_L$$
Sooo, we need:
$$\langle y, v_y \rangle _M = 0$$

And the whole idea of the Parallel transpose formula is simply:
$$v + \text{correction}$$
Where correction is:
$$\frac{c\langle y, v \rangle _M}{1 - c\langle x, y \rangle_M}(x + y)$$

But there is something important... the parallel vector preserves the vector's length, so we will need to get:
$$||v_{new}||_M = ||v||_M$$
Because it is not supposed to grow or shrink somehow, so:
$$\langle v_{new}, v_{new}\rangle_M = \langle v, v\rangle_M  $$

This immediately makes us do two tests.

given:
$$v \in T_x\mathbb{H}^d_L$$
and:
$$w = PT_{x \rightarrow y}(v)$$
We should check the tangent at destination and the norm preserved.

---
1. Tangent at destination

We are trying to check:
$$\langle y, w \rangle _M \approx 0$$

This will tell us if we successfully moved the vector to the correct tangent space.

---
2. Norm Preserved

We are trying to check:
$$||w||_M \approx ||v||_M$$

This will tell us if we preserved the vectors geometric size.

---
We can try even the round trip, to check if it works:
We will start with:

$$v \in T_x\mathbb{H}^d_L$$
Now we will do:
$$w = PT_{x \rightarrow y}(v)$$
Now we reverse it:
$$v = PT_{x \rightarrow y}(w)$$

Now we will expect:
$$w \approx v$$

---
Now we will implement it on Julia:
```julia
function parallel_transport(x, y, v, c)
    xy = minkowski_dot(x, y)
    yv = minkowski_dot(y, v)

    denominator = 1 - c * xy

    coefficient = c * yv / denominator

    return v .+ coefficient .* (x .+ y)
end
```

This way, we finished the chapter 2 - we should have 4 in total. Sooo, this are the least messy. Now I will stop for a bit and write the full code and steps, and yous should learn it perfectly and understand why we do it.

```julia
using LinearAlgebra

abstract type AbstractManifold end

struct LorentzModel{d, T} <: AbstractManifold
    c::T
end

struct PoincareBallModel{d, T} <:AbstractManifold 
    c::T
end

function minkowski_dot(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: AbstractFloat}
    return -x[1] * y[1] + dot(x[2:end], y[2:end])
end

function on_hyperboloid(x::AbstractVector, c::T) where {T <: AbstractFloat}
    on_surface = isapprox(minkowski_dot(x, x), -1 / c; atol = 1e-9, rtol = 1e-9)

    on_uppper_sheet = x[1] > 0

    return on_surface && on_uppper_sheet
end

function in_poincare_ball(y::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    return norm(y) < 1 / sqrt(c)
end

function minkowski_norm(x::AbstractVector)
    return sqrt(abs(minkowski_dot(x, x)))
end

function poincare_to_lorentz(y::AbstractVector, c::Real)
    @assert in_poincare_ball(y, c)
    denom = 1 - c * norm(y)^2
    x0 = (1 + c * norm(y)^2) / (sqrt(c) * denom)
    spatial = (2 .* y) ./ denom
    return vcat(x0, spatial)
end

function poincare_to_lorentz(y::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    @assert in_poincare_ball(y, c)

    a = (1 - c * norm(y)^2)

    return a / sqrt(c) * a 
end

function minkowski_orthogonality_check(x::AbstractVector{T}, v::AbstractVector{T}) where {T <: AbstractFloat}
    return isapprox(minkowski_dot(x, v), zero(T), atol = 1e-10)
end

function project_tangent!(v::AbstractVector, x::AbstractVector, c::Real)
    v .+= (c * minkowski_dot(x, v)) .* x
    return v
end

function parallel_transport(x::AbstractVector{T}, y::AbstractVector{T}, v::AbstractVector{T} ,c::T) where {T <: AbstractFloat}
    numerator = c * minkowski_dot(y, v)
    denominator = 1 - c * minkowski_dot(x, y)

    return v + (numerator / denominator) * (x + y)
end

function exponential_map(x::AbstractVector{T}, v::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    a = sqrt(c) * minkowski_norm(v)

    return cosh(a) .* x + (sinh(a) ./ a) .* v
end

function lorentz_distance(x::AbstractVector{T}, y::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    return (1 / sqrt(c)) * acosh(-c * minkowski_dot(x, y))
end

function logarithmic_map(x::AbstractVector{T}, y::AbstractVector{T}, c::T) where {T <: AbstractFloat}
    d = lorentz_distance(x, y)
    scale = (sqrt(c) * d) / sinh(sqrt(c) * d)
    return scale .* (y .+ c * minkowski_dot(x, y) .* x)
end
```

But we will do it only for fun... because it isn't production ready, for a production ready code we will use mostly libraries. For example a production ready code may look like (I intentionally didn't add allocations, because I am trying to show how to use the libraries):

```julia
using LinearAlgebra
using Manifolds

# ===== Creating our Hyperboloid ===== 
d = 3
M_lorentz = Hyperbolic(d)

# ===== Creating points && Checking them =====
p_lorentz = [1.0, 0.5, 0.0, sqrt(1 + 1.0^2 + 0.5^2)]
q_lorentz = [0.2, 0.8, 0.0, sqrt(1 + 0.2^2 + 0.8^2)]

is_point(M_lorentz, p_lorentz, error=:warn)
is_point(M_lorentz, q_lorentz, error=:warn)

# ===== Creating a tangent vector && Checking it =====
v_raw = [1.0, 2.0, -0.5, 0.0]
is_vector(M_lorentz, p_lorentz, v_raw, error=:warn) # False
 
v_proj = project(M_lorentz, p_lorentz, v_raw)
is_vector(M_lorentz, p_lorentz, v_proj, error=:warn) # True

# ===== Exponential map && Logarithmic_map =====
target_point = exp(M_lorentz, p_lorentz, v_proj)
p_to_q_tangent = log(M_lorentz, p_lorentz, q_lorentz)

# ===== Distance && Parallel Transport =====
dist = distance(M_lorentz, p_lorentz, q_lorentz) # ≈ 0.8075
pt = parallel_transport_to(M_lorentz, p_lorentz, v_proj, q_lorentz)

# ===== Diffeomorphism Operators =====
lorentz_to_poincare = convert(PoincareBallPoint, p_lorentz) # = [0.4, 0.2, 0.0] 
poincare_to_lorentz = convert(HyperboloidPoint, lorentz_to_poincare) # = [1.0, 0.5, 0.0, 1.5]
```

From now on, we will try our best to use just libraries, because companies don't care if you say "I can implement the raw form on Julia!", they care if you can make the code work and be safe. This is why it is highly useless to write everything in raw form (beside if you want to understand better the workflow). But we shall know to write even some formulas, because if we brag "Mannn, I am at a PhD level of knowledge in Differential geometry" -> Get hit with "Calculate the distance between p and q" -> Silence. Not a workflow you'd like.

Now we continue with the most important chapter.

# Chapter 3. Manifold Safeguards and Embeddings

Table of contents:

1. Timelike Re-Projection
2. Poincaré Radius Clamping
3. Guarded acosh/atanh wrappers
## 1. Timelike Re-Projection

We already worked a lot with our hyperboloid, and we know each of its constrains. We know what this means:
$$\mathbb{H}^d_{L} = \left\{ x \in \mathbb{R}^{d + 1}: \langle x,x \rangle_M = - \frac{1}c,x_0 > 0 \right\}$$
We start simply. Let us say $c = 1$ and $d = 2$.
So our hyperboloid lives in:
$$\mathbb{R}^3$$
and the coordinates are:
$$(x_0, x_1, x_2)$$
We know that the constrain is:
$$-x_0^2 + x_1^2 + x_2^2 = -1$$
Let us choose:
$$x = (\sqrt{2}, 1, 0)$$
Little check:
$$−(\sqrt2)^2+1^2+0^2$$
$$\quad \implies -2 + 1$$
$$\quad \quad \implies = -1$$
Let us check the last constrain:
$$x_0 > 0$$
$$\quad \quad \implies \sqrt{2} > 0$$
Yup, it is a valid hyperbolic point. So this means that our point is:
$$x \in \mathbb{R}^2_1$$

Everything looks great!
But we have a problems... the fact that computers don't do perfect math. 
Suppose that a numerical operation gives us this result:
$$x=(1.414214, 1.000001, 0)$$
Now let us check the invariant:
$$\langle x, x \rangle_M = -(1.414214)^2 + (1.000001)^2 +0^2$$
The result wouldn't be $-1$, because the error will be minimal:
$$−0.999999… \quad or \quad −1.000001…$$
Therefore, we will get:
$$x \not \in \mathbb H^2_1$$
even though we may think that numerically this is normal... we just got a floating-point drift.

But why do we care? We care about it because:
```
tiny numerical error
        ↓
x leaves the manifold
        ↓
geometric assumptions become false
        ↓
Exp / Log / distance / tangent operations
are now operating on invalid state
        ↓
eventually potentially NaN / Inf / instability
```

During the run the tiny numerical error can accumulate. This is why we want a safety mechanism.
Firstly we care about the idea as... what does the words Timelike means in Timelike Re-Projection.

We remember about our Minkowski inner product:
$$\langle x, x \rangle _M = -x_0^2 + x_1^2 + ... + x^2_d$$
We call timelike every vector that has:
$$\langle x, x \rangle _M < 0$$
So timelike is a broad term - a vector can be timelike without being on our hyperboloid.

Now, suppose we have a vector that is timelike:
$$z = (z_1, z_2, ..., z_d)$$
with:
$$\langle z, z \rangle_M < 0$$
We want to turn it in a valid hyperboloid point, but even so... what can we do about it? 
Easy. We can normalize it.
For this, we need:
$$\langle x, x \rangle_M < - \frac{1}c$$
Suppose we have:
$$x = \alpha z$$
then:
$$\langle x, x \rangle_M = \langle \alpha z, \alpha z\rangle_M$$
Since the inner term is bilinear:
$$\quad \implies \alpha ^2\langle z, z \rangle_M$$
This means that we want:
$$ \alpha ^2\langle z, z \rangle_M = - \frac{1}c$$
Therefore we will divide both sides by $\langle z, z \rangle_M$:
$$\alpha ^2 = \frac{-\frac{1}c}{\langle z, z \rangle_M}$$
Now we square both sides:
$$\alpha = \sqrt\frac{-\frac{1}c}{\langle z, z \rangle_M}$$
So if $\alpha$ is equal to this, let us go back to our formula:
$$x = \alpha z$$
Now we plug in $\alpha$:
$$x = (\sqrt\frac{-\frac{1}c}{\langle z, z \rangle_M}) z$$
since the place where z stands doesn't matter, since it is a multiplication, we put it here:
$$x = z (\sqrt\frac{-\frac{1}c}{\langle z, z \rangle_M})$$
Ta-da~ Here is our beautiful formula!

Now I will give you a numerical example. Let us say that we have a point with this coordinates:
$$z = (1.5, 1.0, 0.0) \quad \text{and} \quad c=1$$
Let us check if it is timelike:
$$\langle z, z \rangle _M = -(1.5)^2 + 1.0^2 + 0^2$$
$$\implies = -2.25 + 1.0$$
$$\quad \implies = -1.25 $$
Does it pass the timelike check?
$$-1.25 < 0$$
Yup, this vector is timelike, but is it on the hyperboloid?
$$-1.25 \not = -1$$
Nope. The vector is timelike, but it is not on our hyperboloid.

Now let us get our normalizer:
$$\alpha = \sqrt{\frac{-1}{-1.25}} = \sqrt{\frac{4}{5}} = \frac{2}{\sqrt{5}}$$
(We converted the decimals into a whole number by multiplying both numbers by 100)
Now we multiply the vector by this value:
$$x = \frac{2}{\sqrt{5}} \cdot \left(\frac{3}{2}, 1, 0\right) = \left(\frac{3}{\sqrt{5}}, \frac{2}{\sqrt{5}}, 0\right)$$
This is our new vector. Let us check if this vector is on our hyperboloid:
$$\langle x, x \rangle_M = -\left(\frac{3}{\sqrt{5}}\right)^2 + \left(\frac{2}{\sqrt{5}}\right)^2 + 0^2 = -\frac{9}{5} + \frac{4}{5} = -\frac{5}{5} = -1.0$$
We normalized it!

Out whole idea looks like this:
```
                    z
                    │
             Is it timelike?
             <z,z>ₘ < 0
                    │
              ┌─────┴─────┐
             NO           YES
              │             │
           reject       normalize
                            │
                            ▼
                   <x,x>ₘ = -1/c
                            │
                            ▼
                         x₀ > 0
                            │
                            ▼
                    valid hyperboloid
```

Now we will implement the idea on julia.
```julia
function minkowski_dot(x, y)
    return -x[1] * y[1] + dot(x[2:end, y[2:end]])
end

function sanitize_hyperboloid!(x::AbstractVector{T}, c::Real) where {T <: AbstractFloat}
    minkowski_sq = minkowski_dot(x, x)

    if !(minkowski_sq < 0)
        throw(ArgumentError("The vector is not timelike"))
    end
    
    if !(x[1] > 0)
        throw(ArgumentError("The point is not on the upper sheet"))
    end

    scale = sqrt((-1 / c) / minkowski_sq)

    x .*= scale
    return x
end
```

This is the one who checks that our point is on the hyperboloid (This raw form is what we will use, since another doesn't exist).

This was the whole idea behind this concept, now we can keep going with the next topics.

## 2. Poincaré Radius Clamping

We learned a lot about the Poincaré ball model too. We know about its constrains:
$$\mathbb{B}^d_c = \left\{ y \in \mathbb{R}^d: ||y|| < \frac{1}{\sqrt{c}} \right\}$$
We know what everything means, as for:

- `d` - this is the dimension
- `c` - this is the curvature, which has to stay `c > 0`
- `||y||` - this is the normal euclidean distance

the part that we care about is:
`1 / √c` - this is the radius of our Poincare ball.

Which is:
```
                  boundary
              ||y|| = 1/√c
             ╭─────────────╮
          ╭──╯               ╰──╮
        ╭─╯                       ╰─╮
       │            • y             │
       │                            │
       │         origin             │
       │            •               │
        ╰─╮                       ╭─╯
          ╰──╮               ╭──╯
             ╰─────────────╯
```

As already said in the past - the boundary is not part of the manifold!
This is why the formula is:
$$||y|| < \frac{1}{\sqrt{c}}$$
and not:
$$||y|| \le \frac{1}{\sqrt{c}}$$

But why do we care about clamping in the first place?
Because the computers doesn't care about our cute mathematical definition.

Suppose:
$$c = 0$$

The radius will be:
$$r = \frac{1}{\sqrt{1}} = 1$$
This means that our constrains is:
$$||y|| < 1$$
Now imagine optimization pushes an embedding towards the edges:
```
norm = 0.9999999999999999
```

This is still mathematically valid, but the problem is another, that next iteration it may push it to:
```
norm = 1.0000000000000002
```

Now we are outside of the Poincare model due to a floating-point arithmetic. 
And since it is outside the Poincare model, this will become nasty, because many Poincare formulas contain this expression in the denominator:
$$1 - c||y||^2$$
Which is:
$$1 - c||y||^2 = 0$$
at the boundaries
and is:
$$1 - c||y||^2 < 0$$
outside the Poincare ball.
So what will happen eventually? 
Since the number is smaller than 1, the result start growing, and we will eventually go by some doom steps as:
```
division by ~0    1st step of doom - alertness
      ↓
huge values       2nd step of doom - desperation
      ↓
Inf               3rd step of doom - depression
      ↓
NaN               4th step of doom - demoralization
      ↓
gradient explosion   5th step of doom - depersonalization
      ↓
training goes to hell   6th step of doom - nihilism
```

This is why, clamping is a numerical safety barrier. So we are basically saying that if a number gets too close to the boundaries, the safety barrier will have to keep it a tiny distance inside the legal domain. Something as:
```
boundary
│
│  ← ε safety margin
│
├─────────────── safe limit
│
│       • y
│
│
origin
```

We will choose our safety boundaries to be:
$$\frac{0.9999}{\sqrt{c}}$$
But what does it mean? Instead of saying the boundaries to be:
$$r = \frac{1}{\sqrt{c}}$$
So when we have $\sqrt{c} = 1$, we will get:
$$r = 1$$
We will set the safe boundaries at:
$$r_{safe} = 0.9999$$
So we let each point get extremely close to the boundaries, but never reach it.

But how does clamping actually work? Because we understand the safety boundaries, but not the process we will do. This is why I will explain the idea.

Suppose we have:
$$y = (y_1, y_2, y_3,..., y_d)$$
And its norm is:
$$R = ||y||$$
so what will we do if:
$$R < r_{safe}$$
We will do nothing, because it didn't pass through our safety boundaries.

But what if we will get as result:
$$R \ge r_{safe}$$
We will simply shrink it.
The scale factor will be:
$$\alpha = \frac{r_{safe}}{R}$$
and our sanitized version will be:
$$y_{new} = \alpha y$$

This is how the whole idea works, now I will give you an example:
$$y=(0.8,0.60008)$$
We will calculate the magnitude:
$$\Vert{}y\Vert{} = \sqrt{1.0000960064} \approx 1.000048002$$
It is out! Now we will use our safety radius to sanitize it (suppose $c = 1$)
$$\alpha = \frac{0.9999}{1.000048002}$$
$$\approx 0.999852005104$$
This is slightly less than 1. Now we will update our point:
$$y_{new} = 0.999852005104(0.8,0.60008)$$
$$y_{new}= (0.7998816040832, 0.5999911912228)$$
Now we check for its magnitude:
$$\Vert{}y\Vert{} = \sqrt{(0.7998816040832)^2 + (0.5999911912228)^2}$$
$$\sqrt{0.9998000101} \approx 0.9999$$
As we can see, the new outcome is inside the Poincare ball.

Now we implement the idea on julia:
```julia
function sanitize_poincare!(y::AbstractVector{T}, c::T, margin::T = T(0.9999)) where {T <: AbstractFloat}
	
	r_safe = margin / sqrt(c)
	r = norm(y)
	
	if r >= r_safe
		y .*= r_safe / r    # This is simply: y = y .* (r_safe / r)
	end
	
	return y
end
```

This way, we can have this as a safety net for each point in the Poincare.
Since this topic finished, we can continue with the next one.

## 3. Guarded acosh/atanh wrappers

This is our third safety net. Why do we need so many safety nets? 
We need them because the computer can produce something as `0.9999...` instead of `1`, so we may get many problems later.

Let us talk about this new safety net.
Many formulas use `acosh` and `atanh`. We will take just one.

Let us remember the lorentz distance:
$$d(x, y)_{\mathbb H} = \frac{1}{\sqrt{c}}\operatorname{acosh}(-c\langle x, y \rangle_M) $$
`acosh` is the inverse hyperbolic cosine. And it has a range, for example, if we do:
$$\operatorname{acosh}(z)$$
Then:
$$z \ge 1$$
Why? Because `acosh` produces a real-valued output only when the number is 1 or greater.
For example:
```
acosh(1.0)       → valid
acosh(1.5)       → valid
acosh(100.0)     → valid

acosh(0.999999)  → INVALID for real-valued acosh
```

The problem is that we expect the expression inside to be 1 or greater, we want:
$$-c\langle x, y \rangle_M \ge 1$$
But as usual, something may happen as:
```
expected: 1.0000000000000000

actual:   0.9999999999999999
```

Due to the floating-point arithmetic. We have two situations, for example:

Mathematically this will be seen as valid.
```
              ≥ 1
                ✓
```

But a computer is a tad different, it will have a slightly different reaction:
```
0.9999999999999999
        ↓
     acosh(...)
        ↓
      NaN
```

Not cute at all, because our geometry was perfect, it is just that the numerical representation was wrong. 

This why we will use something different. We will use:
```julia
safe_acosh(x)
```

instead of using
```julia
acosh(x)
```

Conceptually we will do:
```
                    x
                    │
                    ▼
             Is x inside
             acosh domain?
                    │
             ┌──────┴──────┐
             │             │
            YES            NO
             │             │
             ▼             ▼
         acosh(x)      clamp to 1
                           │
                           ▼
                        acosh(1)
```

So what will we do? We will simply do:
$$x_{safe} = max(x, 1)$$
Then:
$$\operatorname{acosh}_{safe} = \operatorname{acosh}(max(x, 1))$$
That it. But before supposing that you already know what `max(x, n)` means, I will explain it.
Basically, when we use `max`, both of the numbers (our `x` will get compared to `n`), and it will automatically choose the bigger number. So the idea is:
- If $x \ge 1$: $x_{\text{safe}} = x$ (the value is left unchanged).
- If $x < 1$: $x_{\text{safe}} = 1$ (the value is raised up to $1$).

So when we get something as:
$$z = 0.9999999999999999$$
it will pass through our safety net:
$$\operatorname{acosh}_{safe} = \operatorname{acosh}(max(0.9999999999999999, 1)) =0$$
So instead of getting hit with a `NaN`, we got hit with a `0`.

But there is a small problem... we have to detect the actual bug, not just convert every possible number to 1. Because we can't treat everything equally. I mean, we can get `0.999999999` (small difference), we will just make it be `1`, what if we get `-100` (Gigantic difference)? We should understand where is the problem, not blindly convert all the numbers to 1 and close an eye on the elephant in the room.

So instead of simply doing:
```
tiny numerical violation
↓
repait
```

We will do:
```
tiny numerical violation
        ↓
      repair

massive violation
        ↓
      ERROR
```

An example of this idea is:
```
x = 0.9999999999999999
       ↑
tiny violation → clamp

x = 0.7
       ↑
large violation → something is seriously wrong
```

The whole idea is to repair small violations and to don't hide broken mathematics.
So the Julia implementation is:
```
safe_acosh(x) = acosh(max(x, 1))
```

Now... we start with
$$\operatorname{atanh}(x)$$
Th difference is that `atanh` produces a real-valued output only when:
$$-1 < x < 1$$
As we noticed, `atanh` doesn't take boundaries.
So:
```
atanh(0.0)       ✓
atanh(0.5)       ✓
atanh(-0.9)      ✓
atanh(1.0)       ✗
atanh(-1.0)      ✗
atanh(1.000001)  ✗
atanh(-1.000001) ✗
```

This makes us understand that we can't simply do:
$$x_{safe} = max(-1, min(x, 1)$$
Because this will produce `-1`or `1` - which are technically outside the boundaries. 
This is why we will introduce some tiny margins as:
$$\epsilon > 0$$
and clamp into:
$$[-1 + \epsilon, 1 - \epsilon]$$

So the whole idea is:
```
Poincaré safeguard
       │
       ▼
radius clamp
       │
       ▼
safe ||x||
       │
       ▼
guarded atanh
       │
       ▼
stable computation
```

so the Julia implementation looks like:
```julia
function safe_atanh(x; ϵ = 1e-12)
    return atanh(clamp(x, -1 + ϵ, 1 - ϵ))
end
```

What is `clamp(x, lo, hi)`?, this is simply a system:
$$\operatorname{clamp}(x, a, b) = \begin{cases} 
a & x < a \\\\ 
x & a \le x \le b \\\\ 
b & x > b 
\end{cases}$$
we write our `x` (the one getting checked) as the first number, the we write the lowest number our `x` will get converted into if small enough, and in the end we will write the greatest number.

This were the safeguards session, now we will continue with the embedding part - a hard one. So we do a mental preparation of the Julia code:
```julia
function minkowski_dot(x, y)
    return -x[1] * y[1] + dot(x[2:end, y[2:end]])
end

function sanitize_hyperboloid!(x::AbstractVector{T}, c::Real) where {T <: AbstractFloat}
    minkowski_sq = minkowski_dot(x, x)

    if !(minkowski_sq < 0)
        throw(ArgumentError("The vector is not timelike"))
    end
    if !(x[1] > 0)
        throw(ArgumentError("The point is not on the upper sheet"))
    end

    α = sqrt((-1 / c) / minkowski_sq)

    x .*= α
    return x

end

function sanitize_poincare!(y::AbstractVector, c::Real, margin::T = T(0.9999)) where {T <: AbstractFloat}
	
	r_safe = margin / sqrt(c)
	r = norm(y)
	
	if r > r_safe
		y .*= r_safe / r
	end
	
	return y
end

safe_acosh = acosh(max(x, 1))

function safe_atanh(x; ϵ = 1e-12)
    return atanh(clamp(x, -1 + ϵ, 1 - ϵ))
end
```

This are our safety nets. Now we will try to learn how to embed - one of the few reasons we are cramming so much geometry for.
Now we will go to the next part of the chapter, the embedding.

# Chapter 4. Embedding

Table of contents:

1. HyperboloidEmbedding struct
2. PoincareEmbedding struct
3. Fermi-Dirac Loss
4. Hyperbolic InfoNCE
5. RSGD Graph Optimization Loop

## 1. HyperboloidEmbedding struct

Let us start slowly with the idea.

Suppose we have a graph:
```
        A
       / \
      B   C
     / \
    D   E
```

We want each vector to receive a vector.
In Euclidean ML, we might do:
```
A → [ 0.2,  0.7, -0.1]
B → [-0.4,  0.3,  0.8]
C → [ 0.1, -0.2,  0.5]
D → [ 0.8,  0.1, -0.3]
E → [-0.2,  0.9,  0.4]
```

So each node gets a vector in:
$$\mathbb{R}^3$$
Which will be all the possible vectors containing 3 real numbers.
So, as an example we may have:
$$(2.3,−1.7,0.4) \in \mathbb{R}^3$$
But the hyperbolic embedding looks at another idea.

Instead of saying:
"Every node can be a vector of $\mathbb R^d$"

We say:
"Every node must be a valid point on the hyperboloid"

But now let us think about the embedding matrix, how it looks?

Suppose:
`N` - the number of the nodes
`d` - the hyperbolic dimension

But now each node needs $d + 1$ Lorentz coordinates. We can write the whole idea as:
$$X \in \mathbb R^{(d+1) \times N}$$

So we get:
```
             N nodes
        ┌──┬──┬──┬──┬──┐
        │  │  │  │  │  │
 d+1    │  │  │  │  │  │
 coords │  │  │  │  │  │
        │  │  │  │  │  │
        └──┴──┴──┴──┴──┘
         A  B  C  D  E
```

vertically, we have the dimension number, while horizontally we have the Nodes (Think about them as the samples)

This can mean:
$$X[:, i]$$
Which translates simply as:
"the entire embedding vector belonging to node $i$"

Let me give an example, let us say that we have 4 nodes:

- Node 1: Organism (Root node at the center of the hierarchy)
- Node 2: Mammal (Branch node)
- Node 3: Bird (Branch node in a different direction)
- Node 4: Dog (Leaf node under Mammal)

Let us say that they are in a $d = 3$ huperbolic space, this mean that each node needs $d + 1 = 4$ coordinates $(x_1, x_2, x_3, t)$ (The time node is the last!). 

We make them as hyperbolic points (where we have 4 nodes and 4 coordinates).
$$\begin{bmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ t \end{bmatrix} = \begin{bmatrix} 0.000 & 2.129 & 0.000 & 10.018 \\\\ 0.000 & 0.000 & 2.129 & 0.000 \\\\ 0.000 & 0.000 & 0.000 & 0.000 \\\\ \mathbf{1.000} & \mathbf{2.352} & \mathbf{2.352} & \mathbf{10.068} \end{bmatrix}$$$$\hspace{1.3cm} \begin{array}{cccc} \uparrow & \uparrow & \uparrow & \uparrow \\\\ \text{Node 1} & \text{Node 2} & \text{Node 3} & \text{Node 4} \\\\ \text{(Organism)} & \text{(Mammal)} & \text{(Bird)} & \text{(Dog)} \end{array}$$
Now, if we write:
```julia
X[:, 3]
```

We are demanding the 3rd node and all the rows.
$$n_3 = \begin{bmatrix} 0.000 \\\\ 2.129 \\\\ 0.000 \\\\ 2.352 \end{bmatrix}$$
We will get what we demanded.

What would happen if we do:
```julia
X = randn(4, 100)   # We will get 100 nodes with d = 3 (4 lorentz coordinates)
```

Will this points be on the hyperboloid? 
Almost certainly not!

Because for every column `X[:, i]`, we need:
$$\langle X[:, i], X[:, i] \rangle_M \approx -\frac{1}c$$

and
$$X[1, i] > 0$$

If we look at our vector ($n_3$), let us check the invariants.
$$\langle n_3, n_3 \rangle_M = (0.000)^2 + (2.129)^2 + (0.000)^2 - (2.352)^2$$$$= 0 + 4.532641 + 0 - 5.531904$$$$= -0.999263 \approx -1.0$$
Now we will just pass it through the Timelike Re-projection and it matches easily! Because the difference is almost nonexistent. Now check if it is on the upper sheet
$$X[4, 3] = 2.352 > 0 \quad \text{(PASSED)}$$
We took the 4th row (in case the time coordinate is the last like in our situation we will simply write `X[d + 1, i]`) because there we have the time coordinate (We choose $d = 3$, so the time coordinate will be $d  = 3 + 1 = 4$).

Did you know that we can actually get whatever point we want on the hyperboloid? We just have to make a coordinate codependent on the other coordinates. We will make $x_0$ to adapt to other points. Now I will show how, but firstly let us get the formula.

$$-x_0^2 + \sum_{i = 1} ^d x_i^2 = -\frac{1}c$$
we know that $x_0$ is the time like coordinate, while all other $x_{1:d}$ are the spatial.

now let us isolate the our time like coordinate - $x_0^2$:
$$-x^2_0 = - \frac{1}c - \sum_{i = 1} ^d x_i^2 $$
Now we change the signs by multiplying everything by `-1`:
$$x^2_0 = \frac{1}c + \sum_{i = 1} ^d x_i^2 $$
Now we square both sides, so we get rid of the exponential.
$$x_0 = \sqrt{\frac{1}c + \sum_{i = 1} ^d x_i^2}$$
We can rewrite:
$$\sum_{i = 1} ^d x_i^2 = ||x_{spatial}||^2$$
we substitute:
$$\boxed{x_0 = \sqrt{\frac{1}c + ||x_{spatial}||^2}}$$

This is our formula! This formula will let us choose whatever coordinates we want, beside the $x_0$. Let us try it.
We choose $d = 4$ (which are $d = 4 + 1 = 5$ Lorentz coordinate) and assume that the $c = 1$. I will choose...
$$\mathbf{x}_{\text{spatial}} = [-1.25, \, 1.0, \, 0.25, \, 2.0, x_0]$$
as random points. As you can see, I didn't said what my timelike coordinate will be. Now the math will tell us:
$$\Vert{}\mathbf{x}_{\text{spatial}}\Vert{}^2 = (-1.25)^2 + (1.0)^2 + (0.25)^2 + (2.0)^2$$$$= 1.5625 + 1.0 + 0.0625 + 4.0 = 6.625$$
Now we can get easily our timelike coordinate:
$$x_0 = \sqrt{1 + \Vert{}\mathbf{x}_{\text{spatial}}\Vert{}^2} = \sqrt{1 + 6.625} = \sqrt{7.625} \approx 2.76134$$
So now we got our timelike coordinate!
$$x = [-1.25, \; 1.0, \; 0.25, \; 2.0, \; \mathbf{2.76134}]$$

Let us see if it is on the hyperboloid:
$$\langle x, x \rangle_M = (-1.25)^2 + (1.0)^2 + (0.25)^2 + (2.0)^2 - (2.76134)^2$$$$= 6.625 - 7.625 = -1.0 \quad \checkmark \quad and \quad x_0 = 2.76134 > 0 \quad \checkmark$$
So it is valid!

Now let us think about putting some nodes on the manifold. The problem is one, we don't know where each shall be placed, because that the optimizations job - this is why we place them randomly. Even if we place them randomly, we don't even want to place points ridiculously far away, some nodes near the boundary, or some nodes with enormous gradients. 

This is why we want a controlled starting configuration. So we may start near the origin:
$$o = (1, 0, 0, 0) \quad \leftarrow \text{the timelike coordinate is the first one this time}$$
And the optimization will do its job.

But placing them near the origin doesn't mean that they have to be in a place, that would be horrible. So this is why they will get plotted with a bit of noise, so the optimization can separate them later.

Now let us say that we have a graph:
```
             A
          /  |  \
         B   C   D
        / \      |
       E   F     G
```

We want an object that contains something that looks like this:
```
HyperboloidEmbedding
│
├── manifold
│      └── H³
│
├── embeddings
│      ┌───────────────┐
│      │ A │ B │ C │ D │ ...
│      └───────────────┘
│
└── graph
       └── nodes + edges
``` 

For a 3-dimensional hyperbolic space:
$$\mathbb H^3$$
Each node will be represented by 4 coordinates, something as:
```
A → [x₀, x₁, x₂, x₃]
B → [x₀, x₁, x₂, x₃]
C → [x₀, x₁, x₂, x₃]
...
```

Therefore if there are `N` nodes, we will write:
```julia
embedding :: Matrix{Float64}

size = (N, 4)
```

The column is the important unit, since it is the same as:
```julia
embeddings[:, i]
```

which means that the hyperbolic point belong to node `i`.

We will write the object as:
```julia


mutable struct HyperboloidEmbedding{T}
	manifold::Hyperbolic
	embeddings::Matrix{T}
	nodes::Vector
end
```

But before continuing I will point out something. The `Vector` notation is too vague, we want the node identifiers to be explicit. So when we have a vector composed of integers we will get:
```julia
nodes::Vector{Int}
``` 

Yet the problem is we can't even restrict it so badly, because we will deal with floats, with strings, with UUIDs, and so on...

This is why we will give it a generic parametric type:
```julia
mutable struct HyperboloidEmbedding{T, N}
	manifold::Hyperbolic
	embeddings::Matrix{T}
	nodes::Vector{N}
end
```

Now they are more free, because:
```julia
embeddings::Matrix{Float64}
nodes::Vector{Int}
```


But we will make this structure for our embedding:
```julia
using Manifolds
using Graphs
using JSON3
using LinearAlgebra
using Random

mutable struct HyperboloidEmbedding{T, N, G<:AbstractGraph}
    manifold::Hyperbolic
    graph::G
    nodes::Vector{N}
    nod_to_col::Dict(N, Int)
    embeddings::Matrix{T}
end
```

But what does mutable means? We already learned about it, but in case of something, I will just add a few words. Why did we write `mutable` right from the start? Because Julia structs are immutable by default - meaning you can't change the values inside once created. `mutable` tells Julia: "Hey, let me update these values later."

Because we are going to update the embeddings.
```
epoch 1
A → x₁

       ↓ optimization

epoch 2
A → x₂

       ↓ optimization

epoch 3
A → x₃
```

So we write `mutable` because the values will change. 

But even so.. how do we create our near origin initialization? 
We will not throw the die and place them randomly, we will start with giving the spatial coordinates a small value:
```julia
spatial = 0.01 .* randn(d)
```

Then we will get our timelike coordinate with our previous formula:
$$x_0 = \sqrt{\frac{1}c + ||x_{spatial}||^2}$$

Now we implement the whole idea on Julia, but before it, I will write the process down:
```
small random spatial coordinates
             │
             ▼
     calculate x₀
             │
             ▼
       valid point
             │
             ▼
       store column
```

Now we can implement the near-origin initialization in a function. This is why we will implement it on Julia:
```julia
function near_origin_initialization(
	d::Int,
	c::T,
	rng::AbstractRNG;
	scale::T = T(0.01),
) where {T <: AbstractFloat}
	
	spatial = scale .* randn(rng, T, d)
	
	x0 = sqrt(inv(c) + dot(spatial, spatial))
	
	return [x0; spatial]
```

What does this function do? 

I will explain.
Imagine that we choose `d=2`, `c=1`, and `scale=0.01`

Now let us say that `randn(rng, Float64, 2`) gave us something as:
$$x_{spatial} = [3.0, -4.0]$$

Now we will multiply it by `scale = 0.01`
$$x_{spatial} = [0.03, -0.04]$$

Now, we will comput their Minkowski dot product:
$$\langle x_{spatial}, x_{spatial} \rangle = 0.03^2 + (-0.04)^2 = 0.00009 + 0.0016 = 0.0025$$

Now we calculate our time coordinate:

Firstly we substitute $c = 1.0$ `(inv(1.0) = 1.0)` and add it to the Minkowksi dot product of our spatial coordinates:
$$x_0 = \sqrt{\frac{1}{1.0} + 0.0025} = \sqrt{1.0 + 0.0025} = \sqrt{1.0025} \approx 1.00124922$$

Now our final result is:
$$x = [1.00124922, 0.03, -0.04]$$

And it is a valid hyperbolic point. Each node will pass through this process and get an actual random position close to origin.

Now we are going to build the embedding matrix:
```julia
function embedding_matrix(
    d::Int,
    c::T,
    n_nodes::Int,
    rng::AbstractRNG;
    scale::T = T(0.01),
) where {T <: AbstractFloat}

    X = Matrix{T}(undef, d + 1, n_nodes)

    for i in 1:n_nodes
        X[:, i] .= points_initialization(d, c, rng; scale)
    end

    return X
end
```
I will explain what the `;` do next to `rng`.

Let us say that we have:
```julia
function random_numbers(a::Int=2, b::Int=6, c::Float64=9.2
	return a + b + c
end
```

Let us say that we want to give `a` a value as `8` and `c` a value as `15.5`. 
```julia
random_numbers(8, 15.5) # Ah, wait, we want to give the value to c, not b...
random_numbers(8, c=15.5) # Wait... We got hit with a MethodError: no method matching...

# What can i do? 
```

The answer is easy, we will use the `;`, instead of `,`, for some values.
```julia
function random_numbers(a; b::Int=6, c::Float64=9.2)
	return a + b + c
end
```

Now we can simply write:
```julia
random_numbers(8, c=15.5)

"""
Output:

29.5
"""
```

But we will usually use everything in one function, something as:

```julia
using LinearAlgebra
using Graphs
using JSON3
using Random
using Manifolds

mutable struct HyperboloidEmbedding{T, N, G<:AbstractGraph}
	manifold::Hyperbolic
	graph::G
	nodes::Vector{N}
	node_to_col::Dict{N, Int}
	embeddigs::Matrix{T}
end

function HyperboloidEmbedding(
	graph::G,
	nodes::Vector{N},
	d::Int,
	rng::AbstractRNG = Random.default_rng(),
	scale::T = T(0.01)
	) where {T<:AbstractFloat, N, G<:AbstractGraph}
	
	if !(nv(graph) == length(nodes))
		throw(ArgumentError("The graph and nodes doesn't describe the same vertices (edges)"))
	end
	
	if !(length(unique(nodes)) == length(nodes))
		throw(ArgumentError("A node or more don't have an unique identifier"))
	end
	
	M = Hyperbolic(d)
	
	node_to_col = Dict(node => i for (i, node) in enumerate(nodes))
	
	X = Matrix{T}(undef, d + 1, length(nodes))
	
	for i in axes(X, 2)
		X[:, i] = near_origin_points(M, c, nodes, rng, scale)
	end
	
	return HyperboloidEmbedding(M, graph, nodes, node_to_col, X)
end
```

This is the whole idea of our code... But as usual, nobody will hit you with a julia list with nodes and edges. Instead they will hit you with JSON and CSV. This is why I will teach you how to handle this situations.

## 1.1. Handling CSV and JSON

In this part I will explain how to handle the files on Julia, since we will need it.

Let us say that we have a json called `graph.json`, the `json` will have inside something as:
```json
{
    "nodes": [...],
    "edges": [...]
}
```

This is why we want to steal the data from it.

We will simply write:
```Julia
using JSON3  # firstly add it by clicking in the julia terminal `]`, then we will simply write: add JSON3

data = JSON3.read(read("graph.json", String))

# It will look like this:
"""
data:

{
     "directed": true,
   "multigraph": false,
        "graph": {},
        "nodes": [
                   {
                      "name": "Mental Health Disorders",
                      "type": "disease",
                        "id": "n01"
                   },
                   {
                      "name": "Mood Disorders",
                      "type": "disease",
                        "id": "n02"
                   },
                   {
                      "name": "Anxiety Disorders",
                      "type": "disease",
                        "id": "n03"
                    ...
                    
""" 
```

This will take our json file and actually give the content into a variable. We can access both data by simply writing:
```julia
println(data.nodes)

"""
JSON3.Object[{
   "name": "Mental Health Disorders",
   "type": "disease",
     "id": "n01"
}, {
   "name": "Mood Disorders",
   "type": "disease",
     "id": "n02"
	}
	...
"""

println(data.edges)

"""
JSON3.Object[{
   "relation": "subtype_of",
     "source": "n02",
     "target": "n01"
}, {
   "relation": "subtype_of",
     "source": "n03",
     "target": "n01"
}, {
"""
```

So we may even get the parts by writing:

```julia
first_node = data.nodes[1]

println(first_node.id)
println(first_node.name)
println(first_node.type)

"""
Output:

n01
Mental Health Disorders
disease
"""
```

But instead of always writing `data.nodes[...]`, we would get them in a easier way by simply writing:
```
node_ids = [String(node.id) for node in data.nodes]
node_names = [String(node.name) for node in data.nodes]
node_types = [String(node.type) for node in data.nodes]
```

Now we can do:
```Julia
println(node_ids[1])
println(node_names[1])
println(node_types[1])

"""

"n01"
"Mental Health Disorders"
"disease"

"""
```

Since our embedding matrix will look like:
```
                columns
             1    2    3    4   ...
          ┌────┬────┬────┬────┐
x₁        │    │    │    │    │
x₂        │    │    │    │    │
x₃        │    │    │    │    │
...       └────┴────┴────┴────┘
```

We need to transform the id into index, something as:
```julia
node_to_index = Dict(id => i for (i, id) in enumerate(node_ids))
```

Now we will get:
```julia
println(node_to_index["n01"])

"""
Output:

1
"""
```

But we will use even `csv`, where suppose we have a file named `edges.csv`:
```julia
using CSV
using DataFrames

edges = CSV.read(read("edges.csv", DataFrame))
```

In this case, our columns will be:
```
source | target | relation
```

and we can access to them as:
```julia
edges.source
edges.target
edges.relation

"""
n02
n01
subtype_of
"""
```

Now we can add the edges to the graph.
```julia
for row in eachrow(edges)
    source_index = node_to_index[row.source]
    target_index = node_to_index[row.target]
end
```

To create the graph we will do:
```julia
graph = SimpleDiGraph(length(node_ids))
```

If we connect both ideas we get:
```julia
graph = SimpleDiGraph(length(node_ids))

for row in eachrow(edges)
	src = node_to_index[row.source]
	dst = node_to_index[row.target]
	
	add_edges!(graph, src, dst)
```

This is the whole idea of what we will do. Now we can do a little project (even if we will let it at half, because we still don't know the RSGD and how to implement it. But we may try to do the start, so we don't end up implementing everything in one lesson and get overwhelmed).
## Start of the Project

Sadly, now we are not going to play kids game anymore with a "trust-me-people-will-like-the-toy-implementation-code" - nobody cares about a toy implementation. We are going to use a real data csv - a medical one this time.

`nodes.csv`:
![[Pasted image 20260913082829.png]]

`edges.csv`:
![[Pasted image 20260913082754.png]]

And our `graph.json`, that I will show just a small part of:
```json
directed	true
multigraph	false
graph	{}
nodes	

0:
name:	"Mental Health Disorders"
type:	"disease"
id:	"n01"

1:	
name:	"Mood Disorders"
type:	"disease"
id:	"n02"
...	
```

I will do the whole code, and make sure you will understand it, because it would be really useful, since this part is always used in embedding.

```julia
using LinearAlgebra
using Graphs
using Random 
using JSON3
using Manifolds
using CSV
using DataFrames

# The first thing we will start with, will be extracting our data
data = JSON3.read(read("/home/shuposhuposhrimpo/Downloads/graph.json", String))

nodes = data.nodes

node_ids = [String(node.id) for node in data.nodes]
node_name = [String(node.name) for node in data.nodes]
node_types = [String(node.type) for node in data.nodes]

node_to_idx = Dict( id => i for (i, id) in enumerate(node_ids)) # We will return it in the embedding.

edges = CSV.read("/home/shuposhuposhrimpo/Downloads/edges.csv", DataFrame)

graph = SimpleDiGraph(length(node_ids))

for row in eachrow(edges) 
    src = node_to_idx[row.source]
    dst = node_to_idx[row.target]

    add_edge!(graph, src, dst)
end

# We got everything we need, now we can start with our HyperboloidEmbedding structure

mutable struct HyperboloidEmbedding{T, N, G <: AbstractGraph, M <: AbstractManifold}
    manifold:: M
    graph::G
    node::Vector{N}
    node_to_col::Dict{N, Int}
    embedding::Matrix{T}
end

function near_origin_initialization(
    M::Hyperbolic,
    ::Type{T},
    rng::AbstractRNG;
    scale::T = T(0.01),
) where {T <: AbstractFloat}

    d = manifold_dimension(M)

    spatial = scale .* randn(rng, T, d)

    x0 = sqrt(one(T) + dot(spatial, spatial))

    return [x0; spatial]
end

function HyperboloidEmbedding(
    graph::G,
    node::Vector{N},
    d::Int,
    ::Type{T},
    rng::AbstractRNG = Random.default_rng(),
    scale::T = T(0.01),          
) where {T <: AbstractFloat, N, G <: AbstractGraph}

    if !(nv(graph) == length(nodes))
        throw(ArgumentError("The graph vertices doesn't match the nodes vector"))
    end

    if !allunique(nodes) # It has the same idea as the:
    #  length(unique(nodes)) != length(nodes)
    throw(ArgumentError("The nodes don't have an unique identifier"))
    end

    M = Hyperbolic(d)
    node_to_col = Dict(n => i for (i, n) in enumerate(node))

    X = Matrix{T}(undef, d + 1, length(nodes))

    for i in axes(X, 2)
        X[:, i] .= near_origin_initialization(M, T, rng; scale)
    end

    return HyperboloidEmbedding(M, graph, node, node_to_col, X,)
end

embedding = HyperboloidEmbedding(graph, node_ids, 2, Float64)
```

This is just the first part. Now we will continue the learning part and use this code as example as we continue. 

## 2. PoincareEmbedding struct

We will not make the Poincare ball as a totally new struct we will use every two seconds. We will use the Poincare Ball for the simple idea of Display/Export. This is why we will simply do:

```
HyperboloidEmbedding
        │
        │ to_poincare(...)
        ▼
PoincareEmbedding
```

This is why we will not create a new embedding. We will convert the existing embedding into the Poincare one.

The `struct` is identical, beside a small change in the manifold:
```julia
mutable struct PoincareEmbedding{T, N , G <: AbstractGraph}
	manifold::
	graph::G
	node::Vector{N}
	node_to_col::Dict{N, Int}
	embedding::Matrix{T}
end
```

Now we have to convert from Hyperboloid to Poincare. To which the code will be:
```Julia
mutable struct PoincareEmbeddig{T, N, G <:AbstractGraph}
    manifold::Hyperbolic
    graph::G
    nodes::Vector{N}
    node_to_col::Dict{N, Int}
    embedding::Matrix{T}
end

function to_poincare(
    H::HyperboloidEmbedding{T,N,G}
) where {T, N, G <: AbstractGraph}

    X = H.embedding

    Y = Matrix{eltype(X)}(undef, size(X, 1) - 1, size(X, 2))

     for i in axes(X, 2)
        Y[:, i] .= convert(PoincareBallPoint, X[:, i])
    end

    return PoincareEmbeddig(M, H.graph, copy(H.node),copy(H.node_to_col), X,)

end
```

The idea `convert` is easy, it is simply:
```
convert(PoincareBallPoint, hyperboloid_point)
```

The topic was relatively small, but we will continue, so we reach interesting parts where we can make codes from scratch to try some embeddings.

## 3. Fermi-Dirac Loss

In the previous lessons we learned how to build structures and so on, but now we need a way to tell the model: 
"This two nodes are connected. Their embedding should therefore be close"

Let us imagine this:
```
Mental Health Disorders
        │
        ├── Mood Disorders
        └── Anxiety Disorders
```

From our graph, we will have:
```
n02 → n01   subtype_of
n03 → n01   subtype_of
```

Therefore the graph tells us:
```
n02 and n01 are related
n03 and n01 are related
```

we want their embeddings to reflect that.

The entire idea of this chapter will be:
```
graph relationship
       ↓
"these should be close"
       ↓
loss function
       ↓
gradient
       ↓
move embeddings
```

We want a function that converts the distance into a probability of the nodes being connected.
This function will be:
$$p(d) = \frac{1}{e^{(d - r)/t} + 1}$$
- `d` - this is the hyperbolic distance between two nodes (Poincare distance or Lorentz - it depends by us).
- `r` - the radius 
- `t` - the temperature controlling how sharply things change
- `p(d)` - the predicted probability that two nodes are connected

Let me give you a toy example, suppose that:
$$r = 1 \qquad \text{and} \qquad t=0.5$$
Now let us say that the distance between two nodes are:
$$d= 0.5$$
Let us calculate the loss:
$$p(d) = \frac{1}{e^{(0.5 - 1)/0.5} + 1}$$
$$\hspace{1.5cm}\implies p(d) = \frac{1}{e^{-1} + 1}$$$$\hspace{3cm} \implies p(d) \approx 0.731$$
Which is $73.1$% that the nodes are connected.

But some mere probabilities are not enough, we need a loss.
let us say that we do:
$$y = \begin{cases} 1 & \text{connected} \\\\ 0 & \text{not connected} \end{cases}$$
Let us choose the binary cross entropy:
$$L = -y \log(p) - (1 - y) \log(1 - p)$$

Let me give an example of how it will work, let us  say that we have a positive edge ($y = 1$)
Then we will get:
$$L = -\log(p)$$
Let us say that we got $p = 0.9$, by passing it through this function, we will get:
$$L = -\log(0.9) \approx 0.105$$
The loss is small, this is good.

Now let us say that the chance two nodes are connected is of:
$$L = -\log(0.1) \approx 2.303$$
Big loss, this is bad.
Therefore the idea we want is:
```
CONNECTED
distance ↓
probability ↑
loss ↓
```

Let us remember our graph, which has:
```
n02 → n01   subtype_of
n03 → n01   subtype_of
n04 → n01   subtype_of
n05 → n02   subtype_of
n06 → n02   subtype_of
n07 → n03   subtype_of
n08 → n03   subtype_of
n09 → n04   subtype_of
...
```

We will not use the entire graph, because for now we are just learning the topic, this is why we will use a 10-nodes toy tree.

![[Pasted image 20260915095626.png|598]]

Now I will tell you something surprising... The loss is a `struct`. 
This means that we don't write:
```julia
function fermi_dirac_loss(
	r::T,
	t::T,
) where {T <: AbstractFloat}
end
```

We will write:
```julia
struct FermiDiracLoss{T}
	radius::T
	temperature::T
end
```

But why? Why do we do a `struct` instead of a function?

Let us remember the MSE (Mean square error). It wasn't the only existing loss, we had much more losses, we had the BCE (Binary-Cross Entropy), Huber Loss, and many more.

This why we will do a `struct`, so later we can have:
```
FermiDiracLoss
HyperbolicInfoNCE
SomeOtherLoss
```

Now we have another question... which gradient are we actually calculating? 
Let us suppose that we have this two embeddings:
```
x = embedding of node A
y = embedding of node B
```

Their loss will depend on $d$:
$$L = L(d(x,y))$$
So the gradient will tell us which direction shall `x` and `y` move to make their Loss decrease.
For positive edges:
```
x ●────────────● y
       far

        ↓ gradient

x ●────● y
    closer
```

We know that doing:
```
x -= lr * nabla_x 
```

would be wrong in a hyperbolic space. This is why we will usually follow this idea:
```
ambient gradient
       ↓
project_tangent
       ↓
tangent gradient
       ↓
exp_map
       ↓
new valid hyperboloid point
```

Now let us suppose we have: `d = 2, r = 1, t = 0.5`.
$$p = \frac{1}{e^2 + 1} \approx 0.119$$
Let us say that the graph is connected:
$$y = 1$$
Therefore:
$$L = -\log(0.119) \approx 2.128$$
This is bad.

What if the optimization will move them closer?
$$d = 0.5$$
We get:
$$p \approx 0.731 \implies L = -\log(0.731) \approx 0.313$$
Much better!

Now we move them even closer:
$$d = 0.1 \implies p = \frac{1}{e^{-1.8} + 1} \approx 0.858 \implies L = -\log(0.858) \approx 0.153$$

So we get:
```
distance       probability       loss

2.0              0.119           2.128
0.5              0.731           0.313
0.1              0.858           0.153
```

So we will do:
```julia
struct FermiDiracLoss{T}
	radius::T
	temperature::T
end
```

and the production style interface will look like:
```julia
compute_loss_and_grad(
    loss,
    embeddings,
    edges,
)
```

So, conceptually it will look like:
```
FermiDiracLoss
       │
       │
       ▼
compute_loss_and_grad(...)
       │
       ├── loss
       │
       └── gradients
```

But as we already can imagine, the loss doesn't know how to train our model, it can only tell how bad it is.
Because if we want to update the results, we will go by this pipeline:
```
FermiDiracLoss
     │
     │ "How bad are these embeddings?"
     │ "What gradient would reduce that?"
     ▼
gradient
     │
     ▼
RSGD
     │
     │ "How do I move points correctly
     │  on the hyperboloid?"
     ▼
Exp map
     │
     ▼
new embeddings
```

Now we can make the first steps, as:
```julia
struct FermiDiracLoss{T}
	radius::T
	temperature::T
end

loss = FermiDiracLoss(
	1,
	0.5,
)
```

So, for now we have:
```
loss
├── radius      → 1.0
└── temperature → 0.5
```

Now we can write the final function:
```julia
function pair_loss(
	loss::FermiDiracLoss,
	M::AbstractManifold,
	x,
	y,
	label::Int,  # The label is 1 or 0. So we know if connected or no.
)

	d = distance(M, x, y)
	z = (d - loss.radius)/loss.temperature
	
	s = label == 1 ? z : -z
	
	return log1pexp(s)
end
```

The idea behind `s` is easy, since it simply means:
```
condition ? if true : if false
```

So our idea:
```
s = label == 1 ? z : -z
```

translates as:
```
if label is equal to 1, then s = z, otherwise s = -z
```

This is the whole idea of the Fermi-Dirac loss.
Now I will introduce the next loss. After it, we will finally do our real implementations with the RSGD.

## 4. Hyperbolic InfoNCE Loss

This is another type of loose we will use. But it is different from the Fermi-Dirac loss, because the Fermi Dirac loss looks at two nodes and tell the probability of them being connected, for example:

We have to pairs `(n02, n05)`. The loss will tell us how likely is it that these two nodes are connected thanks to their distance.

While the InfoNCE Loss takes an anchor and a group of candidates, as:
```
              n05  ← positive
             /
anchor n02 ──┼── n06  ← negative
             \
              n09  ← negative
```

And ask a question as: "Among all this candidates, which shall be closest to `n02`?"

So we understood the main difference. A full example would of both, starting from Fermi-Dirac; suppose that we have:
```
anchor = n02

positive = n05
negative = n09
```

Fermi-Dirac can indipendently evaluate:
```
n02 ↔ n05
```

and
```
n02 ↔ n09
```

So it will get something as:
```
d(n02,n05) = 0.4
d(n02,n09) = 1.8
```

So it goes by the idea:
```
n02 ↔ n05
     ↓
close
     ↓
GOOD


n02 ↔ n09
     ↓
far
     ↓
GOOD
```

While the InfoNCE Loss would choose the anchor and look at the candidates:
```
anchor: n02

candidates:

n05
n06
n09
n10
```

Then checks their distance:
```
n02 → n05    0.4
n02 → n06    0.7
n02 → n09    1.8
n02 → n10    2.1
```

After evaluating each's distance, it will simply say: 
"Among these candidates, `n05` should receive the highest similarity"

Normally the InfoNCE is written using similarity:
$$L = -\log \frac{e^{s(x, y^+)/\tau}}{\sum_j e^{s(x, y_j)/\tau}}$$
But the formula is not hard at all, because it simply says:
```
                    positive similarity
                    e^(similarity / temperature)                 
                    ─────────────────────────
                    all candidate similarities
```

We can understand that:
```
small distance → high similarity
large distance → low similarity
```

To make it fit perfectly out hyperbolic embedding, we will use:
$$L = -\log \frac{e^{-d(x, y^+)/\tau}}{\sum_j e^{-d(x, y_j)/\tau}}$$

Now let us choose the pair:
```
anchor = n02

positive:
n05 → distance 0.5

negatives:
n06 → distance 1.5
n09 → distance 2.0
```

Wait... From where we know which the positive value is?
Well, we use the graph to choose it. Because our graph contains ideas as `subtype_of`, `asociated_with`. This will tell us which will be the positive. 

Using the idea of: 
$$s = -d \hspace{2.5cm} \text{since we changed `s` for `-d`}$$
We will get:
```
n05: -0.5
n06: -1.5
n09: -2.0
```

Now, assuming that $\tau = 1$, we will get:
```
e^-0.5 ≈ 0.607
e^-1.5 ≈ 0.223
e^-2.0 ≈ 0.135
```

To which the total is:
$$0.607+0.223+0.135=0.965$$
Therefore, the positive probability is:
$$\frac{0.607}{0.965} \approx 0.629$$
So InfoNCE says:
"`n05` currently has about 63% of the probability mass. Not terrible, but we can improve."

The training will lower the distance of `n05`, while raising the distance to `n06` and `n09`.

So:

- Fermi-Dirac Loss → What is the chance this two nodes are close to each other?

- InfoNCE Loss → What candidate is the correct one for this anchor?

This is the whole idea of both.

In a Julia implementation this would look like:
```julia
abstract type AbstractEmbeddingLoss end

struct FermiDiracLoss <: AbstractEmbeddingLoss
	radius::Float64
	temperature::Float64
end

struct HyperbolicInfoNCE <: AbstractEmbeddingLoss
```

And in the future we may do:
```
                ┌── FermiDiracLoss
                │
Embedding ──────┼── HyperbolicInfoNCE
                │
                └── future loss...
```

Since we can swap and use which loss we need.

Now the hell starts, because we got to one of the last topic.

## 5. RSGD Graph Optimization Loop

We are not going to implement the raw function for the RSGD, since the `Manopt.jl` already provides Riemannian gradient descent and stochastic gradient descent.

We will try to make a small project work, by using all we will learn.

But firstly, I will tell that the structure has to look like this:
```
   Data
     │
     ▼
Nodes + relationships
     │
     ▼
positive / negative training pairs
     │
     ▼
HyperboloidEmbedding
     │
     ▼
Fermi-Dirac OR Hyperbolic InfoNCE
     │
     ▼
Riemannian objective
     │
     ▼
Manopt.jl
     │
     │  RSGD
     ▼
updated hyperbolic embeddings
     │
     ▼
Poincaré representation
     │
     ▼
visualization / downstream use
```

Yabai? Absolutely.

Now let us think, what happen from the start? 
We get the data we need, then we extract their relationship.
After this we will create the rest... as the structure of our embedding, the losses... but even if we get the loss of our model... who is going to fix it?

The loss will produce a gradient that will tell us:
"Changing these embedding this way will reduce the loss"

While the RSGD will perform this manifold-aware update.

We already know what the RSGD solves, since I spoke about it at least 4 times. 

In poor words:

The normal vanila gradient descent fails in a hyperbolic space, because it will genuinely make our point get out of the manifold. But we want the gradient descent to respect the geometry, this is why we will use the Riemannian Stochastic Gradient Descent, which is:
$$x_{new} = Exp_x(-\eta\nabla L)$$
Now, let us return to our graph. One thing we might notice is that:
```
	{
      "name": "Sertraline",
      "type": "drug",
      "id": "n32"
    },
```

it actually: 
```
n32 ──targets──> n17
```

and
```
n32 ──treats──> n05
n32 ──treats──> n07
n32 ──treats──> n08
```

This is no longer a simple:
```
n01 --- n02 --- n03
```

This is a heterogeneous graph already! (Which we will learn next month only). 

The basic idea of a homogeneous graph (a simple graph) and a heterogeneous graph is that:

- Homogeneous graph - it has one node type (e.g. `users`) and one edge type (e.g. `friend_with`)

- Heterogeneous graph - it has more node types (e.g. `Patient`, `Disease`, `Drug`) and more edge types (e.g. `diagnosed_with`, `treated_by`).

Soooo, the code bellow will be so hard that you will surely not understand anything, this is why I will make a 10k + characters yap bellow the code, so you can understand it.
Genuinely, I will break the code in manyyyy pieces, because you don't know the horror... Sooo, there are always two options:
1. Understand the whole idea - which you will probably do not.
2. Learn it by heart - which you will, even if un-recommeded by the author.

Now we start... but the deeper we go in the rabbit hole, the harder it will get.

# Full project implementation and breakdown

Before we start I will explain the ideas of relations.
![[Pasted image 20260917203151.png]]

We will start slowly.

`subtype_of (Hierarchical Classification)`: A directed branching network showing taxonomy. Arrow 1 $\rightarrow$ 4 means entity 4 is a specific sub-category of entity 1 (e.g., Viral Infection $\rightarrow$ Influenza (The viral infection is the parent, while the influenza is the child)).

`part_of_pathway (Sequential Process)`: This are the processes. For example, let us take the mitosis. We have:
```
n17 - Cdk1
n21 - Securin
n18 - Tubulin

n27 - Mitosis
```

This three processes will be `part_of_pathway` for `n27`, because they are the processes needed to make the Mitosis.

`associated_with (Mutual Correlation)` - this edge has no direction, so it means that if `n17` is strongly `associated_with` `n27`, then `n27` is mutually `associated_with` `n17`. For example:
```
n17 - APOE-ε4
n27 - Alzheimer's Disease
```

so if we see `n17 associated_with n27`, we can understand that people carrying APOE-ε4 statistically have a higher risk of developing Alzheimer.

This doesn't mean that `n17` is the `part_of_pathway` for Alzheimer. It has a strong correlation, and that it.

`targets (Molecular Binding)` - the `targets` relation maps physical interactions where a drug or molecule binds or modulates a specific biological target, such as an enzyme, a protein, or a receptor.

For example:
```
n3 - Ibuprofen : Drug
n5 - COX-2 : Target
```

When we take Ibuprofen, the ibuprofen relive the pain and reduce the temperature. How? Does it kill the Prostaglandin E2 (which make you get fever)? Nope.

The Ibuprofen targets the COX-2, so it blocks this enzyme from producing the inflammatory chemical. As we noticed, it doesn't `treat` nothing, it `targets` an enzyme and blocks its function.

Same for Caffeine and Adenosine Receptor. The Caffeine targets the Adenosine receptor by physically blocking tiredness signals from binding.

`treats (Therapeutic Action)` - This is basically the `drug/intervention` that treats the `disease/condition`. It records that an intervention produces a therapeutic benefit, symptom relief, or cure for a specific disease in population. For example:
```
n9 - Aspirin 
n31 - Acute Myocardial Infarction (Heart attack)
```

The `Aspirin` treats the `Heart Attack`, by preventing further clot formation in the middle of a `Heart Attack`.

`biomarker_for (Diagnostic Indicator)` - This represents a directed diagnostic to a disease or problem from a measurable biological indicator (Proteins, genetic mutations, and so on...). While `associated_with` shows us a broad statistical correlation, the `biomarker_for` will serve as a validated, measurable clinical proxy. For example:
```
n38 - Troponin I 
n12 - Myocardial Infarction
```

A elevated level of `n38` serves as a direct proof of an active cardiac muscle damage during an acute cardiac event .

So, it goes only by: 
`Measurable molecule → Disease/Trait`

`comorbid_with (Condition Co-occurrence)` - this represents an undirected, symmetric co-occurrence of two diseases. This means that two diseases, medical conditions, or disorders frequently exist together in the same patient population. There is no direction, because if there is one, there is the other - they are mutual. But that doesn't mean that they are forced to exist together, but if there is one disease, there is a risk for the latter. For example:
```
n26 Type 2 Diabetes
n29 Hypertension
```

If a patient has `n26`, then `n26` can co-occure, due to the shared metabolic factors, arterial stiffens, and so on...

Same idea for `major depressive disorder` and `anxiety disorder`. If a patient has a `major depressive disorder`, then there are elevated chances it may have even the `anxiety disorder`.

`implicated_in (Pathological Causation)` - this shows the risk factor from a genetic variation, protein mutation, or environmental factor to a specific disease or pathological outcome. 
Unlike statistical co-occurences which provide chances, the `implicated_in` provides a active role in the progression of the disease.
```
n7 HTT Gene Mutation
n33 Huntington's Disease
```

The HTT Gene Mutation is directly implied, why? because abnormally expanded CAG repeat sequence in the HTT gene directly causes toxic protein aggregation and neurodegeneration.

Same for the HIV. 
`HIV implicated_in AIDS`, because this pathogen actively drives the onset of the immunodeficiency syndrome.

So, we understand that:

| Edge Type             | Direction                      | Source → Destination                         | HIV / Clinical Example                                |
| :-------------------- | :----------------------------- | :------------------------------------------- | :---------------------------------------------------- |
| **`implicated_in`**   | Directed ($\rightarrow$)       | Pathogen / Risk Factor $\rightarrow$ Disease | `HIV` $\rightarrow$ `AIDS`                            |
| **`biomarker_for`**   | Directed ($\rightarrow$)       | Measurable Marker $\rightarrow$ Disease      | `CD4+ Count` $\rightarrow$ `AIDS Progression`         |
| **`treats`**          | Directed ($\rightarrow$)       | Drug / Therapy $\rightarrow$ Disease         | `Biktarvy` $\rightarrow$ `HIV Infection`              |
| **`targets`**         | Directed ($\rightarrow$)       | Drug / Molecule $\rightarrow$ Target Protein | `Dolutegravir` $\rightarrow$ `HIV Integrase`          |
| **`associated_with`** | Undirected ($\leftrightarrow$) | Entity A $\leftrightarrow$ Entity B          | `HIV Infection` $\leftrightarrow$ `HLA-B*5701 Allele` |
| **`comorbid_with`**   | Undirected ($\leftrightarrow$) | Disease A $\leftrightarrow$ Disease B        | `HIV Infection` $\leftrightarrow$ `Tuberculosis`      |
| **`part_of_pathway`** | Directed ($\rightarrow$)       | Gene / Step $\rightarrow$ Pathway            | `Reverse Transcription` $\rightarrow$ `HIV Lifecycle` |
| **`subtype_of`**      | Directed ($\rightarrow$)       | Sub-Category $\rightarrow$ Parent Category   | `HIV-1` $\rightarrow$ `Lentivirus`                    |

---

Now we will start with the, code. Where I will use 1 or 2 libraries you never saw, because I didn't have them on the study plan till now. This is why I will genuinely try to explain everything.

```julia
# ======================================= 
# ======= IMPORTING THE LIBRARIES =======
# ======================================= 
using LinearAlgebra
using Manopt
using Manifolds
using Random
using JSON3
using LogExpFunctions
using ManifoldDiff: grad_distance

# =======================================
# =======   EXTRACTING THE GRAPH  =======
# =======================================
data = JSON3.read(read("/home/<username>/Downloads/graph.json", String))

nodes = collect(data.nodes)
edges = collect(data.edges)

node_ids = [
    String(node.id) for node in nodes
]

node_to_idx = Dict(
    node => i for (i, node) in enumerate(node_ids)
)

# =======================================
# ===== EXTRACTING THE RELATIONSHIP =====
# =======================================
struct GraphEdge
    source::Int
    target::Int
    relation::Symbol
end

training_edge = GraphEdge[
    GraphEdge(
        node_to_idx[String(edge.source)],
        node_to_idx[String(edge.target)],
        Symbol(edge.relation),
    )
    for edge in edges
]
```

Let us stop and explain what we did.

Firstly we imported all the libraries we needed. As we noticed, we don't need any `CSV`, because we have the graph and it already gives us everything we need. 

Then we extracted the data from external sources and putted them into a variables that we will use later.

And in the end we extracted the edges. Let me explain how.

Let us remember a part of our edges:
```
n02	n01	subtype_of
n03	n01	subtype_of
n04	n01	subtype_of
n05	n02	subtype_of
n06	n02	subtype_of
n07	n03	subtype_of
n08	n03	subtype_of
n09	n04	subtype_of
n11	n10	subtype_of
n12	n11	subtype_of
n13	n11	subtype_of
n14	n10	subtype_of
n15	n10	subtype_of
n17	n27	part_of_pathway
n21	n27	part_of_pathway
n18	n27	part_of_pathway
n18	n28	part_of_pathway
n19	n29	part_of_pathway
n19	n31	part_of_pathway
```
Let us see directly what is happening.
We will get the `node_id`, which is:
```julia
nodes_id = [
    "n01", "n02", "n03", "n04", "n05", "n06", "n07", "n08", "n09",
    "n10", "n11", "n12", "n13", "n14", "n15",
    "n17", "n18", "n19", "n21",
    "n27", "n28", "n29", "n31"
] 
```

then we transform the ids into an dict with id -> idx.
```julia
node_to_idx = Dict(node => i for (i, node) in enumerate(nodes_id))

# which will be equal to:
node_to_idx = Dict(
    "n01" => 1,
    "n02" => 2,
    "n03" => 3,
    "n04" => 4,
    "n05" => 5,
    # … 
    "n27" => 20,
    "n28" => 21,
    "n29" => 22,
    "n31" => 23
)
```

Now we are going to convert to edges, but firstly we make the structure we will use:
```julia
struct GraphEdge
    source::Int
    target::Int
    relation::Symbol
end

training_edges = GraphEdge[
    GraphEdge(
        node_to_idx[String(edge.source)],
        node_to_idx[String(edge.target)],
        Symbol(edge.relation)
    )
    for edge in edges
]
```

I will explain each part.
- `training_edges = GraphEdge[...]` - this creates a new vector, which element type is forced to be `GraphEdge`. Everything inside the brackets is evaluated and colected inside the vector.

- `node_to_idx[String (edge.source)]` - we know that inside it is like:

```julia
"n01" => 1,
"n02" => 2,
"n03" => 3,
...
```

so the process is:
```julia
node_to_idx[String(edge.target)]
```

And the first iteration will be:
```julia
node_to_idx["n01"]  →  1
```

So it checks the node id and give the index.

The full process will take our edges (which is `n02	n01	subtype_of`) and turn them into:
```
training_edges = [
    GraphEdge(2,  1,  :subtype_of),   # n02 → n01
    GraphEdge(3,  1,  :subtype_of),   # n03 → n01
    GraphEdge(4,  1,  :subtype_of),   # n04 → n01
    GraphEdge(5,  2,  :subtype_of),   # n05 → n02
    GraphEdge(6,  2,  :subtype_of),   # n06 → n02
    GraphEdge(7,  3,  :subtype_of),   # n07 → n03
    GraphEdge(8,  3,  :subtype_of),   # n08 → n03
    GraphEdge(9,  4,  :subtype_of),   # n09 → n04
    GraphEdge(11, 10, :subtype_of),   # n11 → n10
```

now we continue with the second part of the code, which is our embedding struct and the `near_origin_initialization`

---
```julia
# =======================================
# ===== CREATING HYPERBOLOID STRUCT =====
# =======================================
mutable struct HyperboloidEmbedding{T}
    manifold::Hyperbolic
    node_ids::Vector{String}
    node_to_idx::Dict{String, Int}
    embedding::Matrix{T}
end


function near_point_initialization(
    M:: Hyperbolic,
    rng::AbstractRNG;
    scale::T = T(0,01),
) where {T <: AbstractFloat}
    
    d = manifold_dimension(M)
  
    spatial = scale .* randn(rng, d)

    x0 = sqrt( 1.0 + dot(spatial, spatial))

    return [x0; spatial]
end

function HyperboloidEmbedding(
    node_ids::Vector{String},
    d::Int,
    rng:: AbstractRNG = Random.default_rng();
    scale::T = T(0.01),
) where {T <: AbstractFloat}

    M = Hyperbolic(d)

    node_to_idx = Dict(node => i for (i, node) in enumerate(node_ids))

    X = Matrix{T}(undf, d + 1, length(node_ids))
    
    for i in axes(X, 2)
	    X[:, i] = near_point_initialization(M, eng, scale = scale)
	end
	
	return HyperboloidEmbedding(M, node_ids, node_to_idx, X)
end
```

We already know this part and what it does.
In this part we will simply make a structure of our Hyperboloid embedding and then make a function that casually place the points near the origin.

Now we will continue with the Loss.

---
```julia
# =======================================
# =====   CREATING OUR LOSS STRUCT  =====
# =======================================
struct FermiDiracLoss{T}
    radius::Dict{Symbol, T}
    temperature::Float64
end

function relation_radius(
    loss::FermiDiracLoss,
    relation::Symbol,
)
    return get(loss.radius, relation, 1.0)
end

function pair_loss(
    loss::FermiDiracLoss,
    M::AbstractManifold,
    x,
    y,
    relation::Symbol,
    label::Int,
)
    d = distance(M, x, y)

    radius = relation_radius(loss, relation)

    z = (d - radius) / loss.temperature

    s = label == 1 ? z : -z

    return log1pexp(s)
end
```

Wait...? Why do we use even the `symbols` now? We use them because we genuinely care about the relations, since we have a heterogeneous graph, not simply a homogeneous one. 

So we may have:
```julia
radius = Dict(
    :treats => 1.0,
    :targets => 1.2,
    :associated_with => 1.8,
    :subtype_of => 0.7
)
```

This will tell which relation will make the node be closer or further.

For example:
```
Sertraline ── treats ──> MDD
```

We want Sertraline to be closer to MMD, since Sertraline treats MMD. This is why we may write:
```julia
radius = Dict(
    :treats => 1.0,
    :targets => 1.0,
    :associated_with => 1.8,
    :subtype_of => 0.7
)
```

So each `treats` will be relatively close to the target.

Now the part of `return get(loss.radius, reltion, 1.0)`.
Let us assume that:
```julia
loss.radius = Dict(
    :subtype_of      => 0.8,
    :part_of_pathway => 1.4
)
```

Now we will get:
```julia
relation_radius(loss, :subtype_of)       # → 0.8
relation_radius(loss, :part_of_pathway)  # → 1.4
relation_radius(loss, :some_other_rel)   # → 1.0   (we set default at 1.0)
```

And the last part is our loss.

Now we continue with sampling the negative edges.

---
```julia
# =======================================
# =====   SAMPLING NEGATIVE EDGES   =====
# =======================================
function sample_negative_edges(
    positive_edges::Vector{GraphEdge},
    n_nodes::Int,
    rng::AbstractRNG,
)
    positives = Set(
        (
            edge.source,
            edge.target,
            edge.relation,
        )
        for edge in positive_edges
    )
    negatives = GraphEdge[]
    for edge in positive_edges
        while true
            target = rand(
                rng,
                1:n_nodes,
            )
            target == edge.source && continue
            candidate = (
                edge.source,
                target,
                edge.relation,
            )
            candidate in positives && continue
            push!(
                negatives,
                GraphEdge(
                    edge.source,
                    target,
                    edge.relation,
                ),
            )
            break
        end

    end

    return negatives

end
```

Let us understand what this code does.

```julia
function sample_negative_edges(
    positive_edges::Vector{GraphEdge},
    n_nodes::Int,
    rng::AbstractRNG,
)
```

This function wants to create a fake edge for each existing one.
Each fake edge has:
```
same source and relation of the original graph,
DIFFERENT target,
does not exist as a real edge.
```

Now we loop over each existing edge:
```julia
positives = Set(
    (
        edge.source,
        edge.target,
        edge.relation,
    )
    for edge in positive_edges
)
```

This will create new triple (e.g. `(15, 13, :subtype_of)`).
The triples will be placed in a `Set`.

Why a Set? Because it reaches unreal speeds when tell if that triple exist.

```julia
negatives = GraphEdges[]
```

We prepared the output variable

```julia
target = rand(rng, 1:n_nodes)
```

Let us say we have as edges this:
```julia
positive_edges = [
    GraphEdge(20, 29, :part_of_pathway),
    GraphEdge(20, 30, :part_of_pathway),
    GraphEdge(15, 13, :subtype_of),
]
```

Now, our target may be:
```julia
negatives = [
    GraphEdge(20,  7, :part_of_pathway),
    GraphEdge(20,  17, :part_of_pathway),
    GraphEdge(15, 22, :subtype_of),
]
```

Then we write:
```julia
target == edge.source && continue
```

This will skip self, loops, so if the target has the same number of the source, we will skip it and try again.

Then we look at the candidate:
```julia
candidate = (
    edge.source,
    target,
    edge.relation,
)
```

The candidate is just the full collection.
For example, we write:
```julia
candidate in positives && continue
```

Which means. if we were unlucky enough, we may get a pair that is identical to one of our original pairs.
```julia
# Original edges:
positive_edges = [
    GraphEdge(20, 29, :part_of_pathway),
    GraphEdge(20, 30, :part_of_pathway),
    GraphEdge(15, 13, :subtype_of),
]

# A random candidate:
candidate =
	GraphEdge(20, 30, :part_of_pathway)

# They 2nd edge is identical, so we already have one of this pairs. This is why it will be skipped.
```

The `push!` will append the candidate to `negatives`.

We will use this set for teaching, because it will learn that it has to push the unfamiliar nodes apart.

Now we will continue with the graph loss.

---
```julia
# =======================================
# =====   CREATING OUR GRAPH LOSS   =====
# =======================================
function graph_loss(
    E::HyperboloidEmbedding,
    positive_edges::Vector{GraphEdges},
    negative_edges::Vector{GraphEdges},
    loss::FermiDiracLoss,
)

    M = E.manifold

    total = 0
    count = 0

    for edge in positive_edges
        x = E.embedding(:, edge.source)
        y = E.embedding(:, edge.target)

        total += pair_loss(loss, M, x, y, edge.relation, 1)

        count += 1
    end

    for edge in negative_edges
        x = E.embedding(:, edge.source)
        y = E.embedding(:, edge.target)

        total += pair_loss(loss, M, x, y, edge.relation, 0)

        count += 1
    end

    return total / count
end
```

This function will calculate the average loss of our graph, by doing:
```
graph_loss = (sum of losses on positive edges + sum of losses on negative edges) 
             / (number of positive edges + number of negative edges)
```

The function simply asks: "how bad is the current embedding for this pair?". It will check this idea for each and add the small penalties together.

Imagine we take a small part of our graph:
```julia
positive_edges = [
    GraphEdge(2, 1, :subtype_of),    # Mood Disorders → Mental Health Disorders
    GraphEdge(5, 2, :subtype_of),    # Major Depressive Disorder → Mood Disorders
    GraphEdge(17, 27, :part_of_pathway),  # SLC6A4 → Serotonin Signaling Pathway
    GraphEdge(32, 5, :treats),        # Sertraline → Major Depressive Disorder
]
```

And we get from the `sample_negative_edges`:
```julia
# All are fake conections, they are not true, this is why they are fake.

negative_edges = [
   GraphEdge(2, 10, :subtype_of), # Mood Disorders → Spinal Disorders
   GraphEdge(5, 27, :subtype_of), # Major Depressive Disorder → Serotonin Pathway
   GraphEdge(17, 1, :part_of_pathway),  # SLC6A4 → Mental Health Disorders 
   GraphEdge(32, 11, :treats),    # Sertraline → Degenerative Disc Disease
]
```

we start with the positive label:
```julia
for edge in positive_edges
    x = E.embedding(:, edge.source)
    y = E.embedding(:, edge.target)

    total += pair_loss(loss, M, x, y, edge.relation, 1)

    count += 1
end
```

Let us say that our positive edge is:
```julia
edge = GraphEdge(32, 5, :treats)
# Sertraline (n32) → Major Depressive Disorder (n05)
```

Now we will get each pair embedding (let us say that we choose a 3 dimensions only, for simplicity):
```julia
E.embedding[:, 32] = [0.8, 0.1, 0.3]   # Sertraline
E.embedding[:,  5] = [0.7, 0.2, 0.4]   # Major Depressive Disorder
```

Now we will call our pair loss, which was:
```julia
function pair_loss(
    loss::FermiDiracLoss,
    M::AbstractManifold,
    x,
    y,
    relation::Symbol,
    label::Int,
)
    d = distance(M, x, y)

    radius = relation_radius(loss, relation)

    z = (d - radius) / loss.temperature

    s = label == 1 ? z : -z

    return log1pexp(s)
end
```

Now we pass our embeddings (We try to make it look realistic):
```julia
d = distance(M, x, y)           # ≈ 0.22
radius = relation_radius(loss, :treats)  # = 0.9
temp = loss.temperature            # = 0.5

z = (0.22 - 0.9) / 0.5 = -1.36
s = z                                 # because label == 1
loss_value = log1pexp(-1.36) ≈ 0.23
```

So we will get just:
```
total += 0.23
count += 1
```

For negative edges, we will get:
```julia
edge = GraphEdge(32, 11, :treats)
# Sertraline (n32) → Degenerative Disc Disease (n11)   ← fake
```

The embeddings will be:
```julia
E.embedding[:, 32] = [0.8, 0.1, 0.3]   # Sertraline
E.embedding[:, 11] = [0.1, 0.9, 0.2]   # Degenerative Disc Disease
```

So, we will get:
```julia
d = distance(M, x, y)           # ≈ 1.15
radius = relation_radius(loss, :treats)  # = 0.9
temp = 0.5

z = (1.15 - 0.9) / 0.5 = 0.50
s = -z                                # because label == 0  → -0.50
loss_value = log1pexp(-0.50) ≈ 0.47
```

So, after this two edges, we will get:
```julia
total = 0.23 + 0.47 = 0.70
count = 2
```

Now we can continue with the next part, which is our graph gradient.

---
```julia
# =======================================
# =====   CREATING OUR GRAPH GRAD   =====
# =======================================
function graph_gradient(
	E::HyperboloidEmbedding,
	positive_edges::Vector{GraphEdges},
	negatives_edges::Vector{GraphEdges},
	loss::FermiDiracLoss,
	)
	
	M = E.manifold
	G = zeros(size(E.embedding)) # This is our gradient matrix, it starts at 0
	
	total_edges = length(positive_edges) + length(negative_edges)
	
	for (edges, label) in ((positive_edges, 1), (negative_edges, 0))
		for edge in edges
            i, j = edge.source, edge.target
            x = E.embedding[:, i]
            y = E.embedding[:, j]

            d = distance(M, x, y)
            d < 1e-7 && continue

            radius = relation_radius(loss, edge.relation)

            z = (d - radius) / loss.temperature

            σ = 1 / (1 + exp(-z))
            coeff_d = (label == 1 ? σ : σ - 1) / loss.temperature

            gx = grad_distance(M, x, y)
            gy = grad_distance(M, y, x)

            G[:, i] .+= coeff_d .* gx
            G[:, j] .+= coeff_d .* gy
        end
    end

    G ./= total_edges
    return G
end
```

This code will give us the gradient we need. So now we will have a direction in which to move our embeddings.
Now I will start to explain:
```julia
for (edges, label) in ((positive_edges, 1), (negative_edges, 0))
		for edge in edges
```

This is a really useful way to don't write the same code twice, since firstly it will take:
```julia
for (edges, label) in (positive_edges, 1)
	for edge in edges
	...

# And after finishing, it will run
for (edges, label) in (negative_edges, 1)
	for edge in edges
	...
```

A full numerical example would look like:
```julia
# POSITIVE EDGES:
edge = GraphEdge(32, 5, :treats)
i = 32, j = 5

x = [0.80, 0.10, 0.30]          # Sertraline
y = [0.70, 0.20, 0.40]          # MDD

d = distance(M, x, y) ≈ 0.22
d > 1e-7 → continue processing

radius = 0.9
z = (0.22 - 0.9) / 0.5 = -1.36

σ = 1 / (1 + exp(1.36)) ≈ 0.204     # (correct formula)

gx = grad_distance(M, x, y) ≈ [ 0.41, -0.35, -0.22]
gy = grad_distance(M, y, x) ≈ [-0.41,  0.35,  0.22]

G[:, 32] .+= 0.408 .* gx     # → G[:,32] ≈ [0.167, -0.143, -0.090]
G[:,  5] .+= 0.408 .* gy     # → G[:, 5] ≈ [-0.167,  0.143,  0.090]

# NEGATIVE EDGES:
edge = GraphEdge(32, 11, :treats)
i = 32, j = 11

x = [0.80, 0.10, 0.30]          # Sertraline (same)
y = [0.10, 0.90, 0.20]          # Degenerative Disc Disease

d ≈ 1.15
radius = 0.9
z = (1.15 - 0.9) / 0.5 = 0.50

σ = 1 / (1 + exp(-0.50)) ≈ 0.622

gx ≈ [ 0.55, -0.62,  0.18]
gy ≈ [-0.55,  0.62, -0.18]

G[:, 32] .+= -0.756 .* gx    # adds a push in the opposite direction
G[:, 11] .+= -0.756 .* gy
```

Now, we are going to extract some values from the function.

---
```julia
# =======================================
# =====   EXTRACTING SOME VALUES    =====
# =======================================
rng = MersenneTwister(9)

d = 32

E = HyperboloidEmbedding(nodes_id, d, rng, 0.01)


loss = FermiDiracLoss(
    Dict(
        :subtype_of => 0.8,
        :comorbid_with => 1.5,
        :associated_with => 1.5,
        :part_of_pathway => 1.2,
        :targets => 1.0,
        :treats => 1.0,
        :implicated_in => 1.2,
        :biomarker_for => 1.2,
    ),
    0.5,
)

negative_training_edges = sample_negative_edges(training_edges, length(nodes_id), rng,)

initial_loss = graph_loss(E, training_edges, negative_training_edges, loss)
```

After extracting some values, we will finish it with the RSGD and the final review.

---
```julia
# =======================================
# =====   OUR RSGD TRAINING STEP    =====
# =======================================

function train_embeddings!(
    E::HyperboloidEmbedding,
    positive_edges::Vector{GraphEdges},
    loss::FermiDiracLoss,
    rng::AbstractRNG;
    epochs::Int = 200,
    learning_rate::Float64 = 0.01,
    negative_refresh::Int = 15,
)

    M = E.manifold

    negative_edges = sample_negative_edges(positive_edges, length(E.nodes_id), rng,)

    for epoch in 1:epochs
        if epoch % negative_refresh == 0
            negative_edges = sample_negative_edges(positive_edges, length(E.nodes_id), rng,)
        end

        G = graph_gradient(E, positive_edges, negative_edges, loss)

        for i in axes(E.embedding, 2)
            x = @view E.embedding[:, i]
            g = @view G[:, i]

            tangent_g = project(M, x, g)

            step = -learning_rate * tangent_g
            new_x = exp(M, x, step)

            E.embedding[:, i] .= new_x
        end

          if epoch == 1 ||
           
            epoch % 20 == 0
            current_loss =
                graph_loss(
                    E,
                    positive_edges,
                    negative_edges,
                    loss,
                )

            println(
                "Epoch ",
                lpad(epoch, 3),
                " | loss = ",
                current_loss,
            )
        end
    end

    return E
end

# ======================================= 
# =======      FINAL  EXPORT      =======
# ======================================= 
train_embeddings!(
    E,
    training_edges,
    loss,
    rng;
    epochs = 1000,
    learning_rate = 0.01,
    negative_refresh = 15,
)

final_negative_sampling = sample_negative_edges(training_edges, length(nodes_id), rng)

final_loss = graph_loss(E, training_edges, final_negative_sampling, loss)
```

This is the code we had to do... now I will simply try to explain what we do from the first step to the last - in few words.

```
The first thing we will do is to extract the graph.json, and after exctracting it, we will take the nodes and edges. After this step we willget all the nodes id and tranform it in the index. As soon as we get the index, we make a structure called GraphEdges, and a variable called training_edges (which will be used in the 'positive_edges' placeholder). Then we create our mutable struct for the HyperboloidEmbedding, the near point initialization function, and the full HyperboloidEmbedding function. 

This was our first part, the second is the Loss and much more.
Now we will create the loss structure and the function for our relation_radius and for our pair_loss. Then we create our first negative sampling function and then we create the graph loss function.

Afterwards we finish by creating the graph gradient, extracting the values, and training the model. 
```

Something as:

Extract the `graph.json` into a variable → Get the nodes and edges → Extract all node IDs and map them to indices → Create the `GraphEdges` structure → Build the` training_edges` list (used as positive edges) → Create the mutable` HyperboloidEmbedding` struct → Write the `near-origin initialization` function → Write the full `HyperboloidEmbedding` constructor → Create the `FermiDiracLoss` structure → Write the `relation_radius` function → Write the `pair_loss` function → Write the negative sampling function → Write the `graph_loss function` → Write the `graph_gradient` function → Set random seed, dimension, create the embedding and define the loss → Sample initial` negative edges` and `compute initial loss` → Write the `train_embeddings!` function → Train the model → Compute the `final loss`.

After this mess, we will continue with Word2Vec / DeepWalk / Node2Vec. Which we had to already know, since normal people go by basic -> Hyperbolic Space. But we went straightly as: 
hyperbolic space -> basics

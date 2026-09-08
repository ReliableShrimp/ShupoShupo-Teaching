
Now we will start with this pain... but this are just the basics we need before entering in a more complex topic (Newsflash: Even this will be hard).

Now we start with the first topic.

# Chapter 1. Non-Euclidean Geometry prime

We will firstly start from the non euclidean prerequisites and then with the complex number & rotations groups.

## 1. What a manifold is?

When talking about manifolds everybody would simply state: 
"A manifold is a space that looks flat (local) when you zoom in closely, even though the whole space may be curved (global)."

cute, but that didn't provide much information for a big quantity of people. That is why I will give some examples.

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

That can be written as: $\mathbb{R}^2$ (Since it is 2D)

If you zoom into any point, it will look like a flat plane.

The distance is measured as:
$$ds^2=dx^2+dy^2$$
This is why the shortest path of point A to point B is simply a straight line.

Sooo, before we say that flat = a line, i will say: 
"flat" - take a normal sheet of paper 

![[Pasted image 20260823153743.png]]

If we draw two points on that sheet of paper, its shortest distance (of point A to point B) can be described by a straight line.

This is what flat means.

Now we will draw a small neighborhood around A:

```
        ┌───────────┐
        │           │
        │     • A   │
        │           │
        └───────────┘
```

Here you can move left, right, forward, backward. Nothing about the geometry itself changes.
Basically, this is euclidean geometry 

Now imagine taking a basketball, if we put a square sticker on the basketball, we can see something clearly, the basketball is curved, but the sticker (if sufficiently small), it will look almost like an ordinary flat square and the geometry inside that tiny region will be approximately like the geometry of a flat sheet

The words neighborhood may seem easy, but it is not. Imagine that we have a point:
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

The circle around our point P is the neighborhood. It doesn't have to be a circle mathematically, I just gave you the idea of what a neighborhood is.

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

If we make it smaller and smaller, then its shape becomes increasingly similar to a piece of a plane. So it is the euclidean space! 

No it is not, because a tiny piece of a sphere can behave like a piece of euclidean space, without the whole spehere being the euclidean space. Think about a huge curved road, if we zoom only to the next 10 cm, i may look straight, but that doesn't mean that the entire road is straight, we are simoly looking at such a small section that the curvature is difficult to notice.

Imagine you are in Rome, you look around and see:
```
        horizon

────────────────────────
```

Even if the ground beneath looks flat, the earth isn't flat at all, it is curved, and that a great distinction:

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

So locally it means that you look at a small neighborhood, while the global means that you look at the whole shape.

So the idea of the manifold is that every point has some sufficiently small neighborhood that behaves like ordinary Euclidean space.

So just because something looks like euclidean space while zoomed in, it doesn't mean that it is euclidean space.

And there are two types.

The manifold (We can place a point, place the neighbor and it looks like a euclidean space if zoomed in. You can draw lines, make perpendicular lines, make right angles, whatever you want, as a euclidean space.)

![[Pasted image 20260823154039.png|664]]
This is a manifold. As we can notice, we placed the point and the neighborhood, and if we zoom, it looks like an euclidean space.

The nonmanifold (One or more points can't be represented in the euclidean space - this are nonmanifold.)
![[Pasted image 20260823154237.png|493]]
As we can see, if we place a point at the parts that intersect and then the neighborhood, you will not be able to get the euclidean space from it, so this is a nonmanifold.

Why do we call it a manifold? 
Because we want a space that mathematically we can represent with two coordinates.

For example, the earth can be described by two coordinates
$$(\text{latitude, longitude})$$
Now imagine a sphere:
![[Pasted image 20260823155543.png|523]]

Even if it is in 3D, we can describe its surface in 2D.
$$S^2 ⊂ \mathbb{R}^3$$
That a really interesting concept.

So, no matter what, the surface of the manifold can be represented in a euclidean space (But sometimes it may distort the reality of the result).

So a manifold is a shape to which all points can be represent locally by the euclidean space while the global may be abstract, complex, or curved.

Now we can go forward.

## 2. Geodesics

A geodesic is the curved-space analogue of a straight line. This is the whole idea, but as we can understand - math can speak 20 pages about a line and don't get tired.

But what exactly is that straight line?
Imagine we have two point and connect them with a straight line:
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

This path will be longer than a straight line, this is why in Euclidean geometry:
$$\text{Straight line} = \text{Shortest distance between points}$$
This already gave us some intuition for the geodesics.

Now let us go back to earth and Airi (A) wants to visit Sheena (B), but since the world is not flat, they will not be able to go in a straight line, because they have to pass through the sphere (earth).

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

They shall stay on the surface, this is why we introduce a new concept: 
Geodesics - The shortest path from a point to another on a curved space.

Let us take the earth once again, we need to find the local shortest path between two points.

But before we went full straight on the idea:
$$\text{Geodesics = shortest path}$$
We have to understand this:
$$\text{Geodesics} \neq \text{The shortest global path}$$
$$\text{Geodesics} \approx \text{The shortest local path}$$

What do I mean with this? Let's take one idea per time, so the first idea that I will explain is the shortest local.

1. $\text{Geodesics} \approx \text{The shortest local path}$:

Imagine we take point A and point B, these two points are extremely close to each other on the surface, then the geodesics is for sure the shortest path. And since we don't right or left in the small neighborhood (Which is really similar to out simple euclidean space, which means that a straight line is the shortest path) line is the most direct road from a point to another.

2. $\text{Geodesics} \neq \text{The shortest global path}$:

On a curved space, if you keep walking straight along the geodesic you might end up looping or take the wrong turn. 
It has a simple rule, the global shortest path must be a geodesic, but not all the geodesics are the global shortest paths.

On a sphere the geodesic is a great circle, so let us say that we have to go from Ecuador (A) to Kenya (B) which are roughly 12000 km.

- Local Shortest Path: Walk $12,000\text{ km}$ directly East along the Equator. 
- The Geodesic: Start from point A, by walking to straight to west along the equator, and just like this, after $28,000 \text{  km}$   you will end up at point B. And this is a perfect valid Geodesic, even if it is $16000 \text{  km}$ longer than the direct path. 

Both are geodesics. Yet as we said before, we can't go in a straight line, because we will pass through the earth core. And the rule worked, we can have more geodesics but not all are the shortest paths. 

![[Pasted image 20260823191322.png|523]]

This is how the geodesics looks on a sphere (which is still a manifold). We have two geodesics, yet one is bigger and the other is smaller, even if theoretically we followed one direction (for one west and the other est)

Each geodesics is different based on space, fro example:

1. On the Euclidean space - the geodesic is a straight line
2. On the Spherical space - the geodesic is a great circle
3. On the Hyperbolic space - the geodesics are curves that differ based on the shape.

Now we will start with formulas! Since letting them for later is not funny.

So the general metric arc-length formula is:
$$L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t), \dot{\gamma}(t))} \, dt$$
There is only a correct reaction -> Nah, I am skipping this. And that the right one! But not so quickly, because if you do so, what will you do when we will learn:
$$\text{Einstein}(x_1, \dots, x_N; w) = \frac{\sum_{i=1}^N w_i \gamma_i x_i}{\left(1 + \sqrt{1 - c \left\Vert{} \sum_{i=1}^N w_i \gamma_i x_i \right\Vert{}^2}\right) \sum_{i=1}^N w_i \gamma_i} \quad \text{where } \gamma_i = \frac{1}{\sqrt{1 - c\Vert{}x_i\Vert{}^2}}$$

So we would rather start slow and understand easy formulas before going toward hell.

Now we will start with the arc-length again. We will break it down into components:
1. $\gamma(t)$ - this is the position vector, for example, if we start at:  $\gamma(t) = (t^2, 3t)$, after two seconds, the position will change: $\gamma(2) = (4, 6)$, on a 2D space we went 4 units to the right and 6 up.  
- $\gamma(t)$ - this tells us where we are at every moment. In a simple 2D space it would be:
$$\gamma(t) = (x(t), y(t))$$

But for now, let us say that
$$\gamma(t) = (t, t^2)$$
That means:
```
t = 0  →  (0,0)
t = 1  →  (1,1)
t = 2  →  (2,4)
...
```

So, we can understand that $\gamma(t)$ reveals our position in space over time by describing its path through space.

And here we will see many times this idea:
$$a \leq t \leq b$$

Where $a$ is the position where we start and $b$ is the position where our journey ends.

- $\dot{\gamma}(t)$ - this is basically: $\frac{\partial\gamma}{\partial t}$. So if our gamma is:
$$\gamma(t) = (x(t), y(t))$$
then it will become:
$$\dot{\gamma}(t) = (x'(t), y'(t))$$

this is what we call our velocity vector.

let us take our old values:
$$\gamma(t) = (t, t^2)$$

Now they will become
$$\dot{\gamma}(t) = (1, 2t)$$

and this is our velocity vector. each of them will start from the point.

And now we will discover the metric $g$
- $g$ - now we will stay in the euclidean space, just to give you this example. 
Let us take a vector:
$$v = (3,4)$$
Now we will find ita magnitude.
$$||v|| = \sqrt{3^2 + 4^2 = 5}$$
But now we will rewrite it using the identity matrix ($I$).
So:
$$v^TIv = \begin{bmatrix} 3 & 4 \end{bmatrix}\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 \\\\ 4 \end{bmatrix}$$
which gives us:
$$v^TIv = 25$$
Therefore:
$$\sqrt{v^TIv} = 5$$

So the euclidean geometry metric is playing the role of the dot product.

So $g_{\gamma(t)}$ is the metric at the current position.
But we have to remember that $g_{\gamma(t)}$ changes, based on our position, but we will discuss later about it.

In the end, I wanted to add that the geodesics is not all about drawing a straight line, because even if we do, sometimes it may not be a geodesics, it always depends on how flat it looks on the space (first 6 minutes of https://youtu.be/m6WY6VtPYrk?is=JgUE4t21J_XVOPCH).

## 3. Curvature

Now we will ask ourselves... what happens when we have two nearby geodesics? This will lead us to the sign of curvatures.

Imagine that two people walk in a straight line next to each other.
```
A ─────────────────────→
B ─────────────────────→
```

What will happen? They will never intersect, their distance will stay constant if they start from 1 meter from each other ($d(t)=1)$. This is why, it has no curvature:
$$K = 0$$
Therefore, this is the flat space and this means even that nearby geodesics neither systematically converge nor diverge.

what is $d(t)$? Imagine that we have two walkers, at time $t$, measures the distance between them, and that distance can change as time passes.
For example:
$d(0) = 1$ (They start at 1 meter apart from each other.)
$d(2) = 0.8$ (After 2 secinds, they are 0.8 meters apart)

So $d(t)$ is the separation distance between two geodesics.

No we will have another hero — Which we will call Mimi, $\dot{d}(t)$ - which is exactly the same as:
$$\dot{d}(t) = \frac{\partial d}{\partial t}$$
In words it means - How quickly the distance between the walkers changes?

Now we have the cooler Mimi — $\ddot{d}$, which is exactly:
$$\ddot{d} = \frac{\partial^2 d}{\partial t^2}$$
this basically mean — "how quickly the rate of separation changes?"

Sooo, basically we have this steps:
$d(t)$ - The distance between the walkers (geodesics) (distance)
$\dot{d}(t)$ - How fast the the distance between the walkers changes? (the change in the distance)
$\ddot{d}(t)$ - How fast the change in the  distance changes (the change of the change in the distance)


Now let us suppose that
$$d(t) = 10 - t^2$$
its first derivative (the $\dot{d}(t)$):
$$\dot{d}(t) = -2t$$

and its second derivative is (our $\ddot{d}(t)$):
$$\ddot{d}(t) = -2$$

So the separation is not merely decreasing its rate of decrease is increasing.

For example:
At $t(1)$

$$\dot{d}(1) = -2$$

At $t(2)$

$$\dot{d}(1) = -4$$

The walkers are getting closer faster and faster.
This is what $\ddot{d}(t) < 0$ is telling us.

But what is $K$, since his symbol is really important! This represents curvature.

This is what it is, and it is useful, because:
```
K > 0  → sphere-like
K = 0  → flat
K < 0  → hyperbolic
```

Now we combine this two ideas:
$$K d(t) = \text{ curvature} \times \text{ current separation}$$

For example, suppose:
$$K = 2$$

and:
$$d(t) = 3$$

So after this formula we get:
$$K d(t) = 2(3) = 6$$

So in the end, the sign of $K$ determines the direction of the effect.

Now we will assemble the equation:
$$\ddot{d}(t) + Kd(t) = 0$$
In some words it is:
$$\text{relative acceleration + curvature × separation = 0}$$
We will not really understand it, so even more simplified it is as:
- `relative acceleration` - how fast the walkers change in distance changes.
- `curvature` - that the physical property of the space that forces nearby parallel to bend together (Only if $K > 0$), pull apart (Only if $K < 0$), and to stay parallel (Only if $K = 0$)
- `separation` - the distance between the two walkers.

And now the final formula we will use, will be:
$$\ddot{d}(t)= -Kd(t)$$
now we have two possible scenarios:

---
- `Scenario №1 `: $K = 0$

In this case the result will be 0, no matter what.
$$\ddot{d}(t) = -(0) \times d(t)$$
$$\ddot{d}(t) = 0$$
This means that the acceleration of the separation is zero.
They lines will stay parallel.
```
A ─────────────────>

B ─────────────────>
```

This is flat space

---
- `Scenario №2 `: $K > 0$

In this case, the result will be negative, no matter what.

Let us say that $K = 1$ and $d(t)=2$
$$\ddot{d}(t) = -(1) \times 2$$
$$\ddot{d}(t) = -2$$
Negative relative acceleration means the separation is being pushed toward smaller values.
$$K>0 ⇒ \text{geodesics tend to converge}​$$
This is sphere-like

---
- `Scenario №3`: $K < 0$

In this case, the result is negative, no matter what.

So let us say $K = -1$ and $d(t) = 2$
$$\ddot{d}(t) = -(-1) \times 2$$
$$\ddot{d}(t) = 2$$
The walkers are being driven apart from each other.
$$K<0 ⇒ \text{geodesics tend to diverge}​$$
This is Hyperbolic behavior.

---

Now we will continue with another topic.

## 4. Why does negative curvature fit hierarchies?

This is what we will work with a lot... so this question is really important.

Firstly we will start with a graph:
![[Manager node.png|553]]

Now we imagine that each node has two nodes more... and so on:

| Depth (d) | Nodes |
| --------: | ----: |
|         0 |     1 |
|         1 |     2 |
|         2 |     4 |
|         3 |     8 |
|         4 |    16 |
|         5 |    32 |

what it looks like? It looks like each new depth the nodes multiply by 2. That is why we get:
$$N(d) = 2^d$$
(The N = nodes. This means that the result of nodes is codependent on the depth)
It has an exponential growth.
This is why the more depths we add the more nodes we get.

At depth 10:

$$3^{10} =59049$$

At depth 20:
$$3^{20} =3486784401$$
This means that the tree becomes enormous very quickly.

But where are those nodes situated geometrically?
Imagine we are placing the root in the middle (The first node):
```
                    •
                 •  •  •
              • • •   • • • 
```

no we have: $d = 1$, $d = 2$, $d = 3$...
As the depth grow, we need more space... Because the deeper we go, the more nodes we need to place around the root.

So... How much space did the Euclidean geometry gave us?

We will draw a circle around the root:
$$C(r) = 2\pi r$$
So if we double the space we get:
$$C(2) = 4\pi r$$
The amount of available boundary grows linearly with $r$.

And what if we compare them? I mean, we add +1 radius for every depth we add... But this is wrong, because one is linear, and the other is exponential. There is a serious mismatch.
Why? 
Because let us say we have 10 depths:
$$N(d) = 2^d$$
As:
$$N(10)=1024$$
And now we compare it to the linear growth:
$$C(10) = 62.8$$
So we are trying to place 1028 objects around a circumference that grow of 63?

What if we have 20 depths?
$$2^{20} = 1048576$$
while:
$$2\pi(20)≈126$$
Now  the difference is ridiculous.

So in the end we can't put too many objects inside the euclidean without ether:

1. Putting related things too far apart from each other
2. Putting unrelated things too close to each other
3. Or requiring enormous Euclidean distances

What about placing them in another space... something like the hyperbolic space, where the boundaries grow like:
$$C(r)= 2\pi \sinh r$$

what is $\sinh r$ equal to? This is equal to:
$$\sinh r \approx \frac{e^r - e^{-r}}2$$
The more the $r$ grows, the smaller $e^{-r}$ becomes. This is why $e^{-r}$ almost doesn't matter, this is why we may write:
$$\sinh r \approx \frac{e^r}2$$
so approximately it may look like
$$C(r)= 2\pi \frac{e^r}2$$
and now we do a high school strategy (simplify the 2):
$$C(r) \approx \pi e^r$$

So where is the difference, now we compare:

Euclidean space:
$$C(r) \sim r$$
Hyperbolic space:
$$C(r) \sim e^r$$
Tree:
$$C(r) \sim b^d$$
They are both exponential. They are related!
Suppose the tree has branching factor $b$.

Then:
$$N(d) = b^d$$Hyperbolic space has approximately:
$$C(r) \sim e^r$$Notice that because $e^{\ln(\text{x})} = \text{x}$, we can express $b^d$ as:
$$b^d = e^{\ln(b^d)}$$Using the logarithm power rule $\ln(x^y) = y \ln x$:
$$e^{\ln(b^d)} = e^{d \ln b}$$And by applying the standard exponential rule $(x^m)^n = x^{m \cdot n}$:
$$e^{d \ln b} = (e^{\ln b})^d = e^{(\ln b)d}$$So:$$\boxed{N(d) = e^{(\ln b)d}}$$

This is the beautiful mathematical proof that states:

A hierarchy naturally says:
- "The farther I go from the root, the more branches I need."

Hyperbolic geometry naturally says:
- "The farther I go from the center, the more space becomes available."

So they even have the same pattern here. It will look like:
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

This all will help us later with the Poincare embeddings.

Now we will continue with the Complex numbers, and rotations groups.

# Chapter 2. Complex numbers, and rotations groups

This topic is small, but it is really helpful for us, because without it we will struggle to understand everything we are going to learn.

## 1. What are Complex numbers

What are even the complex numbers ($\mathbb{C}$)? 

We know what real numbers ($\mathbb{R}$) are, for example:
$$5, \quad−3, \quad0,\quad\frac{1}2​,\quad\sqrt2
​.$$
Imagine this, if I give you this equation here:
$$x^2 = 4$$
What are the solutions? Easy. We can understand just by a glance that:
$$2^2 = 4 \quad and \quad (-2)^2$$
Therefore $x$ can have two values:
$$2 \quad or \quad -2$$
That it, simply. 

But what if I give you this equation here:
$$x^2 = -1$$
This one... hmmm... let us try some real numbers:

Positive:$$2^2 = 4$$$$1^2 = 1$$$$0^2 = 0.$$Negative:$$(-1)^2 = 1$$$$(-2)^2 = 4.$$
Hmm... a tad hard... did we noticed something? We can't get a negative number:
$$x^2 \geq 0$$
Soo... our $x^2 = -1$ has no real solution. Now we can pack things up and go away, right? Nope.

Because we can still invent a new number whose square is -1.

This number will be:
$$i$$
And define it by:
$$i^2 = -1$$
So imaginary number doesn't mean a 'fake' number.
$i$ is a number that is not a real number, introduced so that equations such as $(x^2=-1)$ have a solution.

$i$ is not positive, negative or zero. It is simply imaginary. 

The solutions for:
$$x^2 = -1$$
is simply
$$i \quad or \quad -i$$
Because:
$$(-i)^2 = (-i)(-i) = i^2 = -1$$
But we will have different types of equations, as:
$$x = 3 + 2i$$
This equation here, will lead us to the definition of a complex number. Because the form of a complex number is:
$$z = a + bi$$
Let us decode each syllable:
$z - \text{This is the complex number}$

$a - \text{This is just a real number of the complex number (called: Real part). We will see it many times as:}$ 
$Re(z) = a, \text{ this simply means "The real part of z is"}$

$b - \text{This is simply a real number, but we use like this:}$
$$b \times i = bi$$

Now we will decode the idea from earlier:
$$z = 3 + 2b$$

Now we understand that the real part here is:
$$Re(z) = 3$$

and that the imaginary part is:
$$Im(z) = 2$$
Warning: The imaginary part is the coefficient 2 not the term $2i$.

But what if we will encounter something as:
$$z = 5i$$
Then we write it as:
$$z = 0 + 5i$$
So the real part will be zero and the imaginary will be 5. The same idea applies if we have just the real part and not the imaginary.

So we can understand:
$$\mathbb{R} ⊂ \mathbb{C}$$
Because the real numbers give us a part of the equation (the $a$)

But why do even care about them? What do we even understand by:
$$z = a + bi$$
But there is a surprising part, we can associate it as:
$$(a, b)$$
This means that we can draw complex numbers as points in a plane.

We simply create two axes, of which one is the "Real" axis and the second axis is the "Imaginary" one.
The Real axis is the horizontal one, while the imaginary is the vertical one:

So we have:
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

We can't order imaginary numbers as:
$$⋯<−2<−1<0<1<2<⋯$$
Because the imaginary numbers don't live on the same one-dimensional ordering. They live on the plane.

So instead of asking:
- "Is $i$ positive or negative?"

we ask:
- "Where is $i$ in the complex plane?"

## 2. Modulus and Argument

We already know that the complex number can be viewed as a point on the plane:
$$(a, b)$$
But now we have other questions too...

How far is $z$ from the origin?
In which direction is $z$ pointing?

This two question introduce to us two new concepts:
Modulus and Argument.

We will start with the modulus.

### 1. Modulus

Suppose that we have a complex number as: 
$$z = 3 + 4i$$
Now we will place it on the plane.
![[Pasted image 20260826100247.png|436]]

we see that the origin is at:
$$(0, 0)$$
And we want the distance from $(0,0)$ to $(3, 4)$ And that is just the Pythagorean theorem.
Therefore:
$$|z| = \sqrt{3^2 + 4^2}$$
$$|z| = \sqrt{25}$$
$$|z| = 5$$
This simply means the distance from the origin to $z$.
Because we know that we just found he length vector (magnitude).

Now we can continue with the Argument.

### 2. Argument

Now that we know that the magnitude is of 5 (5 units away), but where does it point? 
To answer to this question we will look at the angle between the positive real axis and the line from the origin to $z$

![[Pasted image 20260826101500.png|452]]

This is how we will do it. We will write it as:
$$arg(z) = \theta$$
What does this mean? This is as said: The angle of z measured from the positive real axis.
So for:
$$z = 3 + 4i$$
We will do:
$$\tan\theta = \frac{opposite}{adjacent}$$
So we have:
$$\tan\theta = \frac{4}3$$
Therefore with a bit of trigonometry:
$$\theta = \arctan\frac{4}3$$
In the end we get:
$$\theta \approx 53.13^o$$

But why do we even care about getting $|z|$ and $arg(z)$?
We care about it because this way we described where the point is with the polar coordinates:
$$(r, \theta)$$
But what if we have something as:
$$z = i$$
This is like:
$$z = 0 + 1i$$
This is why the points are simply $(0, 1)$, now we will find its modulus and argument:
Modulus:
$$|z| = \sqrt{0^2 + 1^2} = 1$$
Argument:
$$arg(i) = 90^o$$
Because it is directly above the origin. Or if we really want, we can write it in its radiant form:
$$arg(i) = \frac{\pi}2$$
But we have 4 special cases that will annoy us, because even if their magnitude is the same, the $arg(z)$ is not

| $z$  | $\vert z \vert$ |  $\arg(z)$  |
| :--: | :-------------: | :---------: |
| $1$  |       $1$       |  $0^\circ$  |
| $i$  |       $1$       | $90^\circ$  |
| $-1$ |       $1$       | $180^\circ$ |
| $-i$ |       $1$       | $-90^\circ$ |
Once we reach $360^o$ we will point the same direction as:

| Angle ($\theta$) |     Direction     |
| :--------------: | :---------------: |
|    $0^\circ$     |    Point right    |
|    $90^\circ$    |     Point up      |
|   $180^\circ$    |    Point left     |
|   $270^\circ$    |    Point down     |
|   $360^\circ$    | Point right again |
This is how it work, because angles wrap up (I mean that once you rotate a full $360^o$, you are pointing in exactly the same direction again.)

We already know the polar form, but we are going to go more deep with it.

## 3. Polar Form & Euler's form

After understanding how to get the modulus and argument, we can clearly understand that if we went from: 
$$(a, b)\rightarrow (r, \theta)$$
we can go even vice versa, by simply doing:
$$a = r\cos\theta \quad and \quad b = r\sin\theta$$
Therefore we understand that:
$$z = r(\cos\theta + i\sin \theta)$$
But now we will write it as:
$$z = re^{i\theta}$$
Why?
Firstly we will understand part by part.

What does $e$ means?
As we already know, $e$ is a special number that appears in exponential growth and decay.
For example:
$$e^0 = 1, \quad e^1 = e, \quad e^2 = e \times e$$
So $e^x$ is just an exponential function, nothing strange.

We already know what $e^x$ means when $x$ is a real number. But what if we have:
$$e^{i\theta}$$
Now $x$ is an imaginary number, but this has a beautiful geometrical meaning, let us see why.

First we will introduce its name - Euler formula:
$$e^{i\theta} = \cos\theta + i\sin\theta$$

Now we will decode it:
- $e$ - this is the Euler number (2.71828...)
- $i$ - this is our imaginary unit ($i^2 = -1$)
- $\theta$ - this is the angle
- $\cos\theta$ - this is the horizontal coordinate of a point on the unit circle 
- $\sin\theta$ - this is the vertical coordinate

Let us try this formula with two examples:
Let us say that $\theta = 0$:
$$e^{i0} = \cos0 + i\sin0$$
$$\cos 0 =1 \quad and \quad \sin0 = 0$$
So that means that we have:
$$e^{i0} = 1 + i(0)$$
$$e^{i0} = 1$$
Geometrically, 1 is the point.
$$(1, 0) - \text{ We start at the right side of the unit circle}$$

Now we will try with $\theta = \frac\pi2$:
$$e^{i \frac\pi2} = \cos \frac\pi2 + i\sin \frac\pi2$$
$$e^{i \frac\pi2} = 0 + i(1)$$
Therefor:
$$e^{i \frac\pi2} = i$$
Geometrically we rotated by $90^o$

So what is $e^{i\theta}$? We can understand that $e^{i\theta} = \cos\theta + i\sin\theta$  and therefor even $|e^{i\theta}| = 1$ (This means that it is always unit away from the origin). 
This will change only the direction

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

Now let us return to the polar form, where we previously found:
$$z = r(\cos\theta + i\sin\theta)$$
But now that we know what Euler says:
$$e^{i\theta} = \cos\theta + i\sin\theta$$
We just substitute:
$$z = re^{i\theta}$$
The polar form contains two really important units:

$r$ - the magnitude ($r = |z|$)
$e^{i\theta}$ - the direction at angle $\theta$

This is what the polar form is telling us.

Now I will give you an example:
Let us take back the complex number we used till now:
$$z = 3+ 4i$$
Now let us find everything we need:
$$|z| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$
$$r = \arctan(\frac43) \approx 53.13^o$$
Now we just slam them there:
$$z = 5e^{i55.13^o}$$

But what if we change a bit the formula? We write it as:
$z = re^{i\phi}$

Where:
$r$ - its current distance from the origin
$\phi$ - its current angle

Now we are going to multiply it by $e^{i\theta}$:
$$e^{i\theta} \times z = e^{i\theta}z$$ Now we are going to substitute:
$$e^{i\phi}re^{i\theta}$$
And since we have:
$$re^{i\theta}e^{i\phi}$$
We will do the easy exponential rule ($e^ae^b = e^{a + b}$):
$$re^{i(\theta + \phi)}$$
Therefore we can understand that:
$$e^{i\theta}z = re^{i(\theta + \phi)}$$
Each way, the distance stay the same. The only thing that changes is the angle, because we had before:
$$\phi$$
and now we end up with:
$$\phi + \theta$$

Let us give you an example, so you understand how it works:

Suppose that $z = 2$ which will be $(2, 0)$. In a polar form it will be like:
$$z = 2e^{i\theta}$$
Now we will multiply it by $e^{i\pi/2}$ to rotate it:
$$e^{i\pi/2}z = e^{i\pi/2}(2e^{i0})$$
Now we will combine the exponents:
$$2e^{i\pi/2}$$
Therefore:
$$e^{i\frac{\pi}{2}} = \cos\left(\frac{\pi}{2}\right) + i\sin\left(\frac{\pi}{2}\right) = 0 + 1i = i$$

Now we will get from:
$$(2, 0) \rightarrow (0,2)$$
This is how it works.

Now after all this mess, we will continue with another mess. Because all of this was too easy...

## 4. Möbius Transformation basics

Firstly we will not hit you with a full power formula, we will try to explain some minor ideas. For example, we already know what a transformation is, but just to be sure, I will explain it once again.

The transformation is simply a rule that takes an input and returns an output.
For example:
$$f(x) = x + 2$$
This will take our varibale $x$ and add 2 to it.
For a complex number we can do the same stuff. But instead of:
$$x \rightarrow f(x)$$
We will do:
$$z \rightarrow f(z)$$ 
Which simply means to take a point $z$, and move it to another point $f(z)$

Before the scary part, we will do some simple transformations.
Imagine that we have:
$$f(z) = z + b$$
where $b$ is a complex number.

So if $z = 2 + 3i$ and $b = 1 + i$, then we will get:
$$f(z) = (2 + 3i) + (1 + i)$$ 
$$f(z) = 3 + 4i$$
So the point moved: $(2, 3) \rightarrow (3, 4)$. This is a transformation.

What about scaling and rotating at the same time?
We already learned this really powerful idea:
$$e^{i\theta}z$$
which aimply means: rotate $z$ by $theta$. And also, we can multiply it by a positive real number to scale it.
For example:
$$2z$$
This will double the distance from the origin and rotate the input.
For example, if we had: (the z is the curent we have)
$$a = 2e^{i\pi/2}$$
And now if we multiply it by our current complex number ($z$):
$$az = 2e^{i\pi/2}z$$
This means that $z$ will get rotated by $90^o$ and it will be scaled by 2.

Now we will combine the two transformations! (I mean the $az$ and $b$)
So we get:
$$az + b$$
Now I will break it piece by piece:
- $z$ - This is our original point
- $a$ - This is a complex number that can scale and rotate
- $b$ - This is a complex number that translates (change the position we currently had)
So in the end $az + b$ will just scale/rotate $z$, and then translate it.

Now we will introduce a second part... the division:
$$f(z) = \frac{az + b}{cz + d}$$
This is what we call the Möbius transformation. 
Now we will decode every symbol.
- $z$ - This is our starting complex number and it is a point in the complex plane.
- $a, b, c, d$ - This are our four complex constants. Each of them is a complex number, so they can have a real part and an imaginary part. 

But why do we divide? We divide because we will be able to turn straight lines into circles, turn the plane inside out, and so on. (We will learn it deeper when we will have the full topic.)

I will explain the whole idea of why we do it after a bit.
But we have a restriction! A rule we must follow is:
$$ad - bc \not = 0$$
Why? Because if we get 0, the transformation degenerates into something that isn't a proper Möbius transformation. This is the non-degeneracy condition. For now this simple intuition is enough.

Now I will give you an example:
$$a=1 \quad b=2 \quad c=0 \quad d=1$$
We will get:
$$f(z) = \frac{1z + 2}{0z + 1}$$
$$f(z) = z + 2$$
So if we had something as $z = 3 + i$, we would get:
$$f(z) = 5 + i$$
In the end we get:
$$(3, 1) \rightarrow (5, 1)$$
This was a simple transformation.
Now we will try to rotate it by $270^o$ and shrink it in half.

We will us $z = 5 + 3i$
To rotate it by $270^o$ (in radians it will be: $\frac{3\pi}2$) we will use the Euler form:
$$e^{i\theta} = e^{i 270^\circ} = \cos(270^\circ) + i\sin(270^\circ) = 0 + i(-1) = -i$$
We understand that we have to multiply it by $-i$ if we want to rotate it by $270^o$.
And to shrink it we will use simply $\frac12$. Now we will apply it to the original formula.

$$w_1 = A \cdot z = -\frac{1}{2}i (5 + 3i) = -\frac{5}{2}i - \frac{3}{2}i^2 = \frac{3}{2} - \frac{5}{2}i$$

This is our result!

Now we will continue with the topics we had to continue all along, because for now, we learned just basics, and even if hard, they were useful. 
The next topic will be about the Minkowski space coordinate geometry... already sounds frightening, yet I will make it easy to understand.

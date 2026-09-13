
Now we start with the annoying type of mess that nobody likes, but sadly we have to learn them, otherwise all we learned till now would be a waste, and we don't want this. This is why we would rather keep going with this stuff.

We will the Minkowski messy idea.

# Chapter 1. Minkowski Space Coordinate Geometry

Disclaimer - if you get an ictus while learning all of this, remember that it is not my fault.

## 1. Minkowski inner product

Now I will not start immediately with:
$$\langle x, y \rangle_M = -x_0 y_0 + x_1 y_1 + \cdots + x_n y_n$$
and say:
"the one negative sign that makes this Lorentzian, not Euclidean."

But we have to earn that statement, not just to memorize it. We will firstly understand the whole idea and only then we will start again with this formula.

We will start ridiculously simple. Imagine we have a vector as:
$$x = \begin{bmatrix} 2 \\\\ 3\end{bmatrix}$$
We can think about it as:
$$x = (2, 3)$$
But what if we had a vector with $n + 1$ coordinates:
$$x=(x_0,x_1,…,x_n)$$
and another:
$$y=(y_0,y_1,\ldots,y_n)$$
This is literally:
- $x_0$ = first coordinate of $x$
- $x_1$ = second coordinate
- $x_2$ = third coordinate
- ...
- $x_n$ = final coordinate

Same for y.
(Remember that x is a vector, just like the good ol' feature and samples vector we used, so each new feature add a dimension)

The normal Euclidean inner product is just the sum of each vector summed element-wisely, for example (Suppose that $x = (2, 3)$ and $y = (4, 5)$):
$$\langle x,y\rangle=(2)(4)+(3)(5)$$$$=23$$
Nothing special, right? But Minkowski makes a small change... almost unnoticeable:
$$-x_0 y_0 + x_1 y_1 + \cdots + x_n y_n$$
He places a negative sign at the start, and now we have a question... does it really matters? Personally? Maybe nah. Geometrically? Absolutely.

We know that the Euclidean inner product is: 
$$\langle x, y \rangle$$
But in the Minkowski inner product is:
$$\langle x, y \rangle _M$$
So, let us try again the same example, but with the Minkowski inner product (We will use the same $x = (2, 3)$ and $y = (4, 5)$):
$$\langle x, y \rangle _M = -(2)(4)+(3)(5)$$
$$= 7$$
As we noticed, in the Euclidean geometry we got 23, while in the Minkowski we got 7. Same vectors, same coordinates, different geometries. All of that because of one sign

But why to make the first coordinate negative? We will answer to this question with an idea. 

In the Euclidean space, the squared magnitude (length) is always positive, no matter what. For example:
$$||x||^2 = \langle x, x \rangle > 0$$
Which is just:
$$\langle x,x  \rangle = x_0^2 + x_1^2+ ... +x_n^2$$

In Euclidean geometry, we could get only positive inner products, meanwhile in the Minkowski space, we may get a negative, zero, or a positive number. But now we will explain what each of the results mean.

1. Negative - Timelike Vectors ($\langle x, x \rangle_M < 0$):

Here the inner product is negative and it means that the vector is inside the hyperboloid.

![[Figure_timelike.png|665]]

2. Positive - Spacelike Vectors ($\langle x, x \rangle_M > 0$):

Here, the inner product is positive and it means that the vector is outside the cone.

![[Figure_ll.png|651]]

3. Zero -Null Vectors ($\langle x, x \rangle_M = 0$):

Here, the inner product is 0 and it means that the vector is on the edges of the cone.

![[Figure_Null.png|645]]


So, we understand what each of them means, but we don't understand with what they help us, but that is not a big deal, because we will understand it a bit later. But at least... why do we care? We care because the Lorentz model of hyperbolic space continues inside the Minkowski space.

Now we will talk about the Lorentz distance.

## 2. Lorentz distance

Firstly, let us remember the euclidean distance. We will take two points:
$$x = (x_1, x_2) \quad and \quad y = (y_1, y_2)$$
To find the distance in an Euclidean space we will write:
$$d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$$
For example:
$$x = (1, 2) \quad and \quad y = (4, 6)$$
$$d = \sqrt{(1 - 4)^2 + (2 - 6)^2} = \sqrt{25} = 5$$
This is their distance in an euclidean space.

Minkowski uses the same idea even here. Suppose $x = (x_0, x_1, \dots, x_n)$ and $y = (y_0, y_1, \dots, y_n)$. The displacement from $y$ to $x$ is:
$$x - y = (x_0 - y_0, x_1 - y_1, \dots, x_n - y_n)$$
Now we will use the inner product to calculate this displacement:
$$\langle x-y, x-y \rangle_M = (x_0 - y_0)^2 + (x_1 - y_1)^2 + \dots + (x_n - y_n)^2$$

As we noticed... it is the same stuff, no? Well... yes... because we forgot about something. 
By the Minkowski rule:
$$\langle v, v \rangle_M = -v_0^2 + v_1^2 + \dots + v_n^2$$
We see that the first coordinate shall be negative, this is why we have:
$$\langle x-y, x-y \rangle_M = -(x_0 - y_0)^2 + (x_1 - y_1)^2 + \dots + (x_n - y_n)^2$$

As expected, this negative sign changed the whole geometry.
But as expected, we have a problem, because in the Minkowski space, the distance can become negative... if we will get in the end something as $-3$ and $1$, we would just get $\sqrt{-8}$, which we can't call an ordinary distance, this is why we need to be more careful.

But we have to understand an idea... we can't measure the hyperbolic distance between all the arbitrary point in all the Minkowski space (With this I mean that we wouldn't use all the Minkowski space for our Hyperboloid. We will select a part out of it.)

But before we continue, I wanted to say that we choose two things only. We choose all the points to which the inner product is -1 and we will choose the upper bowel (The upper sheet), not both of them. Overall it means that the point shall be inside the upper bowel only.

This part will be exactly:
$$\langle x,x \rangle_M = -1$$
But we can write it in another form, since we know that:
$$⟨x,x⟩_M=−x_0^2+x_1^2+x_2^2$$
Therefore we're asking for points satisfying:

$$ -x_0^2+x_1^2+x_2^2=-1. $$

Rearrange:

$$ x_0^2-x_1^2-x_2^2=1. $$

And that equation describes our hyperboloid.
So, to check if the point is valid on the hyperboloid, we will check two main ideas:
$$\langle x, x \rangle_M = -1$$
and that the first part ($x_0$) to be bigger than 0.
$$x_0 > 0$$

Now I will give you an example, let us see if $x = (\sqrt{5}, 2, 0)$ would be a valid point on the hyperboloid.
$$\langle x, x \rangle_M = -(\sqrt{5})^2 + 2^2 + 0^2 = -5 + 4 + 0 = -1$$
This is right, since we had to get $-1$
Now we do the second check:
$$x_0 = \sqrt{5} \approx 2.236 > 0$$
So yes, this point is valid in the hyperbolic space

Now we choose another point:
$$o = (1, 0, 0, ..., 0)$$
We will check it:
$$\langle o, o \rangle_M = -(1)^2 + 0^2 +... + 0^2 = -1 + 0 + 0 = -1$$
and:
$$x_0 = 1 > 0$$
So this works.

We will use this exact point may times, as the origin/base point of the Lorentz model.

But we came here for the distance, didn't we? So let us take two points that belong to the hyperbolic space (We can write it as: $x \in \mathbb{H}^n$, $y \in \mathbb{H}^n$ ). (For now we understand that all the points that follow this two rules, belong to the hyperbolic space.)
Now we will calculate:
$$d_{\mathbb{H}}(x, y)$$
Which is the hyperbolic distance.
But how will we even calculate it? This is where the Minkowski inner product comes to clutch once again.

We would take two hyperbolic points (Two points that followed the two rules):
$$ x=(x_0,x_1,\ldots,x_n) $$
and:
$$ y=(y_0,y_1,\ldots,y_n)$$
Now we will have to calculate:
$$\langle x,y\rangle_M = -x_0y_0+x_1y_1+\cdots+x_ny_n$$
This will give us a number that is:
$$\langle x, y \rangle_M \leq -1$$
Why? Because we choose only points that are timelike (inside the hyperboloid), therefore, their result must be smaller then or equal to -1. 

But we know another rule that applies... the distance between the point and itself is always 0.

This is why we need a function that will turn this -1 (The result of the inner product) into 0.

This is why we will use the same idea, just a tad changed:
$$\langle x, y \rangle_M = -1$$
Then:
$$-\langle x, y \rangle_M = 1$$

Now that it equal to 1, $arcosh$ comes to clutch, because:
$$\operatorname{arcosh}(1) = 0$$

Therefore, the distance is:
$$d_{\mathbb{H}}(x, y) = \operatorname{arcosh}(- \langle x,y\rangle_M)$$
The distance is not always 0, it is zero only when $x$ and $y$ are identical, or we are watching the distance of the same point.

Now I will give an example, let us choose the point...  $x = (3, 2, 2)$:
We will try the first two rules, to see if it belongs to the hyperbolic space:
$$-x_0^2 + x_1^2 + x_2^2 = -(3)^2 + 2^2 + 2^2 = -9 + 4 + 4 = -1$$
and the second rule: 
$$x_0 = 3 > 0$$
Since both worked, this means that this point is a valid point in the hyperboloid.
Now we will find the distance (Of the same point):
$$d_{\mathbb{H}}(x, x) = \operatorname{arcosh}\left(-\langle x, x \rangle_M\right)$$
$$-\langle x, x \rangle_M = -(-1) = 1$$
$$d_{\mathbb{H}}(x, x) = \operatorname{arcosh}(1)$$
And in the end we know that $ar\cosh$ of 1 is 0.

The distance will help us a lot with identifying the relationships too, for example:
```
                 [ Living Being ]  <-- Root (o = (1, 0, 0))
                    /        \
             [ Mammal ]    [ Reptile ]
              /      \          \
        [ Dog ]      [ Cat ]    [ Snake ]
```

Now let us suppose that:
- Living Being (Root): Placed at the lowest tip $o = (1, 0, 0)$, height $x_0 = 1.0$.
- Mammal (Parent): Placed higher up at $(1.54, 1.18, 0)$, height $x_0 = 1.54$.
- Dog (Leaf A): Placed higher still at $(3.76, 3.20, 1.80)$, height $x_0 = 3.76$.
- Cat (Leaf B): Placed at $(3.76, 3.20, -1.80)$, height $x_0 = 3.76$ (same height as Dog, but opposite direction).
- Snake (Leaf C): Placed on a completely different branch at $(3.76, -3.60, 0)$, height $x_0 = 3.76$.

Now we will check the relationships, firstly the dog and cat:
$$\langle \text{Dog}, \text{Cat} \rangle_M = -1.25 \implies d_{\mathbb{H}}(\text{Dog}, \text{Cat}) = \operatorname{arcosh}(1.25) \approx \mathbf{0.70}$$

As we can see, the distance between them is nonexistent almost, this is why the AI instantly identifies `Cat` as the top sibling recommendation because they share a close common ancestor (Mammal).

What about the Dog and the Mammal?
$$\langle \text{Dog}, \text{Mammal} \rangle_M = -1.54 \implies d_{\mathbb{H}}(\text{Dog}, \text{Mammal}) = \operatorname{arcosh}(1.54) \approx \mathbf{1.00}$$

Decent distance along the vertical axis, the AI will identify `Mammal` as a parent category.

Now we understand all of this mess which is:

$-x_0^2 + x_1^2 + x_2^2$ -> This creates the hyperboloid -> We take all the point on the upper bowel of the hyperboloid only, and then we just find the points distance.

I'll give another full example:

Let us say that we have:
$$x = (\cosh 1, \sinh 1, 0, 0) \approx (1.543, 1.175, 0, 0)$$$$y = (\cosh 2, \sinh 2, 0, 0) \approx (3.762, 3.627, 0, 0)$$
Now we will try to see if they are points on the hyperbolic space.

---
Firstly we will check if their own inner product is equal to -1.
$$\langle x, x \rangle_M = -(\cosh 1)^2 + (\sinh 1)^2 + 0^2 + 0^2 = -\cosh^2(1) + \sinh^2(1)$$
Using the hyperbolic identities where $\cosh^2(t) - \sinh^2(t) = 1$ , we understand that the result is 1:
$$-\cosh^2(1) + \sinh^2(1) = -1 \implies \langle x, x \rangle_M = -1$$

Now we will check the timelike positivity:
$$x_0 = \cosh 1 \approx 1.543 > 0$$

After understanding that $x$ is in the hyperboloid, we will check on $y$:
We can understand that $x \in \mathbb{H}^3$.

---
Start from the inner product check:
$$\langle y, y \rangle_M = -(\cosh 2)^2 + (\sinh 2)^2 + 0^2 + 0^2 = -\cosh^2(2) + \sinh^2(2) = -1$$
Now we continue with the timelike positivity check:
$$y_0 = \cosh 2 \approx 3.762 > 0$$
Both conditions hold, so $y \in \mathbb{H}^3$.

---

Now we will find the distance.
$$\langle x, y \rangle_M = -(\cosh 1)(\cosh 2) + (\sinh 1)(\sinh 2) + (0)(0) + (0)(0)$$$$\langle x, y \rangle_M = -(\cosh 1 \cosh 2 - \sinh 1 \sinh 2)$$
Now we will apply another identity ($\cosh(a - b) = \cosh a \cosh b - \sinh a \sinh b$):
$$\langle x, y \rangle_M = -\cosh(1 - 2) = -\cosh(-1)$$Since $\cosh(-t) = \cosh(t)$:
$$\langle x, y \rangle_M = -\cosh 1$$
Now we will use the distance formula, which is:
$$d_{\mathbb{H}}(x, y) = \operatorname{arcosh}\left(-\langle x, y \rangle_M\right)$$
We will substitute by adding our result there:
$$d_{\mathbb{H}}(x, y) = \operatorname{arcosh}\left(\cosh 1\right) = 1$$

Don't worry about all that identities we solved, because we will use a table for them (Search on google about the hyperbolic identities).

Now we may find out about the new topic! (Yup, this chapters are short).

# Chapter 2. Möbius gyrovector arithmetics

Now we will think... why do we even need even the new operations? We will discover.

## 1. Möbius Addition

We learned about the hyperbolic space. But we can represent the Hyperbolic space in more way, for example, previously we used the Lorentz model to represent all the Hyperbolic space as a hyperboloid inside the Minkowski space. 
Now we will represent it as a poincare ball model. We don't have to learn it completely! Because we will have a full topic on it. We need just to understand that it is the same Hyperbolic geometry (Yeah, the one we will use for hierarchies.), just represented in a different way. Think about representing a city with a road map, a GPS coordinate, or a satellite imgae. It is the same stuff, just different representation.

What is this poincare ball?

For the n-dimensional case, we use:
$$\mathbb{D}^n = {x \in \mathbb{R}^n : ||x|| < 1}$$
We will decode it slowly.

- x - is a point (vector), for example it could be: $x = (x_1, x_2, ... , x_n)
- ||x|| - this is the normal euclidean magnitude: $||x|| = \sqrt{x_1^2 + ... + x_n^2}$. 
- $||x|| < 1$ - This means that the point must be inside the unit ball (Basically just a disk, so we will call it disk.):
```
              boundary
           .-----------.
        .-'             '-.
      .'                   '.
     /                       \
    |          ● x            |
    |       ●                 |
     \                       /
      '.                   .'
        '-._____________.-'
```

Since the radius of this disk is only of 1. We understand that every point with the magnitude (r - radius) of less then 1 is inside the disk. But even if it is 2D, it is not equipped with the basic Euclidean geometry, it uses the hyperbolic metrics. So even if it looks like a simple disk, it is in the Hyperbolic geometry.

But... can we use normal arithmetic in this disk? Hmmm... Let us try:


Let us say that we know that $x = (0.3, 0.2)$ and that $y = (0.2, 0.1)$. Let us add them:
$$x + y = (0.5, 0.3)$$

Now let us check if this point is inside the poincare ball.
$$\sqrt{0.5^2 + 0.3^2} = \sqrt{0.34} \approx 0.583 < 1$$

Yup, this is still inside the poincare ball...

But what if we take another two coordinates, a tad bigger:
$$x = (0.8, 0) \quad and \quad y = (0.8, 0)$$
Now we will add them:
$$x + y = (1.6, 0)$$
Which actually... is bigger than the poincare ball boundaries:
$$|| x + y|| = 1.6 > 1$$
Oops, we left the poincare ball, so:
$$x + y \not \in \mathbb{D^2} : \text{We left the poincare ball}$$

But we have a deeper problem... That we are in the hyperbolic space, and the operations as $+ , \times$ were made for the Euclidean geometry... So it assumes that our vector lives in a flat vector space where we are free to translate... the problem is that the space is curved, so we want to respect it. But we may get this behavior... from who? From the Möbius addition.

Instead of writing:
$$x + y $$
We may write:
$$x \oplus_M y$$
Ta-da~ We got it, now we may go to sleep, right? No. Not even close... because we will fight a monster right now:
$$x \oplus_M y = \frac{(1 + 2\langle x,y\rangle + \Vert{}y\Vert{}^2)x + (1 - \Vert{}x\Vert{}^2)y}{1 + 2\langle x,y\rangle + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2}$$
Here it is, the beast. Now we will try to solve it.
Before it, we understand that the Möbius addition preserve the geometric properties while adding two points inside the poincare ball together.

---
We will use:  $x = (0.3, 0.2)$ and $y = (0.1, 0.4)$:

We will firstly find the standard inner product:
$$\langle x, y \rangle = x_1 y_1 + \cdots + x_n y_n$$
In our case that:
$$\langle x, y \rangle = (0.3)(0.1) + (0.2)(0.4) = 0.03 + 0.08 = 0.11$$

Now we will get each squared length.
For $x = (0.3, 0.2)$:
$$\Vert{}x\Vert{}^2 = 0.3^2 + 0.2^2 = 0.09 + 0.04 = 0.13$$For $y = (0.1, 0.4)$:
$$\Vert{}y\Vert{}^2 = 0.1^2 + 0.4^2 = 0.01 + 0.16 = 0.17$$

This is all we needed, because now we can just do some easy arithmetic, but firstly I will write down all the results:
$x = (0.3, 0.2)$,  $y = (0.1, 0.4)$
$\langle x, y \rangle = 0.11$, 
$\Vert{}x\Vert{}^2 = 0.13$, 
$\Vert{}y\Vert{}^2 = 0.17$

---
Firstly we will start with the denominator:
$$\text{Denominator} = 1 + 2\langle x,y\rangle + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2$$
$$\text{Denominator} = 1 + 2(0.11) + (0.13)(0.17) = 1 + 0.22 + 0.0221 = 1.2421$$
---
Now we will break the numerator into pieces, firstly we solve the first part ($x$)
$$(1 + 2\langle x,y\rangle + \Vert{}y\Vert{}^2)x = (1 + 2(0.11) + 0.17)x = (1 + 0.22 + 0.17)x = 1.39x$$
We will multiply the result by the point $x$:
$$1.39x = 1.39(0.3, 0.2) = (0.417, 0.278)$$
---
Now we will solve the second part ($y$):
$$(1 - \Vert{}x\Vert{}^2)y = (1 - 0.13)y = 0.87y$$
We will multiply the result by $y$:
$$0.87y = 0.87(0.1, 0.4) = (0.087, 0.348)$$
---
Now we will add both of the points together:
$$(0.417, 0.278) + (0.087, 0.348) = (0.504, 0.626)$$
And now we divide it by the denominator:
$$x \oplus_M y = \left( \frac{0.504}{1.2421}, \frac{0.626}{1.2421} \right) \approx (0.406, 0.504)$$
---

So we understood that in a normal euclidean space, we use the normal addition, while in the Poincare ball (which is another way of representing the hyperbolic space).

But there is another idea. The Möbius addition is not commutative ($x \oplus_M y \neq y \oplus_M x$).
But how? Let us make them perpendicular... so their inner product (dot product) will become 0.

We take the original formula:
$$x \oplus_M y = \frac{(1 + 2\langle x, y \rangle + \Vert{}y\Vert{}^2)x + (1 - \Vert{}x\Vert{}^2)y}{1 + 2\langle x, y \rangle + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2}$$
Firstly we will take two perpendicular vectors, and now we start with the two ideas::

Scenario 1:  $x \oplus_M y$  - 
$$x \oplus_M y = \frac{(1 + \Vert{}y\Vert{}^2)x + (1 - \Vert{}x\Vert{}^2)y}{1 + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2}$$
Scenario 2: $y \oplus_M x$  -
$$y \oplus_M x = \frac{(1 - \Vert{}y\Vert{}^2)x + (1 + \Vert{}x\Vert{}^2)y}{1 + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2}$$
We already noticed a difference... the numerator has different operations.

Let us take:
Let $x = (0.5, 0)$ and $y = (0, 0.2)$.

I will skip the steps and write the results:

Scenario 1: $x \oplus_M y$ -
$$x \oplus_M y = \frac{(0.52, 0.15)}{1.01} \approx (0.5148, 0.1485)$$
Scenario 2:  $y \oplus_M x$ - 
$$y \oplus_M x = \frac{(0.48, 0.25)}{1.01} \approx (0.4752, 0.2475)$$

We can clearly see that:
$$(0.5148, 0.1485) \neq (0.4752, 0.2475)$$

But there is something mind blowing. The Euclidean magnitude is identical to both:
$$\Vert{}x \oplus_M y\Vert{}^2 = 0.5148^2 + 0.1485^2 \approx 0.2870$$$$\Vert{}y \oplus_M x\Vert{}^2 = 0.4752^2 + 0.2475^2 \approx 0.2870$$
$$\Vert{}x \oplus_M y\Vert{} = \Vert{}y \oplus_M x\Vert{} = \sqrt{0.2870} \approx 0.5357$$

So they have a identical lengths.

This is how it works! 

Now we will continue with the Möbius multiplication.

## 2. Möbius scalar multiplication 

Since we learnt about the Möbius addition, we have to learn even the scalar multiplication. 
In a ordinary vector space we ask ourselves what will happen if we multiply a vector by a number?

But we have to understand how to do the same but in a poincare ball.

In a Euclidean space we could just take $x$ (suppose that $x = (2, 1)$) and multiply it by the scalar (a single number - we will take 3). 
$$3x = (6, 3)$$

Geometrically, we have made the vector 3 times longer.
As:
```
x:

O ─────>

3x:

O ──────────────────>
```

Same idea for $\frac12$$, it will make the vector in half.

But we already know something sad, that the poincare ball is curved.
Our Poincare bass is:
$$\mathbb{D}^n = {x : ||x|| < 1}$$
All the points must be inside the ball.

What if our $x = 0.8, 0)$, maybe we want to make it a tad bigger... so we multiply it by the scalar 2!
$$2x = (1.6, 0)$$

Ah, we left the ball, since $||x|| = 1.6 > 1$

So ordinary:
$$rx$$

Works in euclidean geometry but it is not appropriate for the Poincare ball.
Now we will look at what is appropriate for the Poincare ball.
$$r \otimes_M x = \tanh(r \times ar\tanh(||x||)\frac{x}{||x||}$$

It may look scary, but we will break it in pieces. Because the full operation looks like: 
$$x \rightarrow \text{measure its radial position - (distance from the center (origin) to the point)} \rightarrow \text{scale that position} \righarrow \text{convert back}$$

We know that $||x||$ is just the Euclidean length of $x$. For example, let us say that we have $x = (0.6, 0.3)$, now we will find its magnitude:
$$||x|| = \sqrt{0.6^2 + 0.3^2}$$
$$= \sqrt{0.36 + 0.09}$$
$$= \sqrt{0.45}$$
$$\approx 0.671$$

So we understand that the point is about $0.671$ Euclidean units from the center.

Now we will do:
$$\frac{x}{||x||}$$
Now we will take $x$ and turn it into a unit vector that vector points in exactly the same direction. Since we have $x = (0.6, 0.3)$ and $||x|| \approx 0.671$.
Therefore:
$$\frac{x}{||x||} \approx (0.894, 0.447)$$
Its length is approximately 1. 
So we have just separated the "direction" from the "distance from the center."

This is what we needed.

What about the $ar\tanh$? this will tell us how far is $x$ from the center, according to the hyperbolic geometry. For the Poincare ball, the hyperbolic distance from the center id:
$$d(0, x) = 2 \times ar\tanh(||x||)$$

So if $||x|| = 0.5$
we will calculate:
$$d(0, x) = 2 \times ar\tanh(0.5) \approx 1.0986$$

So we notice that the euclidean radius is 0.5, but the hyperbolic distance from the center is 1.0986.

---
But before we continue, I wanted to make a distinction, so we don't accidentally see one of them:
$||x||$, $\frac{x}{||x||}$ 
and say that they are the same.
Let us say that we have $x = (0.6, 0.3)$. Now we will check each of them:

1. $||x||$ - we will try this as first:

Now we will find out how far is the point from the center if we measure it with the Euclidean geometry?
$$\Vert{}x\Vert{} = \sqrt{0.6^2 + 0.3^2} = \sqrt{0.36 + 0.09} = \sqrt{0.45} \approx 0.671$$
The point is 0.671 units away from the center (In Euclidean geometry)

2. $\frac{x}{||x||}$ - we will try this as second:

This will strip away the magnitude and leave us with the direction only, for example:
Vector $x$: "5 miles North-East"
Length $\Vert{}x\Vert{}:$ "5 miles"
Division $\frac{x}{\Vert{}x\Vert{}}:$ $\frac{\text{5 miles North-East}}{\text{5 miles}} = \text{"North-East"}$

This will help us to reduce its magnitude to 1 but let the vector point in the same direction as (We will use $x = (3, 4)$):
$$\Vert{}x\Vert{} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$
$$\frac{x}{\Vert{}x\Vert{}} = \left( \frac{3}{5}, \frac{4}{5} \right) = (0.6, 0.8)$$
Now we check its new length:
$$\Vert{}(0.6, 0.8)\Vert{} = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1$$
So it points the same way as the old $x$, but its magnitude is 1.

---

Sooo, we can find the hyperbolic distance from the center by writing:
$$d(0, x) = 2\times ar \tanh(x)$$
But anyway, we still have to understand that the Euclidean radius $\not =$ Hyperbolic radius

One of the reasons is that the Euclidean radius ranges from 0 to 1, while the Hyperbolic radius ranges from 0 to $\infty$. 

In the end of the journey, we convert it back, by using $\tanh$.

So, we actually scaled the radial coordinate in hyperbolic space and then we turn it back into a valid Poincaré-ball radius. 
And something expected is that: 
$$\tanh(z) < 1$$
Because when we convert it to the poincare ball radius, it has to stay in the ball, so it can't be bigger than 1.

And we end it with $\frac{x}{||x||}$ to get the new points and the same direction.

This is how we got:
$$r \otimes_M x = \tanh(r \times \operatorname{artanh}(||x||)\frac{x}{||x||}$$
Now I will give you an example (We will use $x = (0.3, 0.4$)):

---
Step 1: Find the direction vector ($\frac{x}{\Vert{}x\Vert{}}$) -
$$\Vert{}x\Vert{} = \sqrt{0.3^2 + 0.4^2} = \sqrt{0.09 + 0.16} = \sqrt{0.25} = 0.5$$
Now we find the direction vector and normalize the distance:
$$\frac{x}{\Vert{}x\Vert{}} = \frac{(0.3, 0.4)}{0.5} = (0.6, 0.8)$$
---
Step 2: Convert the Euclidean radius (magnitude) to Hyperbolic space -

Here we will convert the Euclidean radius $\Vert{}x\Vert{} = 0.5$ into its hyperbolic radial coordinate using $\operatorname{artanh}$:

$$\operatorname{artanh}(0.5) = \frac{1}{2} \ln\left( \frac{1 + 0.5}{1 - 0.5} \right) = \frac{1}{2} \ln(3) \approx 0.5493$$
(Little reminder, the full hyperbolic distance from the origin to the point is $2 \times 0.5493 = \ln(3) \approx 1.0986$)

---
Step 3: scale it by $r$ - 

Now we will scale the hyperbolic radial coordinate by $r = 2$:

$$r \cdot \operatorname{artanh}(\Vert{}x\Vert{}) = 2 \cdot \left(\frac{1}{2} \ln(3)\right) = \ln(3) \approx 1.0986$$

---
Step 4: Convert back to the euclidean radius:

Convert $\ln(3)$ back into a valid Euclidean radius inside the disk using $\tanh$

$$\tanh(\ln(3)) = \frac{e^{\ln(3)} - e^{-\ln(3)}}{e^{\ln(3)} + e^{-\ln(3)}} = \frac{3 - \frac{1}{3}}{3 + \frac{1}{3}} = \frac{\frac{8}{3}}{\frac{10}{3}} = \frac{8}{10} = 0.8$$

---
Step 5: Recover the distance vector:

To get the distance vector, we will multiply the original vector by the new radius:

$$2 \otimes_M (0.3, 0.4) = 0.8 \cdot (0.6, 0.8) = (0.48, 0.64)$$

![[Möbius_Scalar.png|583]]

This is a really simplified version of the idea. But don't worry, because we will have a lot of topics on it.

Another way it helps us is in hierarchies. For example, an AI doesn't see words as we do, the AI goes after the semantic meaning of the words, and let us give an example how the AI can use the Minkowski gyrovector arithmetic.

Imagine an ai shop assistant for a shop like amazon...

Suppose that an user searches for  a product embedding: $x = \text{"Basic Running Shoes"}$.
But the users adds: "Show me something more premium and waterproof"

The ai learned 2 different transformation arrows:

Arrow $r_{\text{premium}}$ = "Shift concept toward high-end luxury."
Arrow $r_{\text{waterproof}}$ = "Shift concept toward outdoor gear."

Now the AI will find the product... by finding firstly the location of it, but how will it find the location of it?
The AI will have to take the starting position and add it to the new transformations arrows. This is how it will find the new product.
$$\text{New Product Location} = \text{Starting Shoes} + \text{Premium Shift} + \text{Waterproof Shift}$$

Wait... what would happen if we use basic addition? We will risk to get out of the map, since the hierarchy tree is in a hyperbolic space. And by getting out of the map, because the basic addition ignores the curved geometry of the tree, the math calculates a location outside the catalog universe. The system crashes or returns nonsensical garbage.


Or it may give a wrong answer, since taking a straight step would make the AI look for Waterproof Iphones, instead of what the user needs. This is why it is not recommended to use normal addition and scalar multiplication.

Instead, if we use the Möbius arithemtic, the AI glides perfectly where it needs to and get much closer to the real answer, and if the user asks something as "a little waterproof", the ai will scale the waterproof idea by 0.3, giving it less importance.

So let us solve the problem, before it gets cold.

---
Starting Point: $x_{\text{shoes}} \in \mathbb{D}^n$ (The base embedding for "Basic Running Shoes")
Shift 1: $r_{\text{premium}} \in \mathbb{D}^n$ (The vector direction for "Make it Premium")
Shift 2: $r_{\text{waterproof}} \in \mathbb{D}^n$ (The vector direction for "Make it Waterproof")

Suppose the user gives specific intensity requests:
80% Premium ($\alpha = 0.8$)
50% Waterproof ($\beta = 0.5$)


---
Now we will immediately scale the intensity of the types:
$$\text{Scaled Premium} = 0.8 \otimes_M r_{\text{premium}}$$$$\text{Scaled Waterproof} = 0.5 \otimes_M r_{\text{waterproof}}$$
We used the Möbius scaling to respect the geometry, otherwise it would make the embedding  something totally unrelated.

---
To find the final location of the item, the AI will combine all the ideas, we should have something as:
$$x_{\text{target}} = x_{\text{shoes}} \oplus_M r_{\text{combined}}$$
So, all of it in a single equation would look like:
$${ x_{\text{target}} = x_{\text{shoes}} \;\oplus_M\; \Big( \left( 0.8 \otimes_M r_{\text{premium}} \right) \;\oplus_M\; \left( 0.5 \otimes_M r_{\text{waterproof}} \right) \Big)}$$

___
And in the end, the AI finds the closest actual item in the inventory using hyperbolic distance:
$$d_{\mathbb{H}}(x_{\text{target}}, y_{\text{product}}) = 2 \operatorname{artanh}\left( \Vert{}(-x_{\text{target}}) \oplus_M y_{\text{product}}\Vert{} \right)$$
Ta-da~ This is how the whole process works!
___

Now we will give another important idea... imagine making an AI that store your data:

- Self - This is your real self, the one full of failures and mistakes, and the one who didn't give up, despite the topic being hard
- Mask - This is the facade side, the professional side with no struggles, fully perfect, and engineer-like behavior.

But we need a mix... something with both... something that we choose to show to people - Because showing only the bad sides = Not professional, showing only good sides = Looks like a robot.

Now we will find the perfect mix:

$x_1$ (Abstraction/Granularity): $0.0$ = High-level skill summary $\longrightarrow$ $1.0$ = Granular private logs.
$x_2$ (Domain Context): Negative = Internal struggle/reflection $\longrightarrow$ Positive = Shareable open-source artifact.
$x_3$ (Emotional Tone): Negative = Raw internal vulnerability $\longrightarrow$ Positive = Polished professional output.

Now we can go write down both:
$$x_{\text{self}} = (0.50,\; -0.30,\; -0.40)$$
$$x_{\text{mask}} = (0.20,\; 0.40,\; 0.10)$$
Now we will find the mix of both... Something we choose how much to show of. For example, if self and mask looks like:

Self ($x_{\text{self}}$): "Spent 4 days stuck on Graph Fourier Transforms. Felt completely out of my depth with the Laplacian matrix math, but kept grinding because I want to build medical AI. Finally clicked after writing a custom Julia script."

Self ($x_{\text{mask}}$): "Proficient in Graph Signal Processing, Spectral Graph Theory, and GNN architectures."

We will mix them and use alpha to write down how much we want to reveal about them.
For example:

$\alpha = 0.0$ (Private Session — Just You & Your AI):
"You've been putting in serious work on Graph Fourier Transforms over the last 4 days. Remember how stuck you felt on the Laplacian matrix?..."
Why: 100% authentic, validates your struggle, references your exact private tools

$\alpha = 0.4$ (Study Group / Peer Collaborator):
"I have practical experience working through spectral graph theory and Graph Signal Processing, including implementing custom GNN signal filtering scripts in Julia"
Why: Real and human—shows you build things from scratch and care about medical AI, but leaves out the personal frustration and 4-day mental wall.

$\alpha = 1.0$ (Public Portfolio / LinkedIn Bio):
"Software Engineer specializing in Graph Neural Networks, Spectral Graph Theory, and Applied AI applications in Julia and Python."
Why: Pure professional summary. No internal narrative, just verified capabilities.

So let us get the perfect mix:
$$\Delta_{\text{privacy}} = x_{\text{mask}} \ominus_M x_{\text{self}} = (-x_{\text{self}}) \oplus_M x_{\text{mask}}$$

Now we start:
$$u \oplus_M v = \frac{(1 + 2\langle u, v \rangle + \Vert{}v\Vert{}^2)u + (1 - \Vert{}u\Vert{}^2)v}{1 + 2\langle u, v \rangle + \Vert{}u\Vert{}^2 \Vert{}v\Vert{}^2}$$

I will skip the whole part and show only a small process:
$$\Delta_{\text{privacy}} = \frac{1.33 \cdot (-0.50, 0.30, 0.40) + 0.50 \cdot (0.20, 0.40, 0.10)}{1.225}$$$$\Delta_{\text{privacy}} = \frac{(-0.665 + 0.10,\; 0.399 + 0.20,\; 0.532 + 0.05)}{1.225} = \frac{(-0.565,\; 0.599,\; 0.582)}{1.225}$$$${ \Delta_{\text{privacy}} \approx (-0.461,\; 0.489,\; 0.475) }$$

That it, we got the mix we wanted, and now we will scale it, depending to who are we showing the idea:
$$\alpha \otimes_M \Delta = \tanh\left( \alpha \operatorname{artanh}(\Vert{}\Delta\Vert{}) \right) \frac{\Delta}{\Vert{}\Delta\Vert{}}$$

And we will get different result, depending by what we choose:
```
[ Core State ]
                   x_core = (0.50, -0.30, -0.40)
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
 α = 0.0 (Personal)   α = 0.4 (Collaborator)       α = 1.0 (Public Resume)
 x_view = x_self      x_view = (0.37, 0.01, -0.14)      x_view = x_mask = (0.20,                                                                      0.40, 0.10)
```

# Chapter 3. PREREQ - Riemannian Optimization

Now we go toward the Riemannian optimization! And this is the worst part probably. Now you will understand why!

## 1. Why vanilla SGD is wrong?

Now we have to understand why the normal Stochastic gradient descent is wrong in a poincare ball. 

Let us say that we are training an embedding (Okay, I will explain what training an embedding means. 

Imagine we have some clues:

"London is very close to Paris."
"Paris is far away from Tokyo."
"Tokyo is close to Osaka."

Now we will start the learning with some steps:

Step 1 (Place random points): Place random points on a paper sheet
Step 2 (Check the error): You look at your first clue ("London is close to Paris"). Your random dots placed them on opposite sides of the page! That is a huge error (or loss).
Step 3 (Adjustment): You nudge the dot for London a little closer to Paris, and push Tokyo further away.
Step 4 (Repeat Millions of Times): You repeat this process across billions of clues until every dot sits in a location that satisfies all the rules.
)

So the SGD is:

```
1. Start with Random Coordinates
       │
       ▼
2. Feed Training Data
       │
       ▼
3. Calculate Distance & Measure Error (Loss)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. NUDGE VECTOR COORDINATES                                  │
│    a. Backpropagation: Calculates the Gradient (direction)   │
│    b. SGD: Performs the actual coordinate update (the step)  │ ◄── SGD IS HERE
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
                 Loop until errors are minimal
```

Now I will explain why we will not use the normal SGD in a poincare ball.

Firstly we suppose that our AI uses the Poincare ball to store the embeddings.

```
Poincaré Ball

                 boundary ||x|| = 1
              .---------------------.
           .-'                       '-.
         .'                             '.
        /                                 \
       |                                   |
       |                ● x                |
       |                                   |
       |                                   |
        \                                 /
         '.                             .'
           '-._______________________.-'
```

We understand that the embedding doesn't have to leave the boundary - this is a strict rule.

Let us say that the AI has an embedding that is:
$$x = \begin{bmatrix} 0.7 \\\\ 0 \end{bmatrix}$$
This point is fine, since it is inside the poincare ball ($||x|| = 0.7 < 1$).

Now we will use the SGD, so we train the AI's embedding.
The formula is:
$${x_{\text{new}} = x - \eta \nabla L(x)}$$
I will break it piece by piece:
$x$:  Current point
$L(x)$:  Current loss
$\nabla L(x)$:  Gradient of the loss
$\eta$:  Learning rate (step size)
$x_{\text{new}}$:  Where SGD wants to put the point next

And here is the problem. Imagine that the gradient of the Loss is ($-\nabla L$) is $(-0.5, 0)$:

Let us assume that the learning rate is 1 ($\eta = 1$).
$$x_{\text{new}} = (0.7, 0) + (0.5, 0)$$$${x_{\text{new}} = (1.2, 0)}$$
Oops, we have left the poincare ball.
```
boundary
                            ↓
              .-----------------------.
           .-'                         '-.
         .'                               '.
        /                                   \
       |                           ● x      |
       |                             \       |
       |                              \      |
       |                               \     |
       |                                ●    |
        \                              x_new /
         '.                           OUTSIDE
           '-._______________________.-'
```

But why doesn't it happens in the euclidean space?
It doesn't happen because the Euclidean ML has no boundary.

If you update $x_{\text{new}} = x - \eta \nabla L(x)$ in flat space, you can land at $x_{\text{new}} = 1000$ or $x_{\text{new}} = -500$, and both points are completely valid.

Because the Euclidean space doesn't have boundary.

But the poincare ball has a constrain:
$$||x|| < 1$$
And the SGD doesn't know about any constrains, it just says: "Move this way"

But we may say something as: "make the learning rate smaller"
$$\eta = 0.00001$$
The problem is that the SGD doesn't still know about any boundary, so after a million of epoches we will get out of the ball.
And another problem is that the Vanila SGD assumes that we are using flat Euclidean geometry coordinates, while we are in the Poincare geometry, and the space is not flat, it is curved.

So this is the reason why the normal vanila SGD fails us inside the poincare ball. But we can do something about it...

## 2. Riemannian SGD

Since we understand why the vanila SGD fails in the poincare ball. We need to solve it, no? 
This is why we will change the vanila SGD to the Riemannian SGD.

So, instead of saying:
"Just move according to this ordinary Euclidean arrow."

We will say:
"Figure out the correct direction according to the geometry, then move, then make sure that the resulting point is on the manifold"

The steps will be:
$$\text{gradient} \rightarrow \text{tangent space} \rightarrow \text{step} \rightarrow \text{exponential map}$$

We have 3 major ideas:
1. Riemannian gradient
2. Tangent-space step
3. Exponential map

So we will slowly learn one at time.

We will start with the Tangent-space.

We worked with it when learning the geodesics. 
$$\dot\gamma(t)$$
This is the velocity/tangent vector of a curve.

So the tangent space is basically:
"All the direction in which you can locally move from a point on the manifold"

Imagine that you are a point ($x$) on the earth
```
                  Earth
               .----------.
            .-'            '-.
          .'                  '.
         /          ● YOU      \
         \                    /
          '.                .'
            '-.__________.-'
```

You can move:

- north
- south
- east
- west
- northeast
- etc.

This is all the local direction on the tangent space.
We can write it as:
$$T_x\mathcal{M}$$
where:
- T = tangent
- $(x)$ = the point we're standing at
- $(M)$ = the manifold

So:
$$T_x\mathcal{M}$$
Is basically: "The tangent space of manifold $M$ at point $x$"

Now we will still use the normal gradient and the loss function too.
Let us say that we have:
$$x = \begin{bmatrix} 0.3 \\\\ 0.2 \end{bmatrix}$$
and that the normal euclidean distance is:
$$\nabla L(x) = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

In ordinary SGD we would use:
$$-\nabla L(x) = \begin{bmatrix} -2 \\\\ -1 \end{bmatrix}$$

But the Poincare ball owns its own metrics tensor  $g_x = \lambda_x^2 I$, , where:
$$\lambda_x = \frac{2}{1 - \Vert{}x\Vert{}^2}$$
Before continuing, I will reveal why we do it and what it is.

The $x$ is simply our vector that represents a node, a word, or something else.
The $L$ is the loss.
The $\nabla L(x)$ is the Euclidean gradient that tells us the direction of steepest increase in flat coordinate space. This means that the (2, 1) is the fastest increase in loss, this is why we make them negative, so it does the opposite.

Now we will convert to the poincare metrics, so we change how our optimizer will measure two critical ideas: angle and distance.
$$\lambda_x = \frac{2}{1 - \Vert{}x\Vert{}^2}$$
Now we convert the vector:
$$\Vert{}x\Vert{}^2 = x_1^2 + x_2^2 = 0.3^2 + 0.2^2 = 0.09 + 0.04 = 0.13$$
If we square this answer, we will get the distance from the center.
$$||x|| = \sqrt{0.13} \approx 0.361$$
We understand that the point is 0.361 units away from the center of the ball.
$$\lambda_x = \frac{2}{1 - 0.13} = \frac{2}{0.87} \approx 2.299$$
This $(\lambda_x)$ is a geometry scaling factor at the particular point $(x)$.
It will tell us how much the poincare geometry scales ordinary Euclidean measurements at this location.
After getting the result ($2.299$) we can understand that the Poincare metric has a local scaling factor of approximately: $2.299$

But what does it mean?

Suppose you have a tiny Euclidean movement at \(x\):

$$ v=(0.01,0). $$

Its ordinary Euclidean length is:

$$ \|v\|=0.01. $$

But the poincare geometry says the local length is scaled by ($\lambda_x$), so the poincare length is:
$$∥v∥_{Poincare}=λ_x∥v∥.$$
We get:
$$ \approx2.299(0.01) $$ $$ \boxed{\approx0.02299} $$
So the tiny movement of (0.01) in Euclidean space corresponds to approximately $(0.023)$.

Now we will get the squared lambda.
$$\lambda_x^2 \approx (2.29885)^2 \approx 5.2847$$
After this we will convert the Euclidean gradient to the Riemannian gradient, by using:
$${\operatorname{grad}_R L(x) = \frac{1}{\lambda_x^2} \nabla L(x)}$$
In our case:
$$\operatorname{grad}_R L(x) \approx \frac{1}{5.286} \begin{bmatrix} 2 \\\\ 1 \end{bmatrix}$$
$${\operatorname{grad}_R L(x) \approx \begin{bmatrix} 0.378 \\\\ 0.189 \end{bmatrix}}$$
This vector is our geometry-adjusted gradient, living in the tangent space $T_x \mathcal{M}$.
Now we will add the learning rate ...:
$$v = -\eta \operatorname{grad}_R L(x)$$

Let us say that we choose 0.1 as learning rate:
$$v = -0.1 \begin{bmatrix} 0.378 \\\\ 0.189 \end{bmatrix}$$$${v = \begin{bmatrix} -0.0378 \\\\ -0.0189 \end{bmatrix}}$$
We did:
```
1. Start Point
                x = (0.3, 0.2) ∈ M
                        │
                        ▼
             2. Euclidean Gradient
               ∇L(x) = (2, 1)
                        │
                        ▼
      3. Convert via Metric Factor (1 / λ_x²)
          grad_R L(x) ≈ (0.378, 0.189)
                        │
                        ▼
        4. Apply Learning Rate (-η)
           v ≈ (-0.0378, -0.0189) ∈ T_x M
```

We just got the Riemannian gradient and the tangent space. 

Now we have two variables... 
$$x = \begin{bmatrix} 0.3 \\\\ 0.2 \end{bmatrix} \quad \text{and} \quad {v = \begin{bmatrix} -0.0378 \\\\ -0.0189 \end{bmatrix}}$$
But we don't need this, we want our new embedding... we need $x_{new}$, and to get it, it is not enough to simply subtract them, because we are in a hyperbolic space, not Euclidean. This is why we will use this formula:
$$\operatorname{Exp}_x(v)$$
So our $x_{new}$ will be:
$$x_{new} = \operatorname{Exp}_x(v)$$
So, the exponential map will make out of a tangent-space movement a actual point on the manifold.
Let us remember what a geodesic is. A geodesic is simply the curved-space version of the shortest path thanks to a straight line. Now... suppose that we have $x$ - which is a vector, and $v$ - which is the tangent vector. The exponential map constructs the geodesics as:
$$\gamma(t)$$
and such as:
$$\gamma(0) = x$$
and as:
$$\dot\gamma = v$$

Then the exponential map gives up:
$$\operatorname{Exp}_x(v) = \gamma(1)$$
So we are basically saying:
"Start at $x$, initially move with velocity v, follow the geodesic, and look at where you are at $t=1$."

So we understand now that $\gamma(0)$ is a point along the curve and $\dot\gamma(0)$ is our tangent vector.

So we understand:
$$\text{initial position} + \text{initial tangent direction} → \text{follow the geodesic}$$

The exponential map is equal to:
$${x_{\text{new}} = \operatorname{Exp}_x(v) = x \oplus \left( \tanh\left( \frac{\lambda_x \Vert{}v\Vert{}}{2} \right) \frac{v}{\Vert{}v\Vert{}} \right)}$$
Not kawaii at all, but let us break them on pieces:
$x$ is the current point in the Poincare ball.
$v$ is the tangent vector step.
$\lambda_x = \frac{2}{1 - \Vert{}x\Vert{}^2}$ is the Poincare metric scaling.
$\Vert{}v\Vert{}$ is the standard Euclidean norm of vector $v$.
$\frac{v}{\Vert{}v\Vert{}}$ is the unit direction vector in tangent space.


But why do we use $\tanh$ again? 
We use it to give some restriction to the tangent-space magnitude, because it may flee out of the poincare disk. So it strictly will stay under this range:
$$−1 < \tanh(z) < 1$$
And the full result will be:
$$x_{\text{new}} = \frac{1}{0.96557} \begin{pmatrix} 0.252511 \\ 0.174636 \end{pmatrix} \approx \begin{pmatrix} 0.2615 \\ 0.1809 \end{pmatrix}$$
So our new vector is:
$${x_{\text{new}} = \begin{pmatrix} 0.2615 \\ 0.1809 \end{pmatrix}}$$
Now we give it a validity check:
$$\Vert{}x_{\text{new}}\Vert{}^2 = 0.2615^2 + 0.1809^2 = 0.06838 + 0.03272 = 0.1011 < 1$$

That it. This is how the Riemannian SGD works. But don't worry, we will repeat it many times right now on the spot, so we don't have to memorize it.

Now I will make a full recap of almost everything, so be ready... it will be some random facts.

# Recap of everything

What is the hyperbolic space? 
The hyperbolic space is a non-Euclidean space that is characterized by its negative curvature.
The hyperbolic space grow exponentially, making it perfect for storing hierarchy trees in it - since both grow exponentially.
Because our physical world is Euclidean, we can't visualize it without projecting or bending it into standard coordinates. This is why math uses different models to visualize it.

---
1. The Lorentz model:

The hyperbolic space is visualized as the upper part of the hyperboloid in the Minkowski space. This is one of the most used models of hyperbolic geometry in the whole ML histroy.

Important formulas:
$$\langle x, y \rangle_M = -x_0 y_0 + x_1 y_1 + \cdots + x_n y_n$$
1. Negative - Timelike Vectors ($\langle x, x \rangle_M < 0$):

Here the inner product is negative and it means that the vector is inside the hyperboloid.

2. Positive - Spacelike Vectors ($\langle x, x \rangle_M > 0$):

Here, the inner product is positive and it means that the vector is outside the cone.

3. Zero -Null Vectors ($\langle x, x \rangle_M = 0$):

Here, the inner product is 0 and it means that the vector is on the edges of the cone.

The distance is actually:
$$d_{\mathbb{H}}(x, y) = arc\cosh(- \langle x,y\rangle_M)$$

---

2. The Poincare model:

The hyperbolic space is presented as a 2D disk with the boundaries that are equal to:
$$||x|| = 1$$
The Poincare model is perfect for visualization of the complex ideas in 2D.


There are some others type, which are not so useful for us right now.

---
When to use the Lorentz model and when to use the Poincare model?

- The Poincare ball is perfect to visualize hard concepts. It is a really good pick for Hierarchical Taxonomies & Trees, and it is good for some shallow machine learning.

- The Lorentz model is a really good choice for the Deep Hyperbolic Neural Networks (HNNs), Simple Closed-Form Operations, Top tier choice for deep hierarchies (since it grows exponentially.)

This is how and when they are used.

---
In a Poincare ball we will not use normal arithmetic, since the space is curved and it is not flat. If we directly add two vectors, we will end up taking a massive leap, and beside it, the closer we get to the boundaries, the bigger the leap we will get. This is why we use the Möbius gyrovector arithmetic, since it was made for the curved space.

Möbius addition:
$$x \oplus_M y = \frac{(1 + 2\langle x,y\rangle + \Vert{}y\Vert{}^2)x + (1 - \Vert{}x\Vert{}^2)y}{1 + 2\langle x,y\rangle + \Vert{}x\Vert{}^2\Vert{}y\Vert{}^2}$$
Möbius multiplication:
$$r \otimes_M x = \tanh(r \times ar\tanh(||x||)\frac{x}{||x||}$$
Möbius subtraction:
$$a \ominus b = a \oplus (-b)$$
$${a \ominus b = \frac{(1 - 2\langle a, b \rangle - \Vert{}b\Vert{}^2)a + (1 - \Vert{}a\Vert{}^2)(-b)}{1 - 2\langle a, b \rangle + \Vert{}a\Vert{}^2 \Vert{}b\Vert{}^2}}$$

---
Why the vanila Stochastic Gradient Descent fails in the hyperbolic space?
Because the vanila SGD is not aware of any curve or boundary, it just tells which way to go to lower the loss. 

The solution is the Riemannian SGD, since it moves along the manifold.

NEW:
The formula of the exponential map changes, based on the model we chose to represent the hyperbolic space.

If we choose the Poincare model, we get:
$$x_{t+1} = x_t \oplus \left( \tanh\left( \frac{\eta \lambda_{x_t} \Vert{}\nabla_R \mathcal{L}\Vert{}}{2} \right) \frac{-\nabla_R \mathcal{L}}{\Vert{}\nabla_R \mathcal{L}\Vert{}} \right)$$

If we choose the Lorentz model, we get:
$$x_{t+1} = \cosh(\Vert{}v\Vert{}_L) x_t + \sinh(\Vert{}v\Vert{}_L) \frac{v}{\Vert{}v\Vert{}_L}$$

---
What an embedding is? An embedding is simply a numerical representation of something, and if we see something as:

Dog:
$$x = \begin{bmatrix} 0.12, 0.64, -0.32... \end{bmatrix}$$
That doesn't mean that it means:
12 % of cuteness
64 % of dog-ish-ness.

They are just the geometry and distance between embedding encode relationship. 

Why we use hyperbolic space instead of Euclidean.? 

Because the Euclidean space would simply run out of space, because the Euclidean space will grow by 
$$2\pi r$$
While our beautiful hierarchy tree and taxonomy grow exponentially:
$$2^d$$
Just like this, we will run out of space in no time, for example, if we have 10 layer, we would have:
$2\pi (10) \approx 62.8$ units of perimeter
$2^{10} = 1,024$ leaf nodes 

And the more the depth grows, the more brutal the numbers get.

But the hyperbolic space grows by a similar rhythm, it grows exponentially too.

(Update:
I forgot to say... when I talked about the available space in Euclidean space, I didn't mean: "The Euclidean space will run out of space" - It can't run out, because it is infinite. But the available space is different. The available space means "how much area exists within a given distance from a point, or how much area is available to place many points while keeping them separated."

For example, if we have a ordinary number line:
```
<────────────●───────────>
             0
```

The ordinary number line is infinite, we can put infinitely many points on it. But suppose we want to represent one million different things in an Euclidean space... we want "similar" points to be close to each other, but we can't arbitrarily place everything everywhere without consequences.

So the main question becomes:
"How much room do I have near a particular point while maintaining particular distances between things?"

Now imagine we have a center point:
```
             ?
         ?       ?
      ?     ●      ?
         ?       ?
             ?
```

We may way that every point at distance $r=10$ from the center should be roughly separated from others. How many points can fit around that $r=10$ circle?

So if we have one main point as "animals", how many points can be fit around it? It depends by how far apart the points themselves must be. So if every child node needs 5 units if separation, then we will be able to put just:
$$\frac{62.8}5 \approx 12.56$$
Only 12 points.

Imagine the boundary as this:

```
                 LEVEL 3
          ╭─────────────────╮
         /                   \
        /     LEVEL 2         \
       /   ╭───────────────╮   \
      /   /                 \   \
     /   /    LEVEL 1        \   \
    /   /    ╭─────────╮      \   \
   │   │     │ Animals │       │   │
    \   \    ╰─────────╯      /   /
     \   \                   /   /
      \   ╰───────────────╯   /
       \                     /
        ╰───────────────────╯
```

- inner ring → general concepts
- middle ring → more specific concepts
- outer ring → very specific concepts

So we may have really deep hierarchies, with just the general concepts being over 1k points around the one specific parent point. But how do we get more room for it? We can easily get it thanks to the hyperbolic space, since it grows exponentially, not linearly. )

---

`What is a manifold?` As we already know, a manifold is a figure to which all the points can be locally represented in a euclidean space, while globally it is curved, abstract, or complex. 
We use manifolds because many real life items don't live in the euclidean-space. Manifolds give us a rigorous mathematical language to perform calculus, optimization, and so on...

`What is the tangent space?` The tangent space ($T_pM$) is a vector space that contains all the possible velocity vectors.
A velocity vector on the manifold is a point that moves along the smooth curve of the manifold while being strictly bound to its initial point ($p$). 
We will usually use two velocity vectors, which are $u$ and $v$.
$v$ will be bound to its point $p$, while $u$ will be bound to its point $q$

![[Pasted image 20260903082717.png|635]]

As we can see, they are two different tangent spaces on the manifold.

So we can easily say to people that we use it because the linear algebra breaks on a curved space, and we can't add or subtract from points, because we will break the geometry, and make the result vector leave the manifold or enter inside it. This is why we use tangent spaces! So they give us a flat, linear space where we can perform calculus, linear algebra, add vectors, subtract them, rotate them, and then after getting the result... we have the result in a flat space... not on the manifold... How will we even place them on the manifold? 
We will use the exponential map - the exponential map will take our final result and put it on the manifold!

Now we will... actually make a full U-turn on the geometry. So we will be able to make a production ready code 

Warning:
Ideas we already learned will repeat, but we will enter much deeper with intuition and everything possible, so don't worry much about the full code, because this are the baby steps.

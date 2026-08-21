
# Chapter 1. Spectral Graph

Already the name is badass, don't you think so? Because I do.

It will be pretty harsh as topic, but I'll do y best to explain it in a easy way and make codes with it.

Okay, so most of the tutorials would start by explaining what is a graph... but it looks like I already did explain it in the first month with the "graph_basics". That is why I will make a really fast recap.

Before we immediately say that a graph is the Cartesian coordinate plane, we have to remember about the existence of the nodes (vertex) and the existence of edges (links). The nodes can represent many stuffs, for example:
1. An atom
2. A person
3. An animal
4. A city

and much more.

While a link can represent:
1. The connection between those people
2. The bonds between atoms
3. The moneys transfer

and so on.

That is what we call a graph.
Now we will start with the first idea... which is our degree matrix.
## 1. Degree matrix

Now we can ask ourselves... what is the degree of each node for this graph?

![[Pasted image 20260807094420.png|516]]

Now let's count its degree.

A is connected to B and C.
- So A has:  `Degree = 2`

B is connected to A, C, and D.
- So B has: `Degree = 3`

C is connected to A and B:
- So C has: `Degree = 2`

D is connected just to B:
- So D has: `Degree = 1`

So it would look like this:
```Julia
Nodes = ["A", "B", "C", "D"]

Degree_node = [2 0 0 0; 
               0 3 0 0;
               0 0 2 0;
               0 0 0 1]
```

That is how a degree matrix looks on Julia.

But what exactly is a degree? A degree is just the number of edges connected to a node.

For example:
```
●────●────●
```

The middle node has a degree of 2.

```
     ●
     |
●────●────●
     |
     ●
```

The middle node has a degree of 4.

But why do we care? We care about it because the Degree measures the importance or connectivity of a node.

For example, imagine a social network:
```
Alice → 3 friends

Bob → 50 friends

Charlie → 400 friends
```

We immediately see that Charlie is more popular than Alice and Bob.

But now we may have a question... why do we write it in diagonal? Why to don't write it as: 
`2, 3, 2, 1`? 

We write it in a matrix so later we do matrix mathematics, since the main formula:
$$L = D -  A$$
And they must have the same shape. That is why we write them this way

and what this matrix explains us?
$$D = \begin{bmatrix} 2 & 0 & 0 & 0 \\\\ 0 & 3 & 0 & 0 \\\\ 0 & 0 & 2 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}$$

It explains each node individually as:

```
Node A → 2 neighbors

Node B → 3 neighbors

Node C → 2 neighbors

Node D → 1 neighbor
```

But what do we do  if we have a really big degree matrix?... do we write it by hand?
Nope.
We do not.

We write it as:
```Julia
using LinearAlgebra

degrees = [2, 3, 2, 1]

D = Diagonal(degrees)

"""
Output:

4×4 Diagonal Matrix

2 0 0 0
0 3 0 0
0 0 2 0
0 0 0 1
"""
```

We will explain later in codes what to do when we have a graph with 200 nodes...

## 2. Adjacency matrix

Okay, since we have the degree matrix:

|Node|Degree|
|---|--:|
|A|2|
|B|3|
|C|2|
|D|1|
We can draw the graph, right? No.
You can not, because the degree matrix tells us:
- Bob has 10 friends
- Alice has 12 friends
- Danny has 100 friends

But we don't know who are their friends, the only thing we know is the number of friends. That why we have the adjacency matrix.

The adjacency matrix will help us to identify each friend of the ndoe.

Let us go back to the image we already had:

![[Pasted image 20260807103134.png|485]]

We may write the degree, but then we will not know who to who is connected. 
That is why we use the adjacency matrix. Let's use it.

`Nodes = ["A", "B", "C", "D"]`

Who are A's neighbors? - B and C.
```Julia
A = [0 1 1 0] # Node A
```

Who is B's neighbors? - A, C, D
```Julia 
A = [0 1 1 0; # Adjacency of Node A
     1 0 1 1] # Adjacency of Node B
```

Who is C's neighbors? - A and B
```Julia
A = [0 1 1 0; # Adjacency of Node A
     1 0 1 1; # Adjacency of Node B
     1 1 0 0] # Adjacency of Node C
```

Who is D's neighbors? - B
```Julia
A  = [0 1 1 0; # Adjacency of Node A
     1 0 1 1; # Adjacency of Node B
     1 1 0 0; # Adjacency of Node C
     0 1 0 0] # Adjacency of Node D
```

From this we can understand who is which neighbor. 
Matrix form:
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\\\ 1 & 0 & 1 & 1 \\\\ 1 & 1 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \end{bmatrix}$$

Let us take the row 2.
$$A = \begin{bmatrix} 1 & 0 & 1 & 1 \end{bmatrix}$$
We can read this as:
```
B is connected to A

B is NOT connected to B

B is connected to C

B is connected to D
```

But we can notice something else...
The diagonal is always 0.
$$A_{11} = A_{22} = A_{33} = A_{44} = 0$$
But that normal. because the node can't connect to itself.

But as we noticed... We can't build an adjacency matrix from the degree matrix, but we can do vice versa. 

Since knowing with who the matrix connects and seeing how many connection it has, we can calculate the degree with a simple formula:
$$Degree(i) = \sum_j A_{ij}$$
Now we can continue with Graph Laplacian.

## 3. Graph Laplacian

Here is where the spectral graph theory start becoming a tad more serious.

We will start immediately with the formula and explain it, so we don't have to memorize it step by step.
$$L = D - A$$
...
...

Now we may think... "Is he joking? All this philosophy to explain a single degree matrix and the adjacency matrix just to say that the main formula is just the Degree matrix getting subtracted by the Adjacency matrix?"

Well, yes. But we will understand why we do so and with what this helps us.

What does this formula even does?
Let's take back our Degree and Adjacency matrix.
Degree matrix:
$$D = \begin{bmatrix} 2 & 0 & 0 & 0 \\\\ 0 & 3 & 0 & 0 \\\\ 0 & 0 & 2 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Adjacency matrix:
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\\\ 1 & 0 & 1 & 1 \\\\ 1 & 1 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \end{bmatrix}$$

Now let us use the formula:
$$L = D - A$$
we will get 
$$L = \begin{bmatrix} 2 & -1 & -1 & 0 \\\\ -1 & 3 & -1 & -1 \\\\ -1 & -1 & 2 & 0 \\\\ 0 & -1 & 0 & 1 \end{bmatrix}$$

Let's see what it tells us. We will take the first row:
$$L = \begin{bmatrix} 2 & -1 & -1 & 0 \end{bmatrix}$$
`nodes = ["A", "B", "C", "D"]`

We notice the `2`. This `2` comes from the Degree matrix and it says: "Node A has 2 neighbors". 
Then we notice that Node B has -1 as value... why? Because it tells us: "Node A is connected to Node B", same idea for Node C, beside Node D - doesn't connect since it has a value of 0.

So we can understand that diagonally, we have the degree of the Nodes, and everything outside the diagonals is either -1 or 0, why?
Because if two nodes are connected
we write
```
-1
```

Otherwise
```
0
```

And here we have another question - Why do we subtract?
can't we just do?
$$L = D + A$$

Now I will tell you why.

Imagine that each node has its own temperature 
```
A = 100°C

B = 20°C

C = 40°C

D = 10°C
```

Naturally, the heat flows from the hotter node to the colder node. Node A doesn't care only about its own temperature.
That why Laplacian measures an exact idea: "How different am I from the nodes connected to me?"

For example, if the nodes are identical:
```
A = 10

B = 10

C = 10

D = 10
```

Nobody should change, the graph is balanced, and Laplacian will give a 0.
But how it will?

Now I will explain.

How can the Laplacian Graph show how different is something compared to another?
We know, that all the nodes stores something, for example, let's say that the node stores neural network features. 

```
A = 10
B = 10
C = 10
D = 10
```

Each node store have 10 neural network features.

That why we make a matrix:
$$x = \begin{bmatrix} 10 \\\\ 10 \\\\ 10 \\\\ 10 \end{bmatrix}$$

```julia
x =   [10;
       10;
       10;
       10]
```

This is what we call a graph signal.
The graph signal is just the matrix of all the values of each node.

Now I will give you a real example:

Imagine that our graph looks like:
```
(A) ----- (B) ----- (C) ----- (D)
```

We write down the Degree matrix and the Adjacency matrix.
`nodes = ["A", "B", "C", "D"]`

$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\\\ 1 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & 1 \\\\ 0 & 0 & 1 & 0 \end{bmatrix}, \quad D = \begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 2 & 0 & 0 \\\\ 0 & 0 & 2 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Now we use Laplacian:
$$L = D - A$$
That will give us:
$$L = \begin{bmatrix} 1 & -1 & 0 & 0 \\\\ -1 & 2 & -1 & 0 \\\\ 0 & -1 & 2 & -1 \\\\ 0 & 0 & -1 & 1 \end{bmatrix}$$
Now we will follow this formula:
$$(Lx)_i = \sum_{j \in \text{Neighbors}(i)} (x_i - x_j)$$

Now we will multiply it by the graph signal.
$$Lx = \begin{bmatrix} 1 & -1 & 0 & 0 \\\\ -1 & 2 & -1 & 0 \\\\ 0 & -1 & 2 & -1 \\\\ 0 & 0 & -1 & 1 \end{bmatrix} \begin{bmatrix} 10 \\\\ 10 \\\\ 10 \\\\ 10 \end{bmatrix}$$
$$Lx = \begin{bmatrix} 10 - 10 + 0 + 0 \\\\ -10 + 20 - 10 + 0 \\\\ 0 - 10 + 20 - 10 \\\\ 0 + 0 - 10  + 10 \end{bmatrix} \quad Lx = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 0  \end{bmatrix}$$
Because every node has the exact same value as its neighbors, there is zero signal change across every edge in the graph.
Think about it 4 rooms (A, B, C, D) that are straight connected in a line, but separated by doors.

 Let us say that each of them have a temperature of 27 ℃. If we open all the doors, will something change? Will room A start heating a bit room B? No. Because they are all in a perfect equilibrium.

But the main idea is: "How different is node $i's$ value from the values of its neighbors?"

I will give another idea:
Let's say we have we have again (Just smaller):
```
1 ----- 2 ----- 3
```

Let's say that the signal graph looks like this:
$$x = \begin{bmatrix} 10 \\\\ 12 \\\\ 15 \end{bmatrix}$$

And the Laplacian looks like:
$$Lx = \begin{bmatrix} 1 & -1 & 0 \\\\ -1 & 2 & -1 \\\\ 0 & -1 & 1 \end{bmatrix}$$

We will solve it in no time:

$$Lx = \begin{bmatrix} 1 & -1 & 0 \\\\ -1 & 2 & -1 \\\\ 0 & -1 & 1 \end{bmatrix} \begin{bmatrix} 10 \\\\ 12 \\\\ 15 \end{bmatrix}$$

$$(Lx)_1 = 10 - 12 = -2$$
$$(Lx)_2 = -10 + 24 - 15 = -1$$
$$(Lx)_3 = -12 + 15 = 3$$

And the result will be:
$$Lx = \begin{bmatrix} -2 \\\\ -1 \\\\ 3 \end{bmatrix}$$
What this means? 

Node 1: `-2 -> smaller than its neighbors` - because Node 2 is bigger by 2.
Node 2: `-1 -> smaller than its neighbors` - because it is bigger than Node 1 by `2` and smaller than Node 3 by `3 -> -3` = 2 - 3 = -1 
Node 3: `3 -> bigger than its neighbors` - because it is bigger than Node 2 by `3`

But we can still do something about this result. We can use this formula (I'll explain latter what it means) - the Graph Dirichlet Energy:
$$x^TLx$$
Since we know what our $Lx$ is equal to, we are just going to compute the formula:
$$x^T = \begin{bmatrix} 10 & 12 & 15 \end{bmatrix}$$
This is just our signal graph.
$$x^T Lx = \begin{bmatrix} 10 & 12 & 15 \end{bmatrix} \begin{bmatrix} -2 \\ -1 \\ 3 \end{bmatrix}$$
Now we just solve it:
$$10(-2) + 12(-1) + 15(3)$$
$$= -20 - 12 + 45$$
$$=13$$
This is the total variation (or energy) of the values $x$ across the graph's edges (In poor words: The variation of the graph at its edges - the connections of the graph, as: $E={(1,3),(2,3),(3,4)}$). And 13 in our case it is not a really big number, yet not too small. So the variation is medium.
## 4. Eigendecomposition and Graph Fourier

### 1. Eigendecomposition basic idea

Why do we study eigenvalues again?
Let us remember what we did back in analytic geometry.
$$Ax^2 + Bxy + Cy^2...$$
By finding the eigenvalues we discovered something really important. We discovered the shape of the figure. 
For example, if both eigenvalues were positive, that meant that the shape will be an ellipse, if one was negative and the other positive, then it was a hyperbola, and in the end, if it had one zero, it was a parabola.

The eigenvalues summarized the geometry.

That is why we will do the same stuff with the graphs, by using the eigendecomposition. Because we will not inspect every edge one by one.

The eigendecomposition will do it for us:
$$L = U \Lambda U^T$$

But what does this formula reveals to us? 
This formula will reveal some really important ideas, as:
- Is the graph connected?
- Does it naturally split into communities?
- How many connected components does it have?
- Is information trapped inside groups?
- How "smooth" is a signal on the graph?

All from one decomposition.

Later we will learn even what the eigenvectors reveal.

Firstly we will make a fast recap. 
As we summarized geometry... we will summarize even the graph. But why?

Imagine having a graph like this:
```
A ----- B
|
|
C
```

Can we read it and write everything we need? Absolutely.

But let's say that we have 100 nodes.
Can we read them and understand what we are doing?
Still yes, a tad more hard, but fine.

What if we have 1 millions of nodes?
Can we read them and understand what we are doing?
Absolutely not.

That is why we need someone to summarize it.

But firstly I will teach you again what is an eigenvalue and eigenvector.

### 2. Filler arc (Still imporant)

#### First eigenvalue

As we know a matrix changes a vector by stretching it shrinking it, rotating it and so on ($Ax$). 
In the end both length and direction changes.

But an eigenvector doesn't change its direction, it only changes the size ($Av$). 
The eigenvector is the matrix that we will use and the eigenvalues tell by how much it will shrink or stretch.

But now we replace the idea. 
We will replace $A$ with $L$ .

What does change? 
Well, by using $A$ we asked ourselves this question: "Which direction stays pointing the same?"
When using $L$ we ask ourselves another question: "Which patterns on the graph stay the same?"

Now imagine this:
```
A = 10

B = 20

C = 5

D = 17
```

As we spoke about it, this is the signal on the graph (graph signal).

$$x = \begin{bmatrix} 10 \\\\ 20 \\\\ 5 \\\\ 17 \end{bmatrix}$$
This is not a vector with coordinates, each of this value can represent:
- temperature
- traffic
- pollution
- income

or whatever else.

We know that by getting the $Lx$, we will discover the difference of each neighbor, since Laplacian compares every node with its neighbors.

Now we will use the changed version of the equation:
$$Av = \lambda v$$
We will do:
$$Lv = \lambda v$$

Let's take an already done Laplacian as:
$$L = \begin{bmatrix} 2 & -1 & -1 & 0 \\\\ -1 & 3 & -1 & -1 \\\\ -1 & -1 & 2 & 0 \\\\ 0 & -1 & 0 & 1 \end{bmatrix}$$
Let's make $v$ to be as simple as possible.
$$v = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$
Now we solve this part: ($Lv$)
$$Lv = \begin{bmatrix} 2 & -1 & -1 & 0 \\\\ -1 & 3 & -1 & -1 \\\\ -1 & -1 & 2 & 0 \\\\ 0 & -1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$

First row:
$$2(1) - 1(1) - 1(1) + 0(1)$$
$$2 - 1 - 1 + 0 = 0$$
Second row:
$$-1 + 3 - 1 - 1$$
$$= 0$$
Third row:
$$-1 - 1 + 2 + 0$$
$$=0$$
Fourth row:
$$0 - 1 + 0 + 1$$
$$=0$$

Therefore:
$$L \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix}$$

So since we have:
$$0 = \lambda v$$
So, as expected.
$$\lambda = 0$$

So that how we found one eigenpair.
$$\lambda = 0 \quad v = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$

But would the first eigenvalues always be 0? Yes. Because we have its degree (suppose 5), and since it has the degree of 5, it means that there are 5 connections (five -1)
So we have:
$$L = \begin{bmatrix} 5 & -1 & -1 & -1 & -1 & -1 \end{bmatrix}$$
Now we choose our eigenvector (easier would be to say... 10)

$$v = \begin{bmatrix} 10 \\\\ 10 \\\\ 10 \\\\ 10 \\\\ 10 \end{bmatrix}$$
So now we just solve it:
$$Lv = (5 \times 10 ) +(-1 \times 10) +(-1 \times 10) +(-1 \times 10) +(-1 \times 10) +(-1 \times 10) = 0$$

The first eigenvalue is always 0. Even if it may seem useless... at least it says to you that everything is working perfectly, that the graph is fine. It is like the chapter 1 of the book - it is not the most interesting part, yet without it everything would be confusing.

Now we can learn about the second smallest eigenvector (We forgot... about the eigendecomposition, but whatever, because we are learning the foundation)

#### Second smallest eigenvalue (Fiedler value)

Let us forget about the first eigenvalue. Now we continue with the second smallest eigenvalue (That the name it got, so I will not touch its name.)

Let's look at this graph here:
![[Pasted image 20260807160146.png|559]]

We can already see something... they look like two separate groups.
Here you can see that they look like two separate groups, but what about having 100 million nodes? You will never even notice which are the groups

But let's say somebody said to us:
"Split this graph in two communities".

We would easily do:
![[Figure_1 5.png|559]]

Because there was just one edge connecting them, it looked like a good split

But now let us look at another graph:
![[Pasted image 20260807161544.png|463]]

Can we split into two communities? Yes... but hardly, this would cut way too many edges.

So already we have a small idea:
Some graphs split easily while other do not.

But now... imagine having 10k nodes, it is too hard to check each. That is why we will need a number that would tell us if we can split easily or if the graph is really hard to split...

for example, if the number is `0.01` - really easy to split, if the number is `25` - really hard to split.

And luckily, this is what the second eigenvalue will do for us.

The first eigenvector looked like this:
```
1
1
1
1
1
1
```

We get no information by it.

The second may look like:
```
1
0.9
1.1
-0.8
-1
-1.2
```

This time, this give us information, because we can notice a pattern...
```
Positive

↓

One community

Negative

↓

Another community
```

But from where those numbers came from?
They came from 
```julia
eigen(L) # Spoiler, spoiler - we will not do all by hand, we will use codes too get the answer for us.
```

But now imagine that our data looks like this
```
1
1
1
-1
-1
-1
```

This is smooth, we can easily separate them. 

What about:
```
20

400

-17

900

-200
```

This is not smooth at all, this is why we can't even get close, call back the Gini cruddiness.

But jokes aside.

The second eigenvalue (Fielder vector - that the cool name it actually has) tell us how easy it split the graph into communities and the eigenvector tells us where the graph wants to split.

Let us give an example:

![[Pasted image 20260807184747.png|509]]

```Julia
using Graphs
using LinearAlgebra

g = SimpleGraph(20)

edges_A = [
    (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,1),
    (1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,1) # We made tuples. So every tuple is one connection
]

edges_B = [
    (11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,11),
    (11,14),(12,15),(13,16),(14,17),(15,18),(16,19),(17,20),(18,11)
]

bridges = [
    (1,11),
    (3,14),
    (5,16),
    (7,18),
    (9,20)
]

for e in vcat(edges_A, edges_B, bridges)
    add_edge!(g, e...)
end

degrees = degree(g)
D = Diagonal(degrees)

A = Matrix(adjacency_matrix(g))

L = D - A

F = eigen(Matrix(L))

perm = sortperm(F.values)

λ = F.values[perm]

V = F.vectors[:, perm]

println("Eigenvalues")

for (i,value) in enumerate(λ)
    println("λ$i = ", round(value, digits=6))
end
print("")
print("Fiedler vector")
println(V[:,2])

"""
Output:

Eigenvalues

λ1 = 0.0
λ2 = 0.80197
λ3 = 1.633368
λ4 = 2.033176
λ5 = 2.763044
λ6 = 2.969278
λ7 = 3.0
λ8 = 3.152492
λ9 = 3.377797
λ10 = 3.51346
λ11 = 4.239184
λ12 = 4.317461
λ13 = 4.8925
λ14 = 5.0
λ15 = 5.565771
λ16 = 5.813048
λ17 = 6.065177
λ18 = 6.485774
λ19 = 7.91581
λ20 = 8.46069


Fiedler value
[0.20320115530285, 0.24790194868545096, 0.16294157842330903, 0.2317608096918977, 0.17875325726716984, 0.22996517063704683, 0.19628209734496985, 0.24255403731060757, 0.19745864722398737, 0.2715803798821273, -0.14075257160255591, -0.34106582481764225, -0.3327397419946005, -0.18853582648349923, -0.27618069862984673, -0.20177037872297296, -0.20474767563644414, -0.15186220840701964, -0.21212330155999573, -0.1126208539148378]
"""
```

Let's break the code the code to pieces.

1.  `g = SimpleGraph(20)` - With this code we simply create the graph with 20 nodes with no connections, so think about them with 20 islands that don't have a bridge.

2. ```julia
   for e in vcat(edges_A, edges_B, bridges)
    add_edge!(g, e...)
   end
   ```

what does `vcat(edges_A, edges_B, bridges)` means?
`vcat` (vertical concatenate) - It simply joins vectors together. Since edges_A has 18 edges, edges_B has 18 edges, and bridges has 5 edges, we will get a big vector with 41 edges.

what about `add_edges!(g, e...)`?
`add_edges!` will actually add the edges to the nodes, yet there is one problem... `add_edges!` expects an input as: `graph, node, node` not `graph, tuples(node)`, that is why we put `g` in the `graph` section, and in the `node` one we write `e...` which actually expands the tuples from `(1, 2)` to `1, 2` so now we have two nodes: `g, 1, 2`.

3. `degrees = degree(g)` - this checks the degrees of each node, since it sees that `1` is connected to other `5` nodes, it will write `1` degree as 5, and it looks through all the nodes, and in the end gets a value as `[5,3,4,4,...]` - all the degrees in a vector.
4. `D = Diagonal(degrees)` - this will check the `len(degrees)` and then for each degree we have it will just add up all the zeros and so we will get a matrix:
```
 5 0 0 0...
 0 3 0 0...
 0 0 4 0...
 0 0 0 4...
 ......
```

5. `A = Matrix(adjacency_matrix(g))` - here we will automatically make the adjacency matrix (We actually use `Matrix` to make the adjacency matrix into a matrix instead of returning a sparse one that we don't need.)
6. `F = eigen(Matrix(L))` - this is our Laplacian's eigenvalues and vectors. To get just the eigenvalues we will write `F.values`, while to get the eigenvectors we will write `F.vectors`

7. `perm = sortperm(F.values)` - this will sort our data, because imagine having:

```
3.2
0
7.4
1.8
...
```

That not really convenient, that is why we will do:
```
0
1.8
3.2
7.4
...
```

But we have to remember that `sortperm[F.values]` returns the index, not the actually ordered variable

That is why we write
`λ = F.values[perm]` - so it actually orders them

 8. `println(V[:,2])` - this will print the 2nd eigenvector (The eigenvector of our Fielder value)

Second try:

![[Pasted image 20260808103757.png|653]]

```Julia
using Graphs
using LinearAlgebra

g = SimpleGraph(17)

edges_A = [
(1, 2), (2, 4), (2, 5), (3, 6), (3, 7), (4, 8), (4, 9), (4, 5), (5, 12)
]

edges_B = [
(6, 13), (6, 14), (6, 15), (7, 16), (7, 17), (9, 11), (9, 10), (10, 11), (15, 16)
] 

bridges = [
(1, 3), (5, 6)
]

for e in vcat(edges_A, edges_B, bridges)
	add_edge!(g, e...)
end

degrees = degree(g)
D = Diagonal(degrees)

A = Matrix(adjacency_matrix(g))

L = D - A

F = eigen(Matrix(L))

perm = sortperm(F.values)

λ = F.values[perm]
V = F.vectors[:, perm]

for (i, values) in enumerate(λ)
	println("λ$i = ", round(values, digits=6))
end

print("")
print("Fiedler vector")
println(V[:, 2])

"""
Ouptut:

λ1 = -0.0
λ2 = 0.161301
λ3 = 0.41647
λ4 = 0.552424
λ5 = 0.771868
λ6 = 1.0
λ7 = 1.0
λ8 = 1.0
λ9 = 2.090819
λ10 = 2.533575
λ11 = 2.911612
λ12 = 3.0
λ13 = 3.525687
λ14 = 4.331454
λ15 = 4.844414
λ16 = 5.36583
λ17 = 6.494547

Fiedler vector

[0.06274489739433699, -0.04776482006324739, 0.16313382419290307, -0.17042921646812256, -0.02790564617212625, 0.14434517476424616, 0.2559978139353949, -0.20320655995311332, -0.37534950192452393, -0.4475375915400226, -0.4475375915400227, -0.03327252497794065, 0.17210597465346955, 0.172105974653469, 0.21900278075293933, 0.258335103096443, 0.3052319091959127]
```

The Fiedler value is relatively small ($\lambda2 = 0.161301$), that means that somewhere in this graph we can separate it in 2 communities and don't loose too much strong bonds.

Let us look at the eigenvectors:

```
node    Fiedler value
 1       +0.062745
 2       -0.047765
 3       +0.163134
 4       -0.170429
 5       -0.027906
 6       +0.144345
 7       +0.255998
 8       -0.203207
 9       -0.375350
10       -0.447538
11       -0.447538
12       -0.033273
13       +0.172106
14       +0.172106
15       +0.219003
16       +0.258335
17       +0.305232
```

We can separate the positive from the negative:
```
	                 Fiedler cut
	                   |
	      POSITIVE     |       NEGATIVE
	                   |  
	  1   3   6   7    |   2   4   5   8
	 13  14  15  16    |  9  10  11  12
	 17                |
	─────────────────────────────────────────────
	                 cut
```

We understand where to cut already.

### 3. Signal processing and Graph Fourier Transform

We already know what is a signal in a graph ($x$ - the value of each of our nodes (Can be stock price, temperature, age, and so on...))

But what is actually signal processing? 
Think about signal processing as taking complex signals, breaking them in simple, fundamental building blocks (frequencies/patterns), modifying those blocks, and putting them back to their place.

Imagine striking three keys on a piano simultaneously: low C, middle C, and high C.

To us, it sounds beautifully
The microphone records it as a single squiggly wave.
But the Fourier Analysis acts like an automatic ear. It analyzes that single messy wave and tells you: "This signal contains 40% Low C, 35% Middle C, and 25% High C."

Once we broke the signal into those frequencies, we can apply the filler.

Low-Pass Filter (Bass Boost): Keep the Low C, silence the High C.
High-Pass Filter (Treble Boost): Keep the High C, suppress the Low C.

Let's look at some rooms and see their temperatures:
![[Pasted image 20260808152357.png|663]]

The temperatures are not really different from each other, so there is no need fixing something.

But what about this:
![[Pasted image 20260808152750.png|663]]

Now we notice a hazard - Room 3. It is really different from its neighbors (a really different graph signal).

Now this is where the Signal processing begins.

In ordinary signal processing we ask a question like:
What frequencies/patterns make up this signal?"

Instead, the Fourier Analysis says:
"This complicated signal can be represented using simpler waves."

As for:
$$\text{signal = slow wave + medium wave + fast wave}$$

So the idea is: 
"A complicated signal can be expressed as a combination of simpler patterns."

But there is a problem... Fourier analysis expects from us:
```
Room 1 — Room 2 — Room 3 — Room 4 — Room 5
```

or equivalently:
```
x₁ → x₂ → x₃ → x₄ → x₅
```

But we don't have the rooms arranged in a cute straight line.

Our rooms literally look like this:
```
        1
      /    \
     2 ---- 3
     |      |
     4 ---- 5
      \    /
        6
```

Which is obviously not a straight line.
But we ask ourselves a question:

"What does a "Frequency" means in a graph?"

This is where we will use our Laplaci (Endearing name for Laplacian) (introduction of D, A...)
$$D = \begin{bmatrix} 2 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 3 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 3 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 3 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 3 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 2 \end{bmatrix}$$
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 & 0 & 0 \\\\ 1 & 0 & 1 & 1 & 0 & 0 \\\\ 1 & 1 & 0 & 0 & 1 & 0 \\\\ 0 & 1 & 0 & 0 & 1 & 1 \\\\ 0 & 0 & 1 & 1 & 0 & 1 \\\\ 0 & 0 & 0 & 1 & 1 & 0 \end{bmatrix}$$

And now we use the Laplacian:
$$L = D - A$$
$$L = \begin{bmatrix} 2 & -1 & -1 & 0 & 0 & 0 \\\\ -1 & 3 & -1 & -1 & 0 & 0 \\\\ -1 & -1 & 3 & 0 & -1 & 0 \\\\ 0 & -1 & 0 & 3 & -1 & -1 \\\\ 0 & 0 & -1 & -1 & 3 & -1 \\\\ 0 & 0 & 0 & -1 & -1 & 2 \end{bmatrix}$$

As we know, for a simple unweighted graph, where:
- `A` tells us which rooms are connected
- `D` tells us how many connections each room has
- `L` combines this information.

We can use $Lx$ and discover the difference of each node compared to their neighbors.
$$x = \begin{bmatrix} 22 \\\\ 21 \\\\ 35 \\\\ 20 \\\\ 21 \\\\ 19 \end{bmatrix}$$
and now we compute the $Lx$
$$Lx = \begin{bmatrix} -12 \\\\ -14 \\\\ 41 \\\\ -1 \\\\ -11 \\\\ -3 \end{bmatrix}$$

So now we can easily answer to this question:
"How different is each room's temperature from the temperatures of its neighbors?"

So let's analyze a tad the answers:
1. Node 1 - `-12`. It is by`1` bigger than Node 2, but smaller by `-13` by Node 3:
   1 - 13 = -12
2. Node 2 - `-14`. It is bigger than Node 4 by `1`, smaller than Node 1 by `-1`, and smaller by `-14` by Node 3:
   1 - 1 - 14 = -14
3. Node 3 - `41`. It is bigger than Node 1 by `13`, bigger than Node 2 by `14`, and bigger than Node 5 by `14`:
   13 + 14 + 14 = 41

and so on...

Now let us stare at this other part:
```
22
 |
21
 |
35
 |
20
```

If we do:
$$22−21=1$$
$$21−35=−14$$
and:
$$35−20=15$$

We can clearly say that the graph has a rough signal, and this is what we mean by graph frequency

- Low graph frequency = values change slowly across connected nodes.
- High graph frequency = values change rapidly across connected nodes.

So... how do we find this frequencies?
In regular Signal processing we would've have used sine waves:
$$\sin(t) \quad \sin(2t) \quad \sin(3t)…$$
this are our foundational patterns. Yet our graph is not a straight line. This is why... here the eigendecomposition of our beautiful Laplaci will help us.
$$Lu_i = \lambda_iu_i$$

After solving it, we would get the:
- eigenvectors: $u_i$
- eigenvalues: $\lambda_i$

And the important interpretation is:
	$$\lambda_1 = \text{graph frequency of} \ u_i$$
	The eigenvectors are the vibration shapes, and the eigenvalues are the vibration speeds (frequencies). - This is how we represent the signal of the graph using eigenvectors and eigenvalues.

Now we will write down all the eigenvalues.

Now we solve the formula above:
$$Lu = \lambda u$$
Doing the whole operation is a long journey... yet I will show it to you.
$$\det(L - \lambda I) = 0$$
We will firstly find the eigenvalues by using this formula, where $I$ is the identity matrix (It will have as many rows and columns as our L. It is identical to the degree matrix, it is just that all the Diagonal values are 1.), so multiplying $\lambda$ by the identity matrix will give use a degree matrix where the diagonal is $\lambda$. So we will do just subtraction and get):
$$L - \lambda I = \begin{bmatrix}  2-\lambda & -1 & -1 & 0 & 0 & 0 \\\\  -1 & 3-\lambda & -1 & -1 & 0 & 0 \\\\  -1 & -1 & 3-\lambda & 0 & -1 & 0 \\\\  0 & -1 & 0 & 3-\lambda & -1 & -1 \\\\  0 & 0 & -1 & -1 & 3-\lambda & -1 \\\\  0 & 0 & 0 & -1 & -1 & 2-\lambda  \end{bmatrix}$$

Now we get the determinant = $det(L - \lambda I)$:
$$P(\lambda) = \lambda^6 - 16\lambda^5 + 98\lambda^4 - 284\lambda^3 + 381\lambda^2 - 180\lambda = 0$$
That our polynomial. Now we use elementary class strategy - factoring out (firstly we factor out our $\lambda$):
$$\lambda \left(\lambda^5 - 16\lambda^4 + 98\lambda^3 - 284\lambda^2 + 381\lambda - 180\right) = 0$$

Now we factor out completely:
$$\lambda (\lambda - 1)(\lambda - 3)^2 (\lambda - 4)(\lambda - 5) = 0$$

So our eigenvalues are literally:
$$\lambda \in \{0, 1, 3, 3, 4, 5\}$$
- $\lambda_1 = 0$
- $\lambda_2 = 1$
- $\lambda_3 = 3$
- $\lambda_4 = 3$
- $\lambda_5 = 4$
- $\lambda_6 = 5$

From here we understand the frequency - the first eigenvalue has the lowest frequency (as usual), the second has a low frequency, the third has a medium variation, and so on.

Now we will find the eigenvectors (I will try to explain the idea.)

The first unnormalized eigenvector is:
$$u_1 = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$
Now we check the numerical verification:
$$L \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 2(1) - 1(1) - 1(1) \\\\ -1(1) + 3(1) - 1(1) - 1(1) \\\\ -1(1) - 1(1) + 3(1) - 1(1) \\\\ -1(1) + 3(1) - 1(1) - 1(1) \\\\ -1(1) - 1(1) + 3(1) - 1(1) \\\\ -1(1) - 1(1) + 2(1) \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix}$$

Now we will normalize the eigenvector:
$$u_1 = \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$
We normalized it, so we take away the arbitrary scale. Because it will become constantly annoying for Fourier transform. (We got the $\frac{1}{\sqrt{6}}$ by using this idea: $$\frac{1}{\sqrt{|\vert\mathbf{u_i}|\vert}}$$
So we fixed many problems. You will not have to do it manually, because on Julia it is automatic (getting the eigenvectors and using this idea).

The second eigenvector will be:
$$u_2 = \frac{1}{\sqrt{12}} \begin{bmatrix} 2 \\\\ 1 \\\\ 1 \\\\ -1 \\\\ -1 \\\\ -2 \end{bmatrix}$$
This is the Fiedler vector. And it will have a really special part later in the spectral clustering.
We will get all the eigenvectors, and multiply them by the scalar we just put beside them, so they get normalized.
### 4. Continuation of eigendecomposition (GFT continuation)

After getting all the eigenvectors we will add them to the orthogonal (Vectors are perpendicular to each other) matrix $U$:
$$U = \begin{bmatrix} \vert & \vert & \vert & \vert & \vert & \vert \\\\ u_1 & u_2 & u_3 & u_4 & u_5 & u_6 \\\\ \vert & \vert & \vert & \vert & \vert & \vert \end{bmatrix}$$

So the result is:
$$U \approx \begin{bmatrix} 0.4082 & 0.5774 & 0.0465 & 0.5755 & -0.4082 & 0 \\\\ 0.4082 & 0.2887 & 0.4752 & -0.3280 & 0.4082 & 0.5000 \\\\ 0.4082 & 0.2887 & -0.5216 & -0.2475 & 0.4082 & -0.5000 \\\\ 0.4082 & -0.2887 & 0.4752 & -0.3280 & -0.4082 & -0.5000 \\\\ 0.4082 & -0.2887 & -0.5216 & -0.2475 & -0.4082 & 0.5000 \\\\ 0.4082 & -0.5774 & 0.0465 & 0.5755 & 0.4082 & 0 \end{bmatrix}$$
Actually, just for us to understand:
$$U^T U = I$$
That make us understand that the columns of $U$ form an orthonormal (The vectors are perpendicular to each other $u_1 \perp u_2$ (They are orthogonal at the same time) and they are normalized - each of them have a length (magnitude) of exactly 1) basis for the $6$-dimensional signal space.

So you can visualize a tad better the idea...:
![[Orthonormal.png|630]]

We can clearly see that the orthogonal has a different magnitude but they are perpendicular, while the orthonormal are exactly 1 and they are perpendicular.

Now here comes the interesting part, because we already have all the components we got from the eigendecomposition, we can answer this question: "How much of each graph pattern is present in x".

This is like the standart Fourier analysis, which asks: How much $100\text{ Hz}$? How much $200\text{ Hz}$? How much $500\text{ Hz}$?

Here we will ask:
Here, we ask: How much of graph frequency $\lambda = 0$? How much of graph frequency $\lambda = 1$? How much of graph frequency $\lambda = 3$?

Since the eigenvectors form an orthonormal basis, we will write $x$ as:

$$x = \alpha_1 u_1 + \alpha_2 u_2 + \dots + \alpha_6 u_6$$

We need to compute the scalars $\alpha_1, \alpha_2, \dots, \alpha_6$, which are our Graph Fourier coefficients.
This is why we will have to arrange them in a vector:
$$\hat{x} = \begin{bmatrix} \alpha_1 \\\\ \alpha_2 \\\\ \alpha_3 \\\\ \alpha_4 \\\\ \alpha_5 \\\\ \alpha_6 \end{bmatrix}$$

And since $U$ is orthonormal ($U^{-1} = U^T$), we can multiply both sides by $U^T$:
$$\hat{x} = U^T x$$
This is what we call the Graph Fourier transform.

Using this idea we can write it as:
$$\alpha_1 = u_1^T x$$
So we can get each coefficient:
$$\alpha_1 = \frac{1}{\sqrt{6}} \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 22 \\\\ 21 \\\\ 35 \\\\ 20 \\\\ 21 \\\\ 19 \end{bmatrix}$$

$$\alpha_1 = \frac{22 + 21 + 35 + 20 + 21 + 19}{\sqrt{6}} = \frac{138}{\sqrt{6}} \approx \mathbf{56.34}$$

Important to know... $56.34$ is just the spectral energy coefficient not $56.34℃$.

Now we will do:
$$\alpha_1 u_1 = \left( \frac{138}{\sqrt{6}} \right) \left(\ \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix} \right) = \frac{138}{6} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \end{bmatrix}$$
This is the base we will use.
I will not place all the operations, because it would take too much time, that is why I will just put the result:
$$\hat{x} \approx \begin{bmatrix} 56.338 \\\\ -6.062 \\\\ -2.828 \\\\ -9.899 \\\\ -4.899 \\\\ -4.500 \end{bmatrix}$$

So in the end we got:
```
ROOM DOMAIN                          GRAPH FREQUENCY DOMAIN
┌──────────────────────┐                 ┌─────────────────────────┐
│ Room 1: 22°C         │                 │ λ = 0  →   56.338 (DC)  │
│ Room 2: 21°C         │                 │ λ = 1  →  -6.062 (Low)  │
│ Room 3: 35°C (Spike) │   ───── Uᵀ ────►│ λ = 3  →  -2.828 (Mid)  │
│ Room 4: 20°C         │   ◄──── U ───── │ λ = 3  →  -9.899 (Mid)  │
│ Room 5: 21°C         │                 │ λ = 4  →  -4.899 (High) │
│ Room 6: 19°C         │                 │ λ = 5  →  -4.500 (High) │
└──────────────────────┘                 └─────────────────────────┘
```

Once we perform the the Graph Fourier Transform, we will not have the initial values of the room, because they will be represented as a spectrum of frequency weights.

So, to filter out the anomaly, we will have to make a filter that help us against high spikes.
For example, we will do:
$$\lambda \le 1 \quad \And \quad \lambda \ge 3$$
This will be our filter.
To build a Graph Low-Pass Filter, we define a filter matrix $H(\Lambda)$ that acts like a mask:$$H(\Lambda) = \begin{bmatrix}  1 & 0 & 0 & 0 & 0 & 0 \\  0 & 1 & 0 & 0 & 0 & 0 \\  0 & 0 & 0 & 0 & 0 & 0 \\  0 & 0 & 0 & 0 & 0 & 0 \\  0 & 0 & 0 & 0 & 0 & 0 \\  0 & 0 & 0 & 0 & 0 & 0  \end{bmatrix}$$
we will multiply our frequency vector $\hat{x}$ by the filter:
$$\hat{x}_{\text{low}} = H(\Lambda) \hat{x} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} 56.338 \\\\ -6.062 \\\\ -2.828 \\\\ -9.899 \\\\ -4.899 \\\\ -4.500 \end{bmatrix} = \begin{bmatrix} \mathbf{56.338} \\\\ \mathbf{-6.062} \\\\ 0 \\\\ 0 \\\\ 0 \\\\ 0 \end{bmatrix}$$
But, sadly, showing the result in frequency space is not helpful, so we will have to convert back to ℃.
This is why we will use the  Inverse Graph Fourier Transform (iGFT):$$x_{\text{filtered}} = U \hat{x}_{\text{low}}$$
So it will look like:
$$x_{\text{filtered}} = \alpha_1 u_1 + \alpha_2 u_2 + 0 u_3 + 0 u_4 + 0 u_5 + 0 u_6$$

Firstly we will get the DC component:
$$\alpha_1 u_1 = 56.338 \cdot \frac{1}{\sqrt{6}} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix} = \begin{bmatrix} 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \end{bmatrix}$$
Now we will get the second component:
$$\alpha_2 u_2 = (-6.062) \cdot \frac{1}{\sqrt{12}} \begin{bmatrix} -2 \\\\ -1 \\\\ -1 \\\\ 1 \\\\ 1 \\\\ 2 \end{bmatrix} \approx -1.75 \begin{bmatrix} -2 \\\\ -1 \\\\ -1 \\\\ 1 \\\\ 1 \\\\ 2 \end{bmatrix} = \begin{bmatrix} 3.50 \\\\ 1.75 \\\\ 1.75 \\\\ -1.75 \\\\ -1.75 \\\\ -3.50 \end{bmatrix}$$
Now we will sum them:
$$x_{\text{filtered}} = \begin{bmatrix} 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \\\\ 23 \end{bmatrix} + \begin{bmatrix} 3.50 \\\\ 1.75 \\\\ 1.75 \\\\ -1.75 \\\\ -1.75 \\\\ -3.50 \end{bmatrix} = \begin{bmatrix} 26.50 \\\\ 24.75 \\\\ 24.75 \\\\ 21.25 \\\\ 21.25 \\\\ 19.50 \end{bmatrix}$$

As we can see, the 35 ℃ became a simple 24.75 ℃. Because we filtered the spike successfully.  
Now we will use it on Julia and spam a tad of practice before continuing:
```julia
using Graphs
using LinearAlgebra

g = SimpleGraph(6)

edges = [(1, 2), (1, 3), (2, 3), (4, 6), (5, 6), (4, 5)]
bridges = [(2, 4), (3, 5)]
x = [22, 21, 35, 20, 21, 19]

for e in vcat(edges, bridges)
    add_edge!(g, e...)
end

L = Matrix(laplacian_matrix(g)) # We get immediately the Laplacian matrix without searching the degree and adjacency matrix

F = eigen(L) 

λ = F.values
U = F.vectors

x_hat = U' * x # That is our Graph Fourier Transform

H = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0] # This will be our filter

x_hat_filtered = H .* x_hat # We wilter everything and leave just the first two results

x_filtered = U * x_hat_filtered # This is our new result!

println("Graph Frequencies (λ): ", round.(λ, digits=3))
println("Raw Signal (x):       ", x)
println("Filtered Signal:       ", round.(x_filtered, digits=2))

"""
Output:

Graph Frequencies (λ): [-0.0, 1.0, 3.0, 3.0, 4.0, 5.0]

Raw Signal (x):       [22, 21, 35, 20, 21, 19]

Filtered Signal:       [26.5, 24.75, 24.75, 21.25, 21.25, 19.5]
"""
```

Now I will do the same, just with another graph.

![[Network_spike.png|700]]

I will do all on Julia:
```Julia
using Graphs
using LinearAlgebra

g = SimpleGraph(10)

rack_A = [(1,2), (2,3), (3,4), (4,5), (5,1), (1,3)]

rack_B = [(6,7), (7,8), (8,9), (9,10), (10,6), (6,8)]

bridges = [(3,8), (5,10)]

for e in vcat(rack_A, rack_B, bridges)
    add_edge!(g, e...)
end

# Server Traffic Signal (GB/s)
x = [20.0, 22.0, 19.0, 21.0, 20.0, 15.0, 95.0, 14.0, 16.0, 15.0]

Lx = L * x

L = Matrix(laplacian_matrix(g))

println("Graph energy: $(x'Lx)")

"""
Output:

Graph energy: 13036.0
"""
# Something is wrong then, but where?

for (i, values) in enumerate(x)
    if values >= 25
        println("anomaly spotted at Node: $i -> $values")
    end
end

"""
Output:

anomaly spotted at Node: 7 -> 95.0
"""
# Let's solve it

F = eigen(L)
λ = F.values
U = F.vectors

x_hat = U' * x

H = zeros(10)
H[1: 3] .= 1

x_hat_filtered = H .* x_hat

x_filtered = U * x_hat_filtered


println("Graph Frequencies (λ): ", round.(λ, digits=3))
println("Raw Signal (x): ", x)
println("Filtered Signal: ", round.(x_filtered, digits=2))

"""
Output:


Graph Frequencies (λ): [0.0, 0.576, 1.382, 1.659, 2.382, 3.192, 3.618, 4.618, 4.639, 5.933]

Raw Signal (x):       [20.0, 22.0, 19.0, 21.0, 20.0, 15.0, 95.0, 14.0, 16.0, 15.0]

Filtered Signal:       [21.6, 30.93, 24.39, 4.95, 7.86, 39.66, 52.38, 36.87, 20.64, 17.72]
"""
```

Now we will continue, and in the end of the topics we will do many codes based on real life experience (With what you will work)

## 5. Normalized Laplacian (L_sym), Random-walk Laplacian (L_rw)

Sadly, nothing is perfect in this life. That is why even the Laplacian has its own problem.
Let me explain which.

Remember our Dirichlet Energy formula:
$$x^T L x = \sum_{(i,j) \in E} (x_i - x_j)^2$$
(We used it many times, it is just that we used to get $Lx$ firstly and then multiply it by $x^T$)

The problem is that... let us imagine a twitter network:
- Node A (Popular hub) - Connected to 1000 people
- Node B (Quiet person) - Connected to 2 people

Let us say that Node A changes opinion by 1 point, while Node B changes it by 10 points . 
Here we will use the Dirichlet Energy formula:
$$\text{Node A}: \quad 1^2 + 1^2 + ... 1^2 = 1000$$
$$\text{Node B:} \quad 10^2 + 10^2 = 200$$
Now we will see the difference:
$$\frac{1000}{200} = 5 $$
So despite B changing $10\times$ more, A will contribute 5 times more than B.
That is why we would use the Normalized Laplacian:
$$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$
Now I will show you how it works:
$$D = \begin{bmatrix} 1000 & 0 \\\\ 0 & 2 \end{bmatrix}$$
Now we will take $D^{-1 / 2 }$ :
$$D = \begin{bmatrix} \frac{1}{\sqrt{1000}} & 0 \\\\ 0 & \frac{1}{\sqrt{2}} \end{bmatrix}$$
And now we scale them as:
$$x → D^{-1 / 2}x$$
$$\text{Popular Hub} = 1 → \frac{1}{\sqrt{1000}} \approx 0.0316$$
$$\text{Quiet Person} = 10 → \frac{10}{\sqrt{2}} \approx 7.071$$

Now we see that the one who changed by 10 points has a bigger contribution.
Beside it, let us see their degree-weighted contribution:
$$d_A = 1000, \quad x_A = 1 \quad \text{The popular hub}$$
$$\begin{aligned} 1000 \left(\frac{1}{\sqrt{1000}}\right)^2 \\ = 1000 \frac{1}{1000} \\ = 1 \end{aligned}$$

Now the quiet person:
$$d_B = 2, \quad x_B = 10 \quad $$
$$\begin{aligned} 2 \left(\frac{10}{\sqrt{2}}\right)^2 \\ = 2 \frac{100}{2} \\ = 100 \end{aligned}$$
And this is the fixed version!
But here is even another idea... Why do we have even the Random-walk Laplacian?

Because our beautiful Normalized Laplacian is really good for Linear Algebra and Deep Learning. Meanwhile the Random-walk Laplacian is really good for probabilities.

The formula is:
$$L _{rw}= D^{-1} A$$

Let's use it once:
$$A = \begin{bmatrix} 0 & 1 & 1 & 0 \\\\ 1 & 0 & 1 & 1 \\\\ 1 & 1 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \end{bmatrix}$$

Let us say that the degrees are:
$$d_1 = 2, \quad d_2 = 3, \quad d_3 = 2, \quad d_4 = 1.$$
So now we will put them:
$$D^{-1} = \begin{bmatrix} 1/2 & 0 & 0 & 0 \\\\ 0 & 1/3 & 0 & 0 \\\\ 0 & 0 & 1/2 & 0 \\\\ 0 & 0 & 0 & 1 \end{bmatrix}$$

So the result becomes
$$P = \begin{bmatrix} 0 & 1/2 & 1/2 & 0 \\\\ 1/3 & 0 & 1/3 & 1/3 \\\\ 1/2 & 1/2 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \end{bmatrix}$$
That what we had to understand from both. 
Stick always to Normalized Laplacian whenever you will do Linear algebra and Deep Learning.
Or
Stick to Random-walk Laplacian when working with the probability.

## 6. K-mean (Off topic)

I skipped K-mean, so now I will have to teach you some basics of it... since we will use it for spectral clustering. 
Uhhhhhh, I no no wanna, but I have to.

### 1. What problems K-mean solves?

Let's say that we have 6 data points:
$$X = \begin{bmatrix} 1 & 1 \\\\ 1 & 2 \\\\ 2 & 1 \\\\ 8 & 8 \\\\ 8 & 9 \\\\ 9 & 8 \end{bmatrix}$$

They will look exactly as:
![[Data_Matrix.png|556]]

But there is a problem, nobody is going to tell us which belongs to what group?

We don't have any labels as:
```
Point 1 → Group A
Point 2 → Group A
Point 3 → Group A
Point 4 → Group B
...
```

That is why K-mean helps us.
It will say something as:
"I'll represent each group by a point called a centroid, and I'll assign each data point to the centroid closest to it."

So if we tell K-means:
$$k = 2$$
It is like saying: "I believe there are 2 clusters, so find them."
Then it will eventually find.
$$\mu_1 \approx (1.33,1.33)$$
and 
$$\mu_2 \approx (8.33, 8.33)$$

So now we will have:
```
        ● ● ●
        \ | /
          μ₁


                         μ₂
                       / | \
                     ●   ●  ●
```

Each point belongs to whatever centroid is closer.

But why is it called "K-mean"?
There are two components that represent it.

- K - that the number of clusters we want.
  For example, if we write: $K = 2$, that means that we are asking K-mean to find two groups.
- Mean - To make the threshold of the centroid, it finds the mean of the points. For example:
$$(1,1),\quad(1,2),\quad(2,1)$$
Now we find their mean.
For $x$:
$$\frac{1 + 1 + 2}3 = \frac{4}3$$
For $y$:
$$\frac{1 + 2 + 1}3 = \frac{4}3$$
Result:
$$\mu_1 = (\frac{4}3,\frac{4}3)$$
So, K-mean = K clusters represented by their mean

### 2. Distances and the nearest centroid

Before I start spouting some 2 half-baked lies and one truth - I am going to spoiler you the end.

This is all about the the distance from analytic geometry.
$$d=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$$
So I'll give a fast example about this one.

Imagine Airi has two centroids:
$$\mu_1 = (1, 1)$$
$$\mu_2 = (8, 8)$$
And the point:
$$x = (2, 3)$$

Now she has a question... to which of this centroids will this point belong?

Now we will have to calculate the distance of $x$ to $\mu_1$ and the distance of $x$ to $\mu_2$. 
Let's start:
$$\Vert{}x - \mu_1\Vert{}_2 = \sqrt{(2 - 1)^2 + (3 - 1)^2}$$$$= \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$$
and now the distance of $x$ to $\mu_2$:
$$\Vert{}x - \mu_2\Vert{}_2 = \sqrt{(2 - 8)^2 + (3 - 8)^2}$$$$= \sqrt{(-6)^2 + (-5)^2}$$$$= \sqrt{36 + 25} = \sqrt{61} \approx 7.810$$
And we know that:
$$2.236 < 7.810$$

Therefor $x$ will go to the first centroid, because it has a smaller distance. In a mathematical formula it looks like:
$$C(x_i) = \text{arg min}_j||x_i - \mu_j||_2$$
This will automatically choose the nearest centroid for $x$.

### 3. Why initialization matters?

We already know that the centroids get positioned based on the mean of the points.
But there is a problem... Where do the first centroids come from?

Imagine we have:
```
       ● ● ●
      ● ● ●

                    ● ● ●
                   ● ● ●

                                     ● ● ●
                                    ● ● ●
```

And we put:
$$K = 3$$
Now K-mean needs 3 initial centroids:
$$μ_1​,\quad μ_2​, \quad μ_3​$$

For now, imagine that we choose them randomly from the data points.

Imagine we get:
```
       μ₁
       ● ● ●
      ● ● ●

                    μ₂
                    ● ● ●
                   ● ● ●

                                     μ₃
                                    ● ● ●
```

But randomness is a scary stuff, so it may give us even:
```
       μ₁  μ₃
       ● ● ●
      ● ● ●

                    μ₂
                    ● ● ●
                   ● ● ●

                                     
                                    ● ● ●
```

Now two centroids started at the same cluster, and another cluster is left without any centroid.

Yeah, we may ask ourselves: "But... doesn't K-mean actually move the centroids repeatedly, so it figure it out, right?" 
Maybe, but that not guaranteed. It may try, but then fail a tad and get a result that doesn't really improve the mistake.

That is why K-mean++ exists (Yeah, they just slapped two pluses and now they got from C; C++).

### 4. K-mean++ (The $D(x)^2$ probability)

We understand that putting several centroids at the same cluster is pretty bad as idea - due to the randomness.

That is why K-mean++ fixes this problem.

Imagine we have this points:

| Point | Coordinates |
| ----- | ----------- |
| $x_1$ | $(1, 1)$    |
| $x_2$ | $(2, 1)$    |
| $x_3$ | $(1, 2)$    |
| $x_4$ | $(8, 8)$    |
| $x_5$ | $(9, 8)$    |
| $x_6$ | $(8, 9)$    |

And we want:
$$K =2$$
So now let us explain all the steps.
Firstly, K-mean++ will chooses the first centroid uniformly at random.
Suppose it chooses:
$$\mu_1 = (1, 1)$$
Now it will count the distance for each point:
$$D(x_i) = \Vert{}x_i - \mu_1\Vert{}_2$$
We will find the distance of each $x_i$:
$$D(x_1) = 0 \quad \text{- this is beacuse this is already the centroid and the distance from A to A is 0.}$$
$$D(x_2)^2 = 1 \quad \text{Same distance because it doesn't matter from which side you measure, A to B = B to A}$$
$$D(x_3)^2 = 1$$
$$D(x_4)^2 = 98$$
$$D(x_5)^2 = 113$$
$$D(x_6)^2 = 113$$

Now we calculate the denominator:
$$0 + 1 + 1 + 98 + 113 + 113 = 326$$
And now the probability:
$$P(x_1) = \frac{0}{326} = 0$$$$P(x_2) = \frac{1}{326} \approx 0.0031$$$$P(x_3) = \frac{1}{326} \approx 0.0031$$$$P(x_4) = \frac{98}{326} \approx 0.3006$$$$P(x_5) = \frac{113}{326} \approx 0.3466$$$$P(x_6) = \frac{113}{326} \approx 0.3466$$
So, as we can see, it tries to place the next centroid as far as possible.
Since there is a centroid near $x_1, x_2, x_3$ , it tries to place a centroid as far as possible from them.

But why did we use $D(x_i)^2$ instead of just $D(x_i)$?
One of the main reasons is:
If Point B is $5\times$ farther from a centroid than Point A:
- Under $D(x)$, Point B is $5\times$ more likely to be picked.
- Under $D(x)^2$, Point B is $25\times$ more likely to be picked.

This is one of the main reasons.

This would look like this on Julia:
```julia
using Random
using Clustering

Random.seed!(42)

# 2 features (rows) x 6 samples (columns)
X = [
    1.0  1.0  2.0  8.0  9.0  8.0 ;  # Feature 1
    1.0  2.0  1.0  8.0  8.0  9.0    # Feature 2
]
# I know, this is scary... using columns as samples and features as rows is what we will do on Julia - sadly.

k = 2

result = kmeans(X, k; maxiter=300)

cluster_assignments = assignments(result) # Point labels
centroids = result.centers # Matrix of centroids
inertia = result.totalcost

println("Cluster Assignments: ", cluster_assignments)
println("Centroids:\n", centroids)

"""
Cluster Assignments: [2, 2, 2, 1, 1, 1]

Centroids:
[8.333333333333334 1.3333333333333333; 8.333333333333334 1.3333333333333333]
"""
```

What the results mean? 
- `Cluster Assignments:`

This is where our points got assigned, as:
- Points 1, 2, 3 $\to$ Assigned to Cluster 2
- Points 4, 5, 6 $\to$ Assigned to Cluster 1

- `Centroids:`

This is literally the values that the centroids have:
- $\mu_1 = (8.333, 8.333)$
-  $\mu_2 = (1.333, 1.333)$

That all we needed of K-mean, now we can continue with Spectral Clustering.

## 7. Spectral clustering

So... why do we need spectral clustering in the first place?

- Imagine having a graph with:
- people connected to people,
- webpages connected by hyperlinks,
- computers communicating with computers,
- papers connected by citations,
- proteins connected by interactions,
- customers connected by similar behavior.

So what do you do? Of course you have to organize them somehow, because organizing them is much better than standing and stare at one of those thousands of nodes and while thinking:
"Which nodes naturally belong together?"

That is already a clustering problem.

So why not using just K-mean? 
Because K-mean is bad at being the central point... in graphs like:
```
       ● ● ●
     ●       ●
    ●         ●
     ●       ●
       ● ● ●
```

Now imagine if I say that A and B have a strong connection. We can already imagine them getting in the same centroid.
Even if D is more far:
```
A---B---C---D
```

So even if D is more far from A, we don't look after euclidean distance, we look id these nodes are strongly connected to one and weakly connected to the rest of the graph.

So to perform the spectral we will follow this piepline:
$$G→A→D→L→\text{eigenvectors}→\text{new coordinates}→\text{K-Means}$$
The interesting part is:
$$\text{new coordinates}→\text{K-Means}$$
This is what we call spectral embedding.

And now I will give you an example of how to do it by hand, and how to do it in Julia.

Example:

![[B_P_Graph.png|501]]

Here we have a graph with 8 nodes.
As we can notice, it has a weak bridge, because they are connected by just one edge.

Now let us say that the blue ones are male friends and how related they are, while the pink ones are female friends.

Now we want to group them in two groups; boys and girls.
This is why we will use the spectral clustering.

So let us go by the pipeline we talked about.
$$G→A→D→L→\text{eigenvectors}→\text{new coordinates}→\text{K-Means}$$

Firstly we will do basic steps.

We have 8 nodes and now we will write the adjacency matrix to see which Node is connected to whom.
$$A = \begin{bmatrix} 0 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\\\ 1 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\\\ 1 & 1 & 0 & 1 & 0 & 0 & 0 & 0 \\\\ 1 & 1 & 1 & 0 & 1 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 1 & 0 & 1 & 1 & 1 \\\\ 0 & 0 & 0 & 0 & 1 & 0 & 1 & 1 \\\\ 0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 \\\\ 0 & 0 & 0 & 0 & 1 & 1 & 1 & 0 \end{bmatrix}$$
Now we understand which to which is connected to whom.

Let's find the degree matrix (Random walk) and discover connections each node has.
$$D^{-1} = \text{diag}\left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}, \frac{1}{4}, \frac{1}{4}, \frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right)$$

But here we will use the Random-walk. Because the random walk normalizes the fact that all the nodes have a degree of 3, and two have a degree of 4. So instead of making the ones with a degree of 4 have a really big value in the final embedding coordinates. $L_{rw}$ normalizes with the idea of:  multiply each row by $\frac{1}{d_i}$ to normalize its influence of the node, regardless of its degree.
$$L_{\text{rw}} = I - D^{-1}A = I - P$$
Where $P = D^{-1}A$ as the stochastic transition matrix of a random walk on the graph.
$$L_{\text{rw}} = \begin{bmatrix} 1 & -\frac{1}{3} & -\frac{1}{3} & -\frac{1}{3} & 0 & 0 & 0 & 0 \\ -\frac{1}{3} & 1 & -\frac{1}{3} & -\frac{1}{3} & 0 & 0 & 0 & 0 \\\\ -\frac{1}{3} & -\frac{1}{3} & 1 & -\frac{1}{3} & 0 & 0 & 0 & 0 \\\\ -\frac{1}{4} & -\frac{1}{4} & -\frac{1}{4} & 1 & -\frac{1}{4} & 0 & 0 & 0 \\\\ 0 & 0 & 0 & -\frac{1}{4} & 1 & -\frac{1}{4} & -\frac{1}{4} & -\frac{1}{4} \\\\ 0 & 0 & 0 & 0 & -\frac{1}{3} & 1 & -\frac{1}{3} & -\frac{1}{3} \\\\ 0 & 0 & 0 & 0 & -\frac{1}{3} & -\frac{1}{3} & 1 & -\frac{1}{3} \\\\ 0 & 0 & 0 & 0 & -\frac{1}{3} & -\frac{1}{3} & -\frac{1}{3} & 1 \end{bmatrix}$$

Now we continue with the next step:
$$L→\text{eigenvectors}→\text{new coordinates}→\text{K-Means}$$
We have to find the eigenvalues and eigenvectors.
For that, we will use this formula (The one we always use):
$$L_{wr}u = \lambda u$$
Since we have 8 nodes and our Laplacian is a matrix 8x8, we will get 8 eigenvalues and corresponding eigenvectors.
$$\begin{aligned} \lambda_1 &\approx 0 \\\\ \lambda_2 &\approx 0.113382 \\\\ \lambda_3 &\approx 1.083333 \\\\ \lambda_4 &= 1.333333 \\\\ \lambda_5 &= 1.333333 \\\\ \lambda_6 &= 1.333333 \\\\ \lambda_7 &= 1.333333 \\\\ \lambda_8 &\approx 1.469951 \end{aligned}$$
We can look at $\lambda_2 \approx 0.113382$ that a really small value, meanwhile, already the third eigenvalue ($\lambda_3 \approx 1.083333$) is a huge jump. This is going to matter.

Now we will find the first eigenvalue. 
Since the first eigenvalue is 0, we can choose our eigenvalue to be 1.
$$u_1 = \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$
Now we just normalize it:
$$u_1 = \frac{1}{\sqrt{8}} \begin{bmatrix} 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \\\\ 1 \end{bmatrix}$$
So the first eigenvector is actually:
$$u_1 \approx \begin{bmatrix} 0.3536 \\\\ 0.3536 \\\\ 0.3536 \\\\ 0.3536 \\\\ 0.3536 \\\\ 0.3536 \\\\ 0.3536 \\\\ 0.3536 \end{bmatrix}$$

Now the important eigenvector is this one:
$$u_2 = \begin{bmatrix} 0.3815 \\\\ 0.3815 \\\\ 0.3815 \\\\ 0.2517 \\\\ -0.2517 \\\\ -0.3815 \\\\ -0.3815 \\\\ -0.3815 \end{bmatrix}$$

Now we analyze the result for each node:

| Node |     $(u_2)$ |
| ---: | ----------: |
|    1 | $(+0.3815)$ |
|    2 | $(+0.3815)$ |
|    3 | $(+0.3815)$ |
|    4 | $(+0.2517)$ |
|    5 | $(-0.2517)$ |
|    6 | $(-0.3815)$ |
|    7 | $(-0.3815)$ |
|    8 | $(-0.3815)$ |

We noticed immediately something. The two communities:
$$\underbrace{+, +, +, +}_{1,2,3,4} \quad \underbrace{-, -, -, -}_{5,6,7,8}$$
But now we may have a question. Why aren't the values +1 or -1?
This is an interesting question.

But this is because we didn't start with two communities as:
```
1──2     5──6
│╲ │     │╲ │
│ ╲│     │ ╲│
3──4     7──8
```

If we had something as this, then we were supposed to have: $\lambda_2 = 0$
and then we would've a vector of 4 positive ones and 4 negative ones.
But since we have a bridge: $4 → 5$
This is why our eigenvalue is bigger than 0.
And we can notice that all the values are identical, beside two smaller ones, they are our weak bridge. ($0.2517→−0.2517$)

Now we have the big picture in the spectral space.
We could represent our whole graph with just the second eigenvector ($u_2$).
We would simply do:
$$u_2(i) > 0 ⇒ C_1 \quad and \quad u_2(i) < 0 ⇒ C_2$$
This actually works for such a simple graph. 
But the spectral clustering goes a step ahead, and doesn't stop just here. Because when we will need more centroids, a single eigenvector is not enough to represent all the centroids.
That is why we will use more eigenvectors:
$$y_i = \begin{bmatrix} u_1(i) \\ u_2(i) \\ \vdots \\ u_K(i) \end{bmatrix}$$
In our example, let us say we need just $K = 2$ , so that means we will use just the first 2 eigenvectors.
Now we will put them aside, next to each other:
$$U = \begin{bmatrix} \vert & \vert \\\\ u_1 & u_2 \\\\ \vert & \vert \end{bmatrix}$$
So we will have a matrix (8x2)
$$U \approx \begin{bmatrix}  0.3536 & 0.3815 \\\\  0.3536 & 0.3815 \\\\  0.3536 & 0.3815 \\\\  0.3536 & 0.2517 \\\\  0.3536 & -0.2517 \\\\  0.3536 & -0.3815 \\\\  0.3536 & -0.3815 \\\\  0.3536 & -0.3815  \end{bmatrix}$$

As we already know... the K-mean needs the coordinates. But do we have the coordinates on the graph? Nope. But maybe we noticed, that the result is under our nose. This is the our modal matrix ($U$). 
Now we have just to get our points for each node:
$$\boxed{y_1 = (0.3536, 0.3815)}$$$$\boxed{y_2 = (0.3536, 0.3815)}$$$$\boxed{y_3 = (0.3536, 0.3815)}$$$$\boxed{y_4 = (0.3536, 0.2517)}$$and:$$\boxed{y_5 = (0.3536, -0.2517)}$$$$\boxed{y_6 = (0.3536, -0.3815)}$$$$\boxed{y_7 = (0.3536, -0.3815)}$$$$\boxed{y_8 = (0.3536, -0.3815)}$$
This is the coordinates of each of our nodes. Now we can represent them in the euclidean space.
This is why we call it the spectral space/spectral embedding 
![[Spectral.png]]

Now we would use the K-mean++ and get the centroids placed in their perfect position. 

Now we will represent it on Julia.
```julia
using Clustering
using Random
using LinearAlgebra
using Graphs
Random.seed!(9)

g = SimpleGraph(8)

c1_edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
c2_edges = [(5, 6), (5, 7), (5, 8), (6, 7), (6, 8), (7, 8)]

bridge = [(4, 5)]

for e in vcat(c1_edges, c2_edges, bridge)
    add_edge!(g, e...)
end

degrees = degree(g)
D = Diagonal(degrees)

A = Matrix(adjacency_matrix(g))

L_rw = I - D \ A

F = eigen(L_rw)

K = 2
U = F.vectors[:, 1:K]

X = U'

result = kmeans(X, K, maxiter=300)

cluster_assignments = assignments(result)
centroids = result.centers


println("Cluster Assignments: ", cluster_assignments)
println("Centroids:\n", centroids)

"""
Output:

Cluster Assignments: [1, 1, 1, 1, 2, 2, 2, 2]

Centroids:
[0.3535533905932736 0.3535533905932739; 0.34905961405546665 -0.3490596140554661]
"""
```

This is how the spectral clustering works. Now we will continue with a really useful topic... MLflow and Dockers... will I make them in a new md? No. Because they are short as topics.

I will explain all fastly (Maybe you will learn a tad by yourself about them.)


# Chapter 2. MLflow and Dockers

## 1. What is MLflow?
Now... I wanted to introduce MLflow. But why? Because I got too nostalgic and wanted to check on Python again? Nah!
But all the ML engineers/Ai engineers and so on, are forced to know for their own safety MLflow.

Why?

Because MLflow will solve a really interesting problem.
Let us see.

Imagine you are training a model:
```python
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)
```

Now suppose that your accuracy is of 0.94 - good.

But you want to get a better accuracy.... so you change the learning rate, epoches, and other aspects several times... but you best result was 0.94, so now you may think — Perfect! Now I will just write back the best result... hmmmm... which was the configuration that gave us 0.94? Nobody knows. This is why we use MLflow.

MLflow is a tool for tracking and managing out ML experiments and models.

But what exactly does it records?

There are actually 4 concepts it keeps track of:

1) Parameters:
this is our:
```
learning_rate = 0.001
batch_size = 64
epochs = 20
max_depth = 5
```
We choose and modify them. MLflow records them.

2) Metrics
This are actually the results that are produced by our model.
They look exactly like:
```
accuracy = 0.94
loss = 0.21
precision = 0.91
recall = 0.89
```
As expected, MLflow records them too.

3) Artifacts 
These are the files produced by our experiments, such as:
```
model.pkl
confusion_matrix.png
training_curve.png
predictions.csv
```

MLflow will store them along the run.

4) Model
Well, yeah, eventually we don't care just about accuracy, we will actually reuse one of our trained models. That passed through this:
```
training
    ↓
model
    ↓
save it
    ↓
use it later
```

MLflow provides the tools we need to  save the models.

But now I will reveal a term that we will constantly use — "run"
What is a run in MLflow?
In MLflow, a run is simply one completed training attemp, such as:
```
Experiment
│
├── Run 1
│   ├── learning_rate = 0.001
│   ├── epochs = 10
│   └── accuracy = 91%
│
├── Run 2
│   ├── learning_rate = 0.01
│   ├── epochs = 10
│   └── accuracy = 93%
│
└── Run 3
    ├── learning_rate = 0.005
    ├── epochs = 20
    └── accuracy = 94%

And an experiment is the hierarchy of those runs (Think about it as the main folder that keeps inside all the runs).

So we need MLflow, because ML engineering is not about getting the perfect model in one run, ir will be mostly like:

Train
 ↓
bad
 ↓
change learning rate
 ↓
Train
 ↓
better
 ↓
change architecture
 ↓
Train
 ↓
better
 ↓
change batch size
 ↓
Train
 ↓
...
```

This why we use MLflow.
Now we continue with how to code MLflow.

## 2. MLflow Run

The first thing we will do is to install MLFlow, because without it you can learn the theory and not the practice (not too cute).

We will do:
```
pip install mlflow
```

That it. Now we can write our first MLflow code.

Our baseline.yaml:
```yaml
model:
  max_depth: 5

training:
  learning_rate: 0.01
  epochs: 20

data:
  test_size: 0.2
```

```python
import mlflow
import mlflow.sklearn
from config import load_config
from model import train_model
from evaluate import evaluate_model

def main():
    config = load_config("configs/baseline.yaml")

mlflow.set_experiment("just-a-try")

    with mlflow.start_run():
        model = train_model(config) # I got lazy here, so imagine we have the XGBoost code here.

        metrics = evaluate_model(model, config)

        mlflow.log_params({
            "max_depth": config["model"]["max_depth"],
            "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"],
        })

        mlflow.log_metrics(metrics) # This will send the metrics result to MLflow

        mlflow.sklearn.log_model(
            model,
            name="model"
        )


if __name__ == "__main__":
    main()
```

That some heavy jargon there, this is why I will break it piece by piece 

- `config = load_config(<file>)` - this will take our configuration file with all our learning rates, epoches, test_size, and so on... We will use it most of times than the usual `lr = 0.01` and others, because the config file can help us try hundreds of leanring rates and see which is the best.

Now that we did that on python, we would have a result as:
```python
config = {
    "model": {
        "max_depth": 5
    },
    "training": {
        "learning_rate": 0.01,
        "epochs": 20
    },
    "data": {
        "test_size": 0.2
    }
}
```

and if we write:
`config["training"]["learning_rate"]` 
This will be equal to `lr = 0.01`, we did it because it will become really useful for running experiments.

- `mlflow.set_experiment("just-a-try")` - This is basically telling to MLflow: "The run I am about to create shall belong to the "just-a-try" experiment"
We do it to keep data steady, because later we may have different types of experiments ran, and putting all in one folder is not optimal. Instead, a better option is to put the runs in a folder where they are grouped, as:
```
MLflow
  │
  ├── customer-churn
  │     ├── Run 1
  │     ├── Run 2
  │     └── Run 3
  │
  ├── image-classification
  │     ├── Run 1
  │     └── Run 2
  │
  └── fraud-detection
        ├── Run 1
        └── Run 2
```
- `with mlflow.start_run():`
this is the start of all of our everything. (We write it alongside `with` so we will not have to write:
```python
mlflow.start_run()

# stuff...

mlflow.end_run()
```

- `evaluate_model(model, config)`:
  This will take our model and evaluate it, then return us something as:
```python
metrics = {
    "accuracy": 0.94,
    "precision": 0.91,
    "recall": 0.89
}
```

- `mlflow.log_params({ ... })`:
 this will try the parameters we have in the config file as:
 ```
 Parameters
────────────────
max_depth       5
learning_rate   0.01
epochs          20
 ```

That was the whole part you needed about MLflow for now. Now we go to Dockers (Short topic)

## 3. Docker basics
Okay, so here we are with the so famouse dockers. Sadly they solve a lie that worked for almost a century — the infamous "It works on my machine". 

But how??

Imagine that yout project runs perfectly on your machine, that have this stats:
```
Python 3.11
scikit-learn 1.x
MLflow 3.x
numpy ...
```

and let us say that someone wants to try your code on their machine, and their machine have this stats:
```
Python 3.9
different package versions
different OS
missing dependencies
```

Now your code may crash on their machine and work on yours - and suddenly you have the perfect WOMM (works on my machine) theology excuse.

But here is the buz killer, the boring dockers that saves the situation.

Before I explain how, I will tell you what is an image (not a picture) and Container.

Image:

And image is a packaged template that basically containe your application and its environment.

Think about it as an os. You choose whichever image (os) you need to run the code (if your code is on python, you would use a python image — think about it as downloading python on your computer.)

Why is it useful? because it literally says: "Here is everything that you need to make the code work not only on my machine, but even on your"


Now every of this image can run on a container (an instance — like a Virtual machine (a machine in your machine — so you don't have to buy a new machine just to satisfy requirements.)). 

Now we will show how one look.

## 4. Dockerfile 

Now this is the blueprint. It contains all the information about your environment that make your code run.

For example, a docker file may look like:
```
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

We can read it almost like english, something as:

```
FROM
↓
Start with Python

WORKDIR
↓
Work inside /app

COPY
↓
Put my dependency file inside

RUN
↓
Install dependencies

COPY
↓
Put my project inside

CMD
↓
When the container starts, run train.py
```

Now let's break each of the components part by part.

1. `FROM` - It actually says something as "Start my image from an already existing image that already has Python 3.11". So instead of installing all manually - installing Linux + Python. We will take an already existing one. 
But why did we wrote: `python:3.11-slim?` Because this is a smaller Debian based Python image that has less unnecessary packages unlike the pure `python:3.11`

2. `WORKDIR` - That is the directory where the file will work. For example, `WORKDIR /app`, now everything will work inside "/app". Two things will be done automatically: `mkdir <name_of_the_workfile>` and `cd <name_of_the_workfile>`

3. `COPY` - This will copy a file into the working directory. We can use two types of operation here: 
    1. `COPY . .` - Suppose we entered a folder with our project. Now we will make our dockerfile and after using FROM and WORKDIR we will use `COPY . .`, this will automatically copy all the files/folders from our folder (the folder we are currently sitting in with the project). 
 4. `COPY <File_or_Folder_name> .`, this will copy a specific file or folder from the file or folder we need.

5. `RUN` - When we write `RUN ["<bash_commands>"]' is just a way to download some specific parts, for example, now I will write a code and explain it to you:

```dockerfile
FROM python:3.11-slim
WORKDIR /project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
# This will install the python packages of the file 
# --no-cache-dir will skip the pip's cache, and this will help us hold the image lightweighted

RUN apt-get update && apt-get install -y \
	curl \
	git \
&& rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "train.py"] 


```

5. `CMD` — this is the first command that will always start, but why would we run it? Now I will show you a dockerfile and explain with what it does and how it helps us.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt
RUN p install --no-cache-dir -r requiremrnts
COPY . .

CMD ["python", "app.py"]
```

Now let us look at two scenarios:
- With `CMD`: we can write this to start our dockerfile - `docker run my-python-app` and so, the image will immediately understand that we want to execute our image `python app.py`
- Without `CMD`: we are forced to write full form every time: `docker run my-python-app python app.py`

It is even easier to debug, because we can enter the shell on the container and debug the file just by running `docker run -it my-python-app shell`. Just like this we will have an interactive Linux shell inside the container to inspect files.

And you will have an automatic orchestration (Cloud stuffy that we will learn later)

Now we do another idea:
`docker build` — You are forced to write it after the dockerfile (only if you want to replicate it as an image). The `docker run` will create the blueprint by reading the whole dockerfile and executing it. It will package all the dependencies, file system, and so on in an image.

```bash
docker build -t my-python-app:1.0 .
```

`-t my-python-app:1.0` - the `-t` will let you give the image a name and a version as for:
```
-t my-python-app:1.0
   │             │
   │             └── tag/version
   └──────────────── image name
```
In our case:
- `my-python-app` → the name of the image
- `1:0` → this is the tag (Usually we just write the version here)

And in the end, we can start it with `run`, we just write:
```
docker run my-ml-app
```

`docker run` - this will simply create the container from the image and start it. 
The code we will use for this is:
```bash
docker run -d -p 8000:8000 --name web-service my-python-app:1.0
```

- `-d` - this means "detached" (I am pretty much sure it didn't reveal anything to us). But actually, by running the docker run without `-d` will occupy the whole terminal (As when we make a server on the local machine, we will not be able to use our terminal), but by using the `docker run -d` the process will start in the background, without occupying our terminal. 
- `--name web-service` - this will give the docker a friendly, human-readable name, so we can write:
```bash
docker stop web-service
docker start web-service
docker logs web-service
```

instead of docker give it a random nickname that is made of two parts:
1. An adjective - `hungry, smart, little...`
2. A famous scientist, hacker, or engineer - `wozniak, lovelace...`

And so on... (Anyway, the combination of `boring wozniak` will never appear, because 'wozniak' is not boring.)

- `-p <HOST_PORT>:<CONTAINER_PORT>` - This basically says: "When somebody connects on port "HOST_PORT", redirect them to port "CONTAINER_PORT" inside the container.

The idea looks like this:
```bash
-p 8000:8000
     │    │
     │    └── port inside the container
     └─────── port on your computer
```

And like this, if the python web service is listening on the port:
```shell
0.0.0.0:8000
```

We can access the container from our computer by simply writing:
```shell
http://localhost:8000
```

Why do we use it? We use it because the container has its own network namespace and if we don't write the port, the application may run but we will not be able to access it from our `localhost:8000`. 
And we use it because we may have more containers, and you can't place to all port 8000, you will write:
```bash
docker run -p 8000:8000 app1
docker run -p 8001:8000 app2
docker run -p 8002:8000 app3
```

So the whole docker run anatomy is this:
```bash
docker run
    │
    ├── -d
    │     Run in detached/background mode
    │
    ├── -p 8000:8000
    │     HOST 8000 ──────► CONTAINER 8000
    │
    ├── --name web-service
    │     Give container the name "web-service"
    │
    └── my-python-app:1.0
          Use this Docker image
```


Now I will make a whole docker file and show how we make it:
```dockerfile
FROM julia:1.0

WORKDIR /app

COPY Project.toml Manifest.toml ./ 
# The ./ stand for: "Copy this files inside the current docker image directory"

RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# This will instantiate and precompile dependecies in the building phase

COPY . .

CMD ["julia", "--project=.", "src/main.jl"]
```

```shell
docker build -t julia-app
docker run -d -p 8000:8000 --name julia julia_app
```

That it, now we can continue with LLM and RAG... since they are a long topic.


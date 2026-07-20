
Here we are with graphs! Seems like beating dragons was not enough... manually writing the linear regression, the logistic regression, and so on. Because we will learn graphs, and now we will be able to do an [evidence board ](https://en.wikipedia.org/wiki/Evidence_board)!
We will prepare you to find the murders and t- 
Joking, you are not somebody that will do something so interesting, you will rather stare at a screen and think... why is my neurons (Actually nodes!) are not connecting (Edges!). 

Now I'll teach you some mega basics stuff, because if I start from the GraphSAINT from day one your brain will fry, and so will mine. So we start with this basic idea. 

First thing we have to know are Nodes and Edges!

- Nodes - they are the dots on the graph; they can be a person, an atom, a contact, a dog, a neuron, that it.
- Edges - they are the connections of this nodes, for example: 
  Let us say that the nodes are the contacts, if the nodes are the contacts, then the edges are the relationship between them, if it is an atom, the edge will be the atomic bond, and so on...
- Example:
  Now image that Contact G is a big famous burger-man, he is rich and give free burgers... but you (Contact A) got in a argument with all of the Contacts! (Nodes)... so now you can forgive one of them... who will you choose? 
  Contact A (You) is friend with Contact B, E, and C, Contact B is friends with Contact A, E, and C, Contact C Is a friend with Contact B, A, and E, Contact D is friend with Contact E and C, While Contact E is friend with Contact A, B, C, D, and F, Contact F is friend with Contact E and G, and Contact G is friend with Contact F.

  You choose to forgive Contact E. Because he is the only pass to contact G.

![[A.png|357]]

Now we will learn about our lovely Zachary Karate Class. Yeah, we will talk about a karate classes.

Okay, now imagine that the administrator of the karate club (Mr.Hi) argued with a important person (Jhon A) over the karate club fees. After arguing, half of the students went to Mr.Hi and half to Jhon A. And data scientists were able to mathematically predict exactly which side every single member would choose with 97% accuracy (They missed just a guy - probably didn't choose none of them and did Senppuku).

Now we will actually use it as a playground and learning.

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

Now we know how to check the amount of our nodes, and edges. We can print even each member we want (each node)

But that are still baby steps, now we will learn more baby steps, but they are useful, they are worth spending time on them. So it will get funnier by the time we keep learning (I don't promise, yet I wish...)

We will learn how to add nodes, edges, and check with tho you have links. (StoryMode)

```Storymode

While looking around... as a homeless Shrimp, he found a poster, which cited:

"Mr. Hi and Jhon A argued over some fees... and I heard that Mr. Hi will give the moneys to... who will join him" - Wrote by Mr. Hi

Shrimp was mesmerized... He saw free moneys (not the scam), so he immediately applied to this scam, by using the next code:
```
```Python
import networkx as nx

G = nx.karate_club_graph()

G.add_node("Shrimp", club="Mr. Hi")
# You usually join by using: .add_node("<name>", club="<the_club>")
```
```Storymode
After going there, Shrimp was mesmerized, since there were 17 students... yet he envied a lot and felt a lot of frustration, since his chances of getting the money was getting lower... (since he was the 18th, that made his chances of getting the moneys of 1/18 = 5.5%). He hated it, yet he had to win those moneys! So he can stop learning and keep going. Yet he accidentally bumped and fell... he bumped in a beautiful, petite, slim, and cute young woman, her name was 

'Airi Sezaki'. 

He was so mesmerized by her that he immediately stood up, clear his throat and start apologizing. She accpeted his apology, then they talked a tad more. That was it... Shrimp just made a friend (He had 2, since one was Mr. Hi, because Mr. Hi accepts all the friendship of his clan members... then with Airi). He immediatelly added her by using:
```
```python
G.add_edge("Shrimp", 9) # He added firstly Airi in his friends, which was the 9th student on the list (This creates an edge from Node ("Shrimp") to Node(9))
G.add_edge("Shrimp", 0) # Then he added Mr. Hi as friends.

print(f"New Total Members: {G.number_of_nodes()}") 
print(f"New Total Friendships: {G.number_of_edges()}")

"""
Output:

New Total Members: 35
New Total Friendships: 80

"""
```
```Storymode

Then shrimp checked if he is friends with Airi and less immportantly with Mr. Hi 
```
```python
print(list(G.neighbors("Shrimp")))
# We use list so we transform the output in a list, since it returns an iterator.
# Output: [9, 0]
```
```Storymode
Now Shrimp will continue his journey... with the goal to get the moenys or to be loved by Airi, but since he was a greedy shrimp, he wanted both.
```

Since we discovered how to do the absolute basic, we can continue with the next commands, which are really useful.

Since shrimp just joined, we can admit that nobody needs him, even if he leaves tomorrow - nobody will care. That why we wants to check it, so he wanted to see his stats:

```python

# Calculate the power dynamics now that Shrimp has altered the network
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# Let's check Shrimp's specific scores
print(f"Shrimp's Social Butterfly Score (Degree):{degree_centrality['Shrimp']:.4f}")
print(f"Shrimp's Gatekeeper Score (Betweenness): {betweenness['Shrimp']:.4f}")
print(f"Shrimp's Influencer Score (PageRank): {pagerank['Shrimp']:.4f}")

"""
Output:

Shrimp's Social Butterfly Score (Degree): 0.0588
Shrimp's Gatekeeper Score (Betweenness):  0.0080
Shrimp's Influencer Score (PageRank):     0.0089
"""
```

Now let us break each step into small steps.

- `nx.degree_centrality(G)` - This will help us identify how much of a social butterfly everybody is, and since shrimp just entered and he has just 2 friends out of all the 35 possible friends (Nodes), his social butterfly score will be low to the ground as exactly 0.0588, that means that he almost has no friends... while Mr. Hi would have 0.5000 (max), since everybody is his friend.
- `nx.betweeness_centrality(G)` - This is Shrimp's lowest stat... because this means how important shrimp is... Since everybody can talk directly to Airi and Mr. Hi or talk to them by using a shorter path, he has no importance, because everybody can do everything without him, that why his score is of 0.0080. Mr. Hi is the main role... since he stays in the middle... every student who wants to talk to the other from the other side, has to pass through him, so he is the main guy in all cases. 
- `nx.pagerank` - Here our shrimp can cook, because he is friend with Mr. Hi... and who is Mr. Hi? A really important person, the more important persons you are friend with, the more the score will raise. So we can understand Shrimp gets a massive algorithmic boost just by standing next to the boss. (Additionally, Shrimp has Airi next to him, so since she is the most beautiful - big buff)

Now we will make a code, and explain how it work:

```python

# Let us say that we want to find who has the biggest degree_centrality score from all the clan, so what we can do? 

# We can write a code from python, which will let us do it. 

degree_dict = nx.degree_centrality(G) 

#Think the output will be like this: 
# degree_dict = { 
# 0: 0.4706, <-- Mr. Hi (Connected to almost half the club) 
# 33: 0.5000, <-- John A (The President, heavily connected) 
# 9: 0.0888, Airi (A popular student that is pretty smart...) 
# "Shrimp": 0.0588 Shrimp (Just joined, knows only Mr. Hi & Airi) 
# } 

sorted_degree = sorted(degree_dict.items(), key=lambda item: items[1], reverse=True) 
top_node, top_score = sorted_degree[0]
print(f"The most connected member is Node {top_node} with a score of {top_score:.4f}") 

# Now we will break it to pieces, slowly and steadily. 

#sorted() -> This will sort the data, for example: If this is an integer, we will go from smallest to biggest. 

# .items() -> That cool, because it will literally make from the dictionary we have, list of tuples -> 
# Degree_dict.items() = [ 
# (0, 0.4706), 
# (33, 0.5000), 
# (9, 0.0888), 
# ("Shrimp", 0.0588) 
# ] 

# Now each item is like. (node, score_of_the_node) 

# But sorted() by what? node? score_of_the_node?? Python has no idea, that why key=lambda item: item[1] helps us. 

# Why? because lambda calls out values "item", and it will look which we want (we count from 0, so 0 is technically the node), but since we selected 1, it will give us the score_of_the_node, so it will take the score of the node and place them as: (0.0588, 0., 0.0888, 0.4706, 0.5000). 

# But since we want from biggest to smallest, not smallest to biggest, we will use reverse=True, which will make our result look like: (0.5000, 0.4706, 0.0888, 0.0588) 
#Now our output looks like: 

""" 
Output: 

[ 
(33, 0.5000), 
(0, 0.4706), 
(9, 0.0888), 
("Shrimp", 0.0588) 
] 

""" 
# Then the last step is the eaisest. 
top_node, top_score = sorted_degree[0] 
# (We know that we count from 0), so the 0 is out first place: ("33", 0.5000)) 

# And now we have: 
# top_node = "33" 
# top_score = 0.5000 
print(f"The most connected member is Node {top_node} with a score of {top_score:.4f}") 

# this will print: 
# The most connected member is Node 33 with a score of 0.5000
```

from this... Shrimp realizes that while he is safe standing next to Mr. Hi, the enemy faction holds a massive structural advantage in terms of raw popularity. If a war breaks out, Mr. Hi's money might not be enough to buy everyone over...

Pretty dramatic, but that normal, we will proceed this way till the GraphSAGE and more!

But I guess I'll continue more, so we can actually understand a lot better everything! 

Before going in more advanced topics, I'll rather explain the types of graphs that exist, since this will help us understand what strategy to use when we see a type of graph:
1) **Undirected vs. Directed**
2) **Unweighted vs. Weighted**
3) **Cyclic vs. Acyclic (DAG)**

Firstly we will start with Directed and Undirected graphs:
 
![[image (1).png|464]]

1) Undirected Graphs (graph a)- this are the graphs that are connected by straight lines, from node to node.
   Imagine that you are the number 1 in the graph (a) - You can move to 2, the you can move to 3, 4, or go back to 1, after choosing 3, you can move to 4, 7, or back 2 and from 2 to 1 again.
 
2) Directed Graphs (graph b)- this are the graphs connected by arrows, they explicitly tell you where you can move, and if we go from the 1st node to the 2nd, now we can see that we have no way back, just node 3 or node 4. If we go to node 3, we can't go back, the only way you can go back is when there is a separated arrow that points backwards too.  


Now we will speak about Unweighted vs weighted graphs:

![[image (2).png|499]]

1) Unweighted Graphs (second graph) - Every edge is identical. There are no numbers attached to the lines (edges). The "distance" between two nodes is purely measured by the number of hops (edges) it takes to get there.

2) Weighted Graphs (first graph) - Every edge has a number (a weight) stamped onto it. This weight represents a cost, distance, or time. So we may clearly understand that a better way to get to C would be to go from A to B and then to C, because let us think about this:
   The weight (the number on the graph) means kilometers, now we can clearly understand that doing 10 kilometers is not worth it, when there is a shortcut by going to B and then to C (4 +1 = 5, that means we just saved 5 kilometers.)


And in the end we will learn Cyclic and Acyclic graphs:

![[Pasted image 20260705084016.png|479]]


1) Cyclic Graphs (second graph) - A graph is cyclic if it contains at least one loop. This means you can start at a specific node, travel along a path of edges and return to the star. We can see from the image that we start from node A, then we are forced to go to the node C, after this we can choose, to go to node B or D. Let us say we will go to node B. After that we are forced to go back to node A - we just returned to the beginning... That mean we can loop forever with this logic: A -> C -> B -> A -> C -> B -> A .... and like this, your program crashes with a `Segmentation fault` or `Stack Overflow`  - High algorithm risk (Will cause infinite loops if we don't track the visited nodes)

2) Acyclic Graphs - A graph is acyclic if it is completely impossible to form a loop. No matter which node you start at and which path you follow, you will eventually hit a dead end. They are also called "DAG (Directed Acyclic Graph)" - small algorithm risk (You will hit a dead end)


Now we will learn some really important concepts, starting from BFS and DFS:

### 1) BFS and DFS

2) BFS (Breadth first search) - think about it as the [ripples](https://dictionary.cambridge.org/dictionary/english/ripple?__cf_chl_f_tk=W..qwe3CK7ZQ14V_wFyKH3ymhxj1zqE.LavbKgF5WbQ-1783225416-1.0.1.1-GXV8OfEH1NoBwaeLK.HV3jBonFtmnSlj5CCWX16KNIY#google_vignette) of a pound when a small rock is thrown in it. The pound doesn't make a straight line that goes in a singular direction, it actually makes some circular waves that keep getting bigger at a uniform speed from all the sides.

   BFS uses a data management structure called FIFO (First In, First Out). That where the first element added to the collection is the first one to be removed. 
   
   Imagine that you added on the stove 3 pans, pan A, B, C. First you put pan B, then C, and the last the pan A. Using the FIFO, we will take away firstly the pan B, then C, then A. Why? Because it was the first to be placed, and the first to get away.


   The FIFO (First In, First Out) is a queue, what does that even means? A queue is just a simple technological waiting list that takes the data as it was sent and 

   A queue only lets you do two things:

3. **Enqueue (Join the Line):** When new data arrives, it goes to the **back of the line**.
    
4. **Dequeue (Get Served):** When the system is ready to process data, it takes the item currently at the front of the line.


   Imagine a Fast-Food Drive-Thru. 
   
   **Enqueue** - Car A goes in the front, it orders something, then Car B goes behind Car A and waits, then car C goes behind Car B. They are trapped in a single file track.

   **Dequeue** - Now that all the cars are ready, here starts the serving, it slowly serves Car A, then Car B, and in the end Car C. 

   That what a queue is.
   But imagine even this, there is a queue at the market:

   5) Enqueue:
   
   There is A, B, and C waiting in the line, and then D comes too,  where he will go? to the `rear`

   A, B, C, D

   No searching.  
   No shifting.

   Just place it at the end.
   Time = O(1) - The O(1) means **Constant time**, so even if there were 1M of people, he will take the same microsecond to solve it and place them all in the rear.

   6) Dequeue:

   A B C
   The first customer leaves, what is the result?
   It is:
   
   B C

   Move the front pointer to B.
   You don't touch the other elements.

   Time = **O(1)**  -it has a constant time, because if the one from the font moves away, the one that was sitting behind it, stays there, but now he is called "The Front", because he is in front, 


   Let's say that Shrimp wants to deliver a message to somebody from the other side, but he is in the middle of a crowded room, and can only speak to people close enough to touch his shoulder. 

   That the perfect moment for BFS:

   ![[image.png|477]]

   Shrimp will tell to the first layer the information he wants to say (From Shrimp to People)
   Then the People will tell the information to the second layer (The friends of The People)
   Then the friend of the People say the information to the third layer, and so on... until it doesn't reaches the target.
   So the situations where we use DFS is:
   
   WHEN TO USE:
   The graph must be unweighted (or all edge weights are exactly equal)
   Wants to find the shortest path:

   WHEN TO DON'T USE:
   The graph is weighted in an unequal way.
   The problem is a deep maze (Memory risks)
   
   - Problems:
   1) The Problem: How does LinkedIn know someone is a "2nd-degree connection"?

   The BFS Execution: 
   LinkedIn starts a BFS from _your_ node. Layer 1 is your direct friends (1st degree). Layer 2 is friends-of-friends (2nd degree). If the algorithm is looking for a connection path, BFS finds it using the absolute minimum number of people in the middle.

   2) The Problem: You want to reach to the supermarket by the fewest roads possible, not distance. 

   The BFS Execution: 
   Google maps starts using the shortest path on unweighted graphs, and that how he sees which roads will take less hops to get to the market, after he found the perfect way, he strips the other roads on the side and let just the path with the fewest roads.



   2) DFS (Depth First Search) - Imagine you fell into a massive hedge maze, since you want to find the exit or just map out the maze, you tried to find a strategy.

   Which strategy?

   You don't actually split yourself into clones to check all the paths (That's BFS). Instead, you pick a path and go all the way till there, and after hitting a dead end, you will return back till you don't see a detour, then immediately take every detour and keep going back and forth, till you don't find the exit. That how it works, and it is different from how is FIFO, because this uses LIFO (Last In, First Out). It uses a data management structure called "a Stack", what does it mean?

   Think of a stack of dinner plates in a kitchen:

   You wash a plate and put it down.
   You wash a second plate and place it on top of the first one.
   You wash a third plate and place it on top of the second one.

   When you need a plate to eat dinner, which one do you grab? You grab the one on the very top - the last one you just put down.

   ![[image (3) (1).png|391]]

   We look at the nodes, from them we can understand that the DFS will check the 1st node, then 3rd, then come back to the 1st and go to the 4th, after that it will go back to the first, then the 0 node, and then to the second... and so on.

   WHEN TO USE IT:
   The graph has to be explored till its most deep part
   You need to find a specific order (Topological Sort)                          <--  We will explore more later
   You need to detect hidden cycles (If the graph is cyclic or Acyclic)

   Now we will learn how to code this for real. Because we will need it a lot of times.
   Firstly we will start with the Breadth first search.

   Now we will recreate a graph on python as:

   ![[ChatGPT Image Jul 5, 2026, 05_27_03 PM.png|392]]


```python
from collections import deque
import networkx as nx
# We will manually buid the graph


custom_graph = {
   'A': ['B', 'C', 'H'],      # It is like: 'node': ['neighbords']
   'B': ['A', 'D', 'E'],
   'C': ['A', 'F'],
   'D': ['B', 'G'],
   'E': ['B', 'G', 'H'],
   'F': ['C', 'I'],
   'G': ['D', 'E', 'J'],
   'H': ['A', 'E', 'J', 'I'],
   'I': ['F', 'H', 'J' ],
   'J': ['G', 'H', 'I']
 }
 # sometimes it may happen that we will get a list, not a dictionary, and all the messages I'll write using "#" are just: "In case of list"
 # In case of list you will add here:
 # G = nx.Graph()
 # G.add_edges_from(custom_graph)
 
 
def pure_bfs(graph, start_node):
	visited = set()
	queue = deque()
	traversal_order = []
	
	queue.append(start_node)
	visited.add(start_node) 
	
	print("----STARTING THE EXTRACTION----")
	print("")
	while len(queue) > 0:
		   current = queue.popleft()
		   traversal_order.append(current)
		   print(f"Popped node from front '{current}'. Current queue: {list(queue)}")
		   # Here, instead of writing graph.get(current, []). You will write                  graph.neighbors(current)
		   for nb in graph.get(current, []):
			   if nb not in visited:
				   queue.append(nb)
				   visited.add(nb)
	return traversal_order

# Here you will write result = pure_bfs(G, 'A')
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

Now, before you hyperventilate... I'll explain the whole code in simple words, so you can understand.

- `custom_graph = ...` - This is called an adjacency list. We just recreated the graph on python, the structure is simple, the key value is 'node' and the value is the 'neighbors'

- `visited = set()` - this creates a set (imagine an empty list, just that can't contain duplicates)

- `queue = deque()` - creates a double-ended queue as (A, B, C, D) and it supports operations as 
```python

queue.append('E')      # Add to the rear (right)
queue.appendleft('Z')  # Add to the front (left)

queue.pop()            # Remove from the rear (right)
queue.popleft()        # Remove from the front (left)
``` 

- `traversal_order = []` - an empty python list

- `while len(queue) > 0:` - this will continue the loop of searching till the len(queue) is bigger than 0 (len(queue) - it is just how many nodes we have in the queue, for example for node B we have in the queue )

-  `current = queue.popleft()` - this operation has two important steps. 
   1. It removes from the queue the leftmost value, if we have something like queue = B, C, H
   popleft will remove away B and let a queue as C, H.
   2. It saves the removed value, as for example: `current = 'B'`, because it got removed, so will C and the next ones... 

- `queue.append(start_node)` - We literally just throw our starting node (in our case, `'A'`, since we wrote `pure_bfs(custom_graph, 'A')`) into the queue first. Because a queue without at least one node inside is just... an empty deque staring at you judgmentally, doing nothing.

- `visited.add(start_node)` - We also mark `'A'` as visited immediately, the moment it enters the queue - not when it gets popped. This is important, and I mean IMPORTANT, because if you forget this step, you'll add the same node to the queue multiple times by accident (imagine 5 different neighbors all pointing back to `'A'`, and all of them going "hey is A visited? no? ok let me add it again" - Cutely loops it again with a smile).

- `for nb in graph(current, []):` - Now this is where the actual magic (or crime, depending on your mood) happens. We take the `current` node we just popped, look inside the dictionary `graph[current]`, and get the list of its neighbors. For example, if `current` is `'A'`, then `graph['A']` gives us `['B', 'C', 'H']`, and we loop through each one like a nosy neighbor checking who's home (I used .get so when we maybe will forget smth it will give back and empty list, not an error).

- `if nb not in visited:` - Before blindly shoving a neighbor into the queue, we ask "wait, haven't I met you already?" If the neighbor is already in `visited`, we ignore it completely, and move to the next one. Otherwise we will suffer at 2 AM, wondering why our honey of BFS never finishes.

- `queue.append(nb)` - If the neighbor passed the "have I met you" test, we throw it to the back of the queue, patiently waiting its turn to get popped later.

- `visited.add(nb)` - And we immediately mark it as visited right there, on the spot, not later. Same reasoning as before - mark it visited the second it touches the queue, not the second it gets processed. Trust me, future you will thank present you.

- `return traversal_order` - After the queue finally becomes empty (meaning literally everyone reachable from `'A'` has been visited and processed), we just return the list showing the order we visited everyone in.

this part is a tad messy, so we write again the mess:

```python

custom_graph = {
   'A': ['B', 'C', 'H'],      # It is like: 'node': ['neighbords']
   'B': ['A', 'D', 'E'],
   'C': ['A', 'F'],
   'D': ['B', 'G'],
   'E': ['B', 'G', 'H'],
   'F': ['C', 'I'],
   'G': ['D', 'E', 'J'],
   'H': ['A', 'E', 'J', 'I'],
   'I': ['F', 'H', 'J' ],
   'J': ['G', 'H', 'I']
 }
```
```teaching
Now i will explain how the queue will go... 
We start from A and we have as queue [B, C, H], (A will enter the visited=set())
- That means that we will have visited = set(A)
Since BFS uses FIFO, it will send the new neighbors to the rear and take from the front
Righ now we are on A and have as queue [B, C, H]
So it will go firstly to B, now we have as neighbors [A, D, E] - current queue = [C, H]
But since we already visited A, A is ignored because it has already been visited - new neighbors [D, E] they will be added to the rear - queue = [C, H, D, E], now we have in visited = set(A, B)
Next we will go to is C, so now our neighbors are [A, F], we already visited A, so the only new is F, and it will go in the queue [H, D, E, F], then we will visit H, its neighbors are [A, E, J, T], the visit list is (A, B, C, H), A and E are eliminated, since one is visited already and the other is in the queue, new queue = [D, E, F, J, T].... And so on, we understood how we got that output

```
```python
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

Now that we finally did all we could with this bloody mess, we will suffer again, but with deciphering DFS, because after a cute storytime with Airi Sezaki and a lot of pain, we reached to the same point! Because the pain is a Cyclic graph of just A, B, C with no end.

Jokesssss, because the DFS code is identical to the BFS, just some small tweaks. Now we will explain to you the code and how it will print thr Traversal Order:

```python
custom_graph = {
   'A': ['B', 'C', 'H'],
   'B': ['A', 'D', 'E'],
   'C': ['A', 'F'],
   'D': ['B', 'G'],
   'E': ['B', 'G', 'H'],
   'F': ['C', 'I'],
   'G': ['D', 'E', 'J'],
   'H': ['A', 'E', 'J', 'I'],
   'I': ['F', 'H', 'J' ],
   'J': ['G', 'H', 'I']
 }
 
 # In case of a list we follow the same idea as the BFS in changes.
 
def dfs(graph, start_node):
	visited = set()
	stack = []
	traversal_order = []
	
	stack.append(start_node)
	visited.add(start_node)
	
	print("----STARTING THE EXTRACTION----")
	print("")
	while len(stack) > 0:
		current = stack.pop()
		traversal_order.append(current)
		
		 print(f"Popped node from front '{current}'. Current stack:       {list(stack)}")
		 
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

Popped node from front 'A'. Current stack: []
Popped node from front 'H'. Current stack: ['B', 'C']
Popped node from front 'I'. Current stack: ['B', 'C', 'E', 'J']
Popped node from front 'F'. Current stack: ['B', 'C', 'E', 'J']
Popped node from front 'J'. Current stack: ['B', 'C', 'E']
Popped node from front 'G'. Current stack: ['B', 'C', 'E']
Popped node from front 'D'. Current stack: ['B', 'C', 'E']
Popped node from front 'E'. Current stack: ['B', 'C']
Popped node from front 'C'. Current stack: ['B']
Popped node from front 'B'. Current stack: []

Final DFS Traversal Order: ['A', 'H', 'I', 'F', 'J', 'G', 'D', 'E', 'C', 'B']
"""
```

final idea of how the queue work and how it moves:

```python
custom_graph = {
   'A': ['B', 'C', 'H'],
   'B': ['A', 'D', 'E'],
   'C': ['A', 'F'],
   'D': ['B', 'G'],
   'E': ['B', 'G', 'H'],
   'F': ['C', 'I'],
   'G': ['D', 'E', 'J'],
   'H': ['A', 'E', 'J', 'I'],
   'I': ['F', 'H', 'J' ],
   'J': ['G', 'H', 'I']
 }
```
```teaching
Okay, so now we start with how it visits the nodes. 

As usual, we start from 'A'. The stack is [B, C H], but since we use stack, and they are placed as plates one above each other, we will take H.

Now H's neighbors are: [A, E, J, I], since we already visited A, A is ignored because it has already been visited and we add just E, J, I to the stack = [B, C, E, J , I]. Since it stacks plates, the plate on top is I.

So we visit I. The neighbors of I is [F, H, J], since H is visited and J is in the stack, we add just F: [B, C, E, J, F], since F is on top, we visit F. 

F's neighbors are [C, I], but we already have both, on in the stack the other in visited; current stack is [B, C, E, J] so we continue....

J neighbors are [G, H, I], since H and I are already visited, we will add just G to the stack, [B, C, E, G] (the visited: {'A', 'B', 'C', 'E', 'F', 'H', 'I', 'J'} - every element in the stack is immediatelly added to the visited, the microsecond it enters there.)

Next is G and so on...
```
```python
"""
----STARTING THE EXTRACTION----

Popped node from front 'A'. Current stack: []
Popped node from front 'H'. Current stack: ['B', 'C']
Popped node from front 'I'. Current stack: ['B', 'C', 'E', 'J']
Popped node from front 'F'. Current stack: ['B', 'C', 'E', 'J']
Popped node from front 'J'. Current stack: ['B', 'C', 'E']
Popped node from front 'G'. Current stack: ['B', 'C', 'E']
Popped node from front 'D'. Current stack: ['B', 'C', 'E']
Popped node from front 'E'. Current stack: ['B', 'C']
Popped node from front 'C'. Current stack: ['B']
Popped node from front 'B'. Current stack: []

Final DFS Traversal Order: ['A', 'H', 'I', 'F', 'J', 'G', 'D', 'E', 'C', 'B']
"""
```


After this heartily and cute topics, we will move to the final bosses of this month...

- **Dijkstra**
- **Degree Centrality (easy)**
- **Connected Components**
- **Cycle Detection**
- **Topological Sort**
- **PageRank**
- **Betweenness Centrality**

We will learn them step by step, since we need them really much. Ah, forgot to say, prepare the math, sweetheart, because we will be crunching numbers that you have to be a mathematician. So I will say in a sweet way, no worries, we will tackle them down together and explain what each mean, even if it will be baby steps.

Let us start from the Dijkstra:

### 2. Dijkstra

What we need Dijkstra for?

I'll show you an Undirected, weighted, cyclic graph, and explain how will Dijkstra help us with?
We will use Dijkstra to find the shortest path (the path with the minimum total edge weight) from a single starting node.
When we will use them? We will use Dijkstra when we have a weighted graph with this rule:
weight >= 0. Because if the value of a weight is negative, all the plan is ruined. The graph can be directed or undirected, cyclic or acyclic. The important requirement is that all edge weights are non-negative.

Firstly I'll explain how it works and how it finds the optimal path that is the most budget friendly.

![[ChatGPT Image Jul 6, 2026, 12_18_46 PM.png|674]]

We can make a list firstly:

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & \infty & \infty & \infty & \infty & \infty & \infty & \infty & \infty \\ \hline \text{Parent} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} \\ \hline \end{array}$$

Don't be scared, now we will start counting.

Since we start from A, we can move just to B or H... let's see how much they cost.
To go from A to B the cost is 4 (0 + 4), from A to H is 8 (0 + 8), so we update the current list.

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & \mathbf{4} & \infty & \infty & \infty & \infty & \infty & \mathbf{8} & \infty \\ \hline \text{Parent} & \text{None} & \mathbf{A} & \text{None} & \text{None} & \text{None} & \text{None} & \text{None} & \mathbf{A} & \text{None} \\ \hline \end{array}$$

The parent stores the previous node on the currently best-known path. So now we know that the best way to get to B is by A, since it costs just 4. 
Now we have to choose the smallest, the smallest from 4 and 8 is clearly 12... okay, it's 4 - which is B. That why we look where B can go.
B can go to H and C.
If we go to H the output is 15 (4 + 11), we check if H is empty... no, it has 8 already, but is it a smaller number than B? 15 < 8, nope. We ignore it because we already know a cheaper route to H.
If we go to C the output is 12 (4 + 8), Does C already have a value? No, that mean we add the 12 to it, and the B as parent, since we came from there

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & 4 & \mathbf{12} & \infty & \infty & \infty & \infty & 8 & \infty \\ \hline \text{Parent} & \text{None} & A & \mathbf{B} & \text{None} & \text{None} & \text{None} & \text{None} & A & \text{None} \\ \hline \end{array}$$

Okay, now we have 8 (from H, since we didn't visit it yet) and 12, who is smaller (choose the unvisited node with the smallest tentative distance... Not just the smallest value)? 8. That why we look where can H go, it can go toward B, I, G...
If we go to B the outcome is 19 (8 + 11), is it smaller than the current value of B? No. We keep going without updating it.
If we go to I the outcome is 15 (8 + 7), since I is empty - we add it.
If we go to G the outcome is 9 (8 + 1), sine G is empty - we add it

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & 4 & 12 & \infty & \infty & \infty & \mathbf{9} & 8 & \mathbf{15} \\ \hline \text{Parent} & \text{None} & A & B & \text{None} & \text{None} & \text{None} & \mathbf{H} & A & \mathbf{H} \\ \hline \end{array}$$

We have 12, 9, 15 as values... to which we will go? 9 - which is G.

G can go to I, F, we check both:
If G goes to I the output is 15 (9 + 6) -> That equal to its current value, so we don't update it.
If F goes to F the output is  11 (9 + 2) -> F is empty, so we add it to the list.

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & 4 & 12 & \infty & \infty & \mathbf{11} & 9 & 8 & 15 \\ \hline \text{Parent} & \text{None} & A & B & \text{None} & \text{None} & \mathbf{G} & H & A & H \\ \hline \end{array}$$

Now we have 12, 11, 15. We will choose 11 - which is F.

F can go to to C, D, E.
If F goes to C the output is 16 (11 + 5), is the current value smaller than the old value? No.
If F goes to D the output is 25 (11 + 14), we add it since it is empty.
if F goes to E the output is 21 (11 + 10), we add it since it is empty.

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & 4 & 12 & \mathbf{25} & \mathbf{21} & 11 & 9 & 8 & 15 \\ \hline \text{Parent} & \text{None} & A & B & \mathbf{F} & \mathbf{F} & G & H & A & H \\ \hline \end{array}$$

Now we have 12, 15, 25, 21. We choose 12 - which is C.

C can go to I, F, D.
If C go to I the output is 14 (12 + 2), which is a smaller value than the current one, so we change it from the list and its parent too
If C go to F the output is 16, which is bigger than the old value. No change
If C go to D the output is 19, which is smaller than the old value. We update it.

$$\begin{array}{|c|c|c|c|c|c|c|c|c|c|} \hline \text{Node} & \mathbf{A} & \mathbf{B} & \mathbf{C} & \mathbf{D} & \mathbf{E} & \mathbf{F} & \mathbf{G} & \mathbf{H} & \mathbf{I} \\ \hline \text{Distance} & 0 & 4 & 12 & \mathbf{19} & 21 & 11 & 9 & 8 & \mathbf{14} \\ \hline \text{Parent} & \text{None} & A & B & \mathbf{C} & F & G & H & A & \mathbf{C} \\ \hline \end{array}$$

From here, the algorithm continues exactly the same way until every node has been processed. Now I explain how to find the perfect path:

We have the best paths for all the nodes we want to reach.

Let's try:
If we want node E?
We will watch the parent... which is F (F -> E), we look at F's parent... it is G (G -> F -> E), G's parent is H, so the path is (H -> G -> F -> E), H's parent is A... so the final path with the lowest cost is: 
A -> H -> G -> F -> E

Want node D? The best path that cost the least is:
The parent of D is C... (C -> D), the parent of C is B... (B -> C -> D)... the parent of B is A... so in the end we get:
A -> B -> C -> D

now we will use a tad of python for the code:

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

Now i'll explain a tad:

- `G = nx.Graph()` - this create a empty graph, think of it like a empty notebook page, nothing is write on it yet, no nodes, no edges, just blank sadness waiting to be filled

- `G.add_weighted_edges_from(edges)` - this the part where we finally dump all them tuples inside our empty G

- `nx.single_source_dijkstra(G, source='A', weight='weight')` - and here the real big deal, the thing we did by hand with the whole table and the infinity symbols and the parent columns... this one line just do ALL of that, instantly. `source='A'` mean we just tell it "start counting from A, like we did before". `weight='weight'` just tell it "hey look at the number we stored under weight when you calculate the cost, dont just count how many edges, actually look at the cost of them". And this function is nice because it return not one, but TWO things at once - distances (a dictionary, telling you the cheapest cost to reach every node from A) and paths (another dictionary, telling you the exact route, as a list of nodes, you gotta walk to get there cheapest... THE ONE WE ALREADY DID BY HAND)


WHOA, what a pain to write about this stuff by handdddd, but whatever. Anyway, we will be really professional, so no complains.
Top 100 reasons I hate to....
Okay, maybe I'll complain later, but now we start with the Degree Centrality, because we have to try hard enough, so the cuties on janitor.ai will love you for being so smart (I know, it is sad, but that it, you can't always talk to real people - they are scary, especially by choosing CS. But maybe that just me).

### 3. Degree Centrality

What is the Degree Centrality?
The Degree Centrality is the simplest way to see how important is a node in the network, because it checks its degree (How many connections it has with other nodes.)

We already explored this topic in the past, but today we will count by using the graph that we just had. And beside this, we will use its mathematical formula.

Firstly we have to understand... the logic behind it.
for the math formula:

$$\text{Degree Centrality}(v) = \frac{\text{Degree}(v)}{N - 1}$$

We need the Degree and the Maximum Possible Connections (N - 1).

I'll show an image and we will solve the Degree Centrality by the image, and tell which is which:

![[image (6).png|494]]

Now let's see which will be the degree and maximum possible connections of A...
What is the Degree of A? it's 1, because it connects just to B.

What is the Maximum Possible Connections (N - 1). It is 7, because in the image we have just 8 nodes and N = 8 -> 8 - 1 = 7 (A node can't connect to itself).
So the Degree Centrality(v) = 1/7
which is approximately just 0.143 (The degree centrality of A)
What we understand from it?
We understand how useless is actually A, because if it disappears... what changes? nothing, he is just a dot on a side.

What about E? 
E has 4 connections... since 7 are the max outcome... Degree Centrality = 0.571...
That means that E is really important, because if a signal is transmited from it, it reaches 4 nodes out of 7.

We already know the python code, it is 

```python
import networkx as nx

dc = nx.degree_centrality(G)

# I didn't create a graph, but we just say the code and that it, because if you want to see the result you can print just:

for node in sorted(G.nodes()): 
	print(f"{node} | {G.degree(node)} | {G.degree(node)} / 7 = {dc[node]}")
	# Instead of / 7 in other codes we will just calculate all the nodes and           write the ammount of them
```

We keep moving, because the topic was too easy...

### 4. Connected Components 

The connected Components are just a group of nodes that is connected between them, but totally cut of from the rest of the groups with nodes, as we can observe from the image

![[ChatGPT Image Jul 6, 2026, 03_09_36 PM.png|470]]

is the graph connected? False
In how many components is it broken? 2.

Yeah, this is the whole point of the idea.

we will look at the few operations that explain if the graph is connected and how many components it has.

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
# It will check if the graph is connected, and answer with a boolean logic the result (True/False) (In our case that a False, because A, B, C is all by itself.)

print(f"Number of components: {nx.number_connected_components(G)}")
# This will watch how many components we have (Groups of grpahs, if the graph is connected the result is always 1, while if the graph is not connected like in out case (Our graph is composed out of 2 groups), the result will be 2 (just for our graph))


components = list(nx.connected_components(G))
for component in components:
    print(f"Component: {component}")
```

The hardest logic here is to memorize the code and the print.

- `components = list(nx.connected_components(G))` - This part checks all the components and note their content as:
  First component = {'A', 'B', 'C'}
  Second component = {'D', 'E', 'F', 'G', 'H'}
  We used list because the `nx.connected_componrnts(G)` returns a generator, which will give us the components when we ask for them. Yet we use list so it stores all the components in a list

### 5. Cycle Detection

This actually is really easy - again, we remember clearly that a cycle is just a loop that can happen. For example:

If we can start from a point and then somehow somewhat get to the starting point again, that clearly a cycle
A -> B -> C -> A?... 

We look again at the picture:

![[Pasted image 20260706171002.png|475]]

We can check and see how many cycles we have in a graph with this code:

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

print(f"The total numbers of cycles found are: {len(cycles)}")
for index, cycle in enumerate(cycles, 1):
print(f"The cycles, {index}: {cycle}")

"""
Output:

The total numbers of cycles found are: 3
The cycles, 1: ['E', 'H', 'G']
The cycles, 2: ['E', 'D', 'H']
The cycles, 3: ['E', 'F', 'G']
"""

```

- `cycles = nx.cycle_basis` - This operation finds the smallest set of important cycles that can describe every other cycle in the graph. It doesn't list every existing ones, it lists enough cycles that can give us new information, because if we did:
  A -> B -> C -> A, it would be stupid to say that we will list even B -> C -> A -> B or C -> A -> B -> C.

Why didn't we add A -> B -> C?
Because this is not a cycle, because in the image we can't have a clear loop, we have to retrace our steps to make a fake loop.
That pretty much everything about this idea...
Next.

### 6. Topological sort

What is the topological sort? It is just a way to sort the graph in a linear way 
A -> B -> C -> D -> E -> F -> G and so on... 
But we use it only for Directed Acyclic Graphs. If the graph has loops, then we can't use it, or if the graph is Undirected.

For example imagine that the nodes are tasks:
A -> B
We can not start B before finishing A

or look at the graph:

![[ChatGPT Image Jul 6, 2026, 09_23_37 PM.png|491]]

Can I do G as first? No. Because G depends on several other tasks.
So now the big and famous question becomes...
"What is a valid order in which I can complete all the tasks without breaking any dependency?"
And that why we need Topological sort.

We will code it immediately and it will tell us the ways we can get there in a perfect linear way.

```python

import networkx as n

# We make a Directed Graph, so you know, writing just nx.Graph() will add an Undirected graph, while writing nx.DiGraph() will add a Directed graph

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
# The ''.join(smth) will simply print the outputs, but place after each even the thing we placed in the string, because if the result ['A', 'B', 'H', 'C', 'G', 'D', 'I', 'F', 'E'], but we places '->' in the strings this, the output will be: A -> B -> H -> C -> G -> D -> I -> F -> E

"""
Output

Is this graph a valid DAG? True

=== Valid Execution Order ===
A -> B -> H -> C -> G -> D -> I -> F -> E
"""
```
```meaning
This means:

A must happen before B
A must happen before H
B must happen before C
C must happen before D
C must happen before I
H must happen before G
H must happen before I
G must happen before F
I must happen before F
D must happen before E
F must happen before E
```

- `nx.is_directed_acyclic_graph(DAG)` - It will check if the graph we just made is a Directed Acyclic graph, if it is, it will write True, otherwise - False.

- `list(nx.topological_sort(DAG))` - It will use the topological sort to sort our messy graph.

Okay so this order `A -> B -> H -> C -> G -> D -> I -> F -> E` basically just telling you a valid sequence to do the tasks without breaking no dependency rule, straight up.

What it mean is, every single edge we wrote, like `('A', 'B')` or `('H', 'G')`, is being respected here, meaning the "before" node always show up earlier in this list than the "after" node. So A come before B (respected), A come before H (respected), B come before C (respected), C come before D and C come before I (both respected), H come before G and H come before I (respected), G come before F (respected), I come before F (respected), D come before E (respected), F come before E (respected). Every single one of them checks out, nothing is broken, nothing come before its own requirement.
### 7. PageRank

PageRank is different, because it measures a node's importance not just by counting how many links it has, but by looking at who is linking to it.

It will be really brain tiring... so if you are reading this at a late hour, just go asleep, because it is hard, really. I almost went nuts here. I wanted just to cry, so you will. But maybe I'll explain it so easy that you will master it in 5 minutes...

Before we start I wanted to say that PageRank works like this:
If there are 2 kids - a popular one (Kevin) and a smart one (Michael). 
The popular kid is recommended by 100 kids (They are meh-ish in everything).
The smart kid is recommended by 6 professors and the president of the school.
If we use the Degree centrality the popular kid wins by far, meanwhile, if we use the PageRank the smart kid wins. Because he is recommended by 7 individuals with a really high status.

We start from the formula, because I like scaring people before teaching them:
$$PR(u) = \frac{1 - d}{N} + d \sum_{v \in In(u)} \frac{PR(v)}{L(v)}$$
SCARY.
Yet let me explain it:

- **$PR(u)$**: The PageRank score of the node $u$ you are trying to calculate.
- **$N$**: The total number of nodes in the entire graph.
- **$d$**: The damping factor (usually set to $0.85$)... I'll give a full example here... Imagine that there are 12 stores, each start with 10 points, so all of the crowd who starts from store A, split in half, half goes to the Store B, and half goes to the store C. They will get many points... but then there is store G... 6 stores out of 12 have a door that lead to store G... which make it acumulate many points... while there is Store J and Store K... they don't have a popular name, you may enter there only by accident. Since they are unpopular, no other stores have doors leading to them. To make it worse, Store J only has a door to Store K, and Store K only has a door to Store J. They are an isolated loop...
  What would happen if d = 1? If d = 1; doom will happen...
  Because 6 out of 12 stores dump their traffic into Store G, Store G quickly explodes with points. Then a few shoppers wander into Store J by pure luck. Once inside, they are trapped. They walk to K, then back to J, then back to K. Over time, _everyone_ in the mall gets stuck bouncing between J and K. Store G empties out, Store A drops to 0, and the unpopular Stores J and K end up stealing 100% of the mall's points. This is completely unrealistic! UNREAL!
  That why we put d = 85. Every time someone steps into a store:
  There is an 85% chance (0.85) they follow the doors (meaning Store A's crowd still splits to B and C, and the 6 stores still funnel into G).
  There is a 15% chance (0.15) they get bored, leave the store entirely, and magically teleport to _any_ of the 12 stores at random. Because of that 15% boredom escape hatch, the shoppers trapped in the J & K loop constantly "leak" back out and teleport to other places. The mall traffic stabilizes, and the final points give us the true ranking:
 1) Shop G - The most popular, because it had 6 pipelines sending customers into it
 2) Shop B and C - They get a steady stream of traffic because Store A funnels directly into them.
 3) Shop J and K - The Xiong'an New Area of the stores... even the own owners forget they have a store.
 - $\frac{1-d}N$ - remember we said d = 0.85, so 1 - d = 0.15, that the "boredom leak" we talked about, the 15% chance someone just abandon the store and teleport random. Now we divide that 0.15 by N (total number of stores, which in our story is 12), so basically this whole fraction just calculate "if you teleport random, how much tiny guaranteed points every single store gonna get, no matter how popular or unpopular they is

![[ChatGPT Image Jul 6, 2026, 05_37_18 PM.png|396]]
 
 - $In(v)$ - That the total amount of edge of a node that we are looking at. For example E has 2 edge that point at him, then G has 3, meanwhile A has 0.
 - $PR(v)$ - that the PageRank of the neighbors nodes, as for the one who leak into the store we look at, for example look at G, it has three nodes leaking into it, while the most important of the leaks is E, because B and F leak into it too (Sounds weird, but it is not). So G has three neighbors.
 - $L(v)$ - and this the part people always mess up first time, so listen good. This is NOT how many people go into v, this is how many doors v itself HAS GOING OUT to other places. So say Store A splits its crowd between B and C, that mean L(A)=2L(A) = 2 L(A)=2, cuz A only got 2 outgoing doors. Why this matter? Cuz when v send its points forward, it dont send ALL its points to one place, it gotta SPLIT its points evenly among all its outgoing doors. So if A got 100 points and 2 outgoing doors, each of B and C only get 50 points worth of "influence" from A, not the full 100. That is why we divide $PR(v)$ by $L(v)$ in the actual formula 

Let's solve a problem... the formula of an important and practicle something... chemistry.

EXAMPLE:

Methane:
![[image (8).png|378]]

We can see the Carbon being connected to 4 Hydrogen atoms. Now let us solve the PageRank of this compound (Of this thing):

$$PR(C) = \frac{1 - d}{N} + d \sum_{v \in In(u)} \frac{PR(v)}{L(v)}$$

We put $d = 0.85$, so we get $\frac{0.15}N$.
N will be equal to 5, because we can clearly see 5 nodes (5 atoms): $\frac{0.15}5 = 0.0300$.
So now we have:
$$PR(C) = 0.0300 + 0.85 \sum_{v \in In(u)} \frac{PR(v)}{L(v)}$$


What do we do with the last part? Since we are calculating the pagerank of C, we will do this:

$$PR(C) = 0.0300 + 0.85 \left( \frac{PR(H_1)}{L(H_1)} + \frac{PR(H_2)}{L(H_2)} + \frac{PR(H_3)}{L(H_3)} + \frac{PR(H_4)}{L(H_4)} \right)$$

because the formula changes to explicitly list Carbon's neighbors inside the summation

Look at $H_1$: It only has **1 connection** (pointing straight back to Carbon). So $L(H_1) = 1$.
Because all hydrogens are the same with one connection, $L(H_2) = 1$, $L(H_3) = 1$, and $L(H_4) = 1$.

We replace the $L(v)$ variables in the denominators with the number $1$. The formula changes again:

$$PR(C) = 0.0300 + 0.85 \left( \frac{PR(H_1)}{1} + \frac{PR(H_2)}{1} + \frac{PR(H_3)}{1} + \frac{PR(H_4)}{1} \right)$$

Since total probability is $1.0$ split across $5$ atoms, every hydrogen starts with a score of $\frac{1}{5} = 0.2000$.
We substitute the $PR(H)$ variables with 0.2000 (Imagine this as $\frac{1}N$):

$$PR(C) = 0.0300 + 0.85 \left( 0.2000 + 0.2000 + 0.2000 + 0.2000 \right)$$
And now... since the math is so "Hard", pull up your phone, call Newton, Einstein, Archimedes, and Gauss. Make them think and after a month of constant solving they will give you this:

$$\mathbf{PR(C) = Responsum- est- septuaginta-una-centesimae.}$$ 
Sadly they would probably stick to latin to find common language, so we would just use the calculator and get:

$$\mathbf{PR(C) = 0.71}$$

Which means that the PageRank score for Carbon after this calculation sets at **$0.7100$ (71% importance)


Yup, I hope you understood something at least... Otherwise remember just this:
PageRank looks  at who is linking to the node, not at how many nodes are linked, but at who.

In code that mega simple... remember the Shrimp story? We use the same code:

```python

pagerank = nx.pagerank(G)

print(f"The PageRank of Carbon is: {pagerank['C']: .4f}")

"""
Output:

The PageRank of Carbon is: 0.7100
"""
```

And now we will start with the final boss (usually considered harder than PageRank)

THE DREADED, ABYSMAL... BETWEENESS CENTRALITY. That will be some nasty stuff, because it is harder in concept and math. But clench your teeth and be ready, time to win over this month

### 8. Betweeness Centrality

If we remember Shrimp's example, his Betweeness Centrality was small, why? Not because Airi was his friend or Mr Hi. But because everybody could go wherever they want without relying on Shrimp.
(Message from the future: "it is not that hard")
Betweeness Centrality doesn't care how important your friends are (PageRank), or how many you have of them (Degree Centrality). It cares about this: If every node in the graph wants to send a message to every other node using the absolute shortest path, how many of those paths are forced to travel through you? 
This checks how important you are... because if you were the only bridge between 2 groups of 100 individuals, then your Betweeness Centrality skyrockets. The formula is... messy... don't get deceived just because it looks small:


$$C_{B}(v)=\sum _{s\ne v\ne t}\frac{\sigma _{st}(v)}{\sigma _{st}}$$
Not so scary... yet really hard, I'll explain each step:

- $v$ this is the targeted by us node
- $\sum_{s \neq v \neq t}$ - this is the strict rule of the current sum: neither the start node nor the end node can be the target node itself. (I will explain it later)
-  $\sigma_{st}$ -  this is the total number of the shortest paths that exist between a chosen starting node s and ending node t.
- $\sigma_{st}(v)$ - this counts how many of those shortest paths between s and t are forced to pass directly through our target node v.

I will explain all of this, but solving for an image.

![[image.jpg|425]]

we start from the start:
- $v$ - We choose D as targeted node, so that our $v$
- $\sum_{s \neq v \neq t}$ - The sum loop tells us that we have to find all the pairs of source $(s)$ and target $(t)$, but we have to completely exclude our target node $D$.  Since we have a total of 8 nodes (A, B, C, D, E, F, G, H), we exclude D, so we are left with 7 nodes.that means we use a formula for this (again):
  $$\text{Total Pairs} = \frac{N \times (N - 1)}{2}$$
  Since we have 7 nodes...
  $$\text{Total Pairs} = \frac{7 \times 6}{2} = 21$$
So that our total pairs. The $\sum$ symbol says that we must evaluate the path fraction for all 21 pairs individually and sum their values together...


The pairs:
- **Pairs starting with A:** (A,B), (A,C), (A,E), (A,F), (A,G), (A,H) — 6 pairs
    
- **Pairs starting with B:** (B,C), (B,E), (B,F), (B,G), (B,H) — 5 pairs
    
- **Pairs starting with C:** (C,E), (C,F), (C,G), (C,H) — 4 pairs
    
- **Pairs starting with E:** (E,F), (E,G), (E,H) — 3 pairs
    
- **Pairs starting with F:** (F,G), (F,H) — 2 pairs
    
- **Pairs starting with G:** (G,H) — 1 pair


-  $\frac{\sigma _{st}(v)}{\sigma _{st}}$ = $\frac{\text{Paths through D}}{\text{Total Shortest Paths}}$

The numerator (the top) means the nodes that forcefully pass through D, and the denominator means the total shortest path to get to the output (usually hops).
Now prepare morally.

![[Pasted image 20260707120617.png|418]]

look at the pairs, we can see:

AB -> It can directly go to B in one edge = 0/1 = 0 (It works as: Passes through D?/total number of tied shortest paths) (If it doesn't pass through D, it is immediately 0.) 
AC -> It can directly go to C in one edge = 0/1 = 0 
AE -> It can directly go to E in one edge = 0/1 = 0 
AF -> The shortest path is A -> E -> F (path length of 2 edges and completely bypasses D) = 0/1 = 0 AG -> The shortest path is A -> C -> G (path length of 2 edges and completely bypasses D) = 0/1 = 0 
AH -> The shortest path is A -> C -> H (path length of 2 edges and completely bypasses D) = 0/1 = 0

Pairs starting with B: 
BC -> The shortest path is B -> A -> C (path length of 2 edges and completely bypasses D) = 0/1 = 0 
BE -> There are two tied shortest paths: B -> A -> E and B -> D -> E (path length of 2 edges) = 1/2 = 0.5 
BF -> The unique shortest path is B -> D -> F (path length of 2 edges and passes through D) = 1/1 = 1
BG -> The unique shortest path is B -> D -> G (path length of 2 edges and passes through D) = 1/1 = 1 
BH -> The unique shortest path is B -> D -> H (path length of 2 edges and passes through D) = 1/1 = 1

Pairs starting with C: 
CE -> The shortest path is C -> A -> E (path length of 2 edges and completely bypasses D) = 0/1 = 0 CF -> The shortest path is C -> G -> F (path length of 2 edges and completely bypasses D) = 0/1 = 0 CG -> It can directly go to G in one edge = 0/1 = 0 
CH -> It can directly go to H in one edge = 0/1 = 0

Pairs starting with E: 
EF -> It can directly go to F in one edge = 0/1 = 0 EG -> There are two tied shortest paths: E -> D -> G and E -> F -> G (path length of 2 edges) = 1/2 = 0.5 
EH -> The unique shortest path is E -> D -> H (path length of 2 edges and passes through D) = 1/1 = 1

Pairs starting with F:
FG -> It can directly go to G in one edge = 0/1 = 0 
FH -> The unique shortest path is F -> D -> H (path length of 2 edges and passes through D) = 1/1 = 1

Pairs starting with G: 
GH -> There are two tied shortest paths: G -> D -> H and G -> C -> H (path length of 2 edges) = 1/2 = 0.5

Now we add everything up (That how the sum wants):
0.5 + 1 + 1 + 1 + 0.5 + 1 + 1 + 0.5 = 6.5

And so...
The Betweeness Centrality of D is of:
 $\frac{6.5}{21} = 0.3095$

Actually it is pretty easy.
That means that node D single-handedly influences roughly 31 percent of all optimal traffic.

In python you will usually do:
```python

betweenness = nx.betweenness_centrality(G)

print(f"D's Betweenness Centrality: {betweenness['D']: .4f})

"""
Output:

D's Betweenness Centrality: 0.3095

"""
```

And that pretty much it, if you want to see all the results, then you have to make a for loop... you can write it as:

```python:

for index, bc in enumerate(betweenness, 1):
	print(f"Betweenness of each node, {index}: {bc}")
```


### Progetto_Finale
Final project incoming, so we may be more than prepared right now:

![[Images/Figure_1 2.png|537]]

To generate such an image i'll show what to use.

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

# Do this in case if you want it to draw you the figure above
nx.draw(G, node_color='red', edge_color='black', node_size=1000, with_labels=True)
# This will show the figure
plt.show()

def bfs(graph, start_node):
	visited = set()
	queue = deque()
	traversial_order = []
  
	visited.add(start_node)
	queue.append(start_node)

	while len(queue) > 0:
		current = queue.popleft()
		traversial_order.append(current)
		
		for nb in graph.neighbors(current):
			
			if nb not in visited:
				visited.add(nb)
				queue.append(nb)
		
	return traversial_order

result = bfs(G, 'Central')
print(f"\nFinal BFS Traversal Order: {result}")


def dfs(graph, start_node):
	visited = set()
	stack = []
	traversial_order = []
	
	visited.add(start_node)
	stack.append(start_node)
	  
	while len(stack) > 0:
		current = stack.pop()
		traversial_order.append(current)
		
		for nb in graph.neighbors(current):
			if nb not in visited:
				visited.add(nb)
				stack.append(nb)
				
	return traversial_order

result = dfs(G, 'Central')

print(f"\nFinal DFS Traversal Order: {result}")

cycles = nx.simple_cycles(G)

print("----Cycles----")
print("")
for index, cycle in enumerate(cycles, 1):
	print(f"{index} Cycle: {' -> '.join(cycle)}")

print(f"Is the graph connected? {nx.is_connected(G)}")
print(f"Number of components: {nx.number_connected_components(G)}")


"""
Output:

<Image>

Final BFS Traversal Order: ['Central', 'Park', 'Museum', 'University', 'Airport', 'Stadium', 'Shopping']

Final DFS Traversal Order: ['Central', 'Museum', 'University', 'Stadium', 'Shopping', 'Airport', 'Park']


----Cycles----

1 Cycle: Central -> Park -> University -> Museum
2 Cycle: University -> Airport -> Shopping -> Stadium

Is the graph connected? True
Number of components: 1
"""
# The best one for reaching the 'Shoping' is clearly DFS
```
```Changes
Now we do a small change, let's say the graph is weighted! like this:
```

![[FirstFigureee.png|426]]

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


print("")
print("----Total Stats----")
print("")

for node, score in degree_centrality.items():
	print(f"\nthe degree centrality of {node} | {score: .4f}")

print("")
for node, score in page_rank.items():
	print(f"The importance of {node} (due to neighbors) | {score}")

print("")
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


the degree centrality of Central |  0.3333

the degree centrality of Park |  0.3333

the degree centrality of Museum |  0.3333

the degree centrality of University |  0.6667

the degree centrality of Airport |  0.3333

the degree centrality of Stadium |  0.3333

the degree centrality of Shopping |  0.3333

The importance of Central (due to neighbors) | 0.12554954818573194
The importance of Park (due to neighbors) | 0.12913469450799694
The importance of Museum (due to neighbors) | 0.10700042117576271
The importance of University (due to neighbors) | 0.2545872696474754
The importance of Airport (due to neighbors) | 0.1403089331311898
The importance of Stadium (due to neighbors) | 0.1293488984221484
The importance of Shopping (due to neighbors) | 0.11407023492969458

How much of a gatekeeper is Central | 0.03333333333333333
How much of a gatekeeper is Park | 0.0
How much of a gatekeeper is Museum | 0.26666666666666666
How much of a gatekeeper is University | 0.6333333333333333
How much of a gatekeeper is Airport | 0.0
How much of a gatekeeper is Stadium | 0.26666666666666666
How much of a gatekeeper is Shopping | 0.06666666666666667
"""

# Park is 0 because museum dominates it (Toxic relationship ALERT). I mean, why would you go to the park if to get to the biggest gatekeeper cost twice as much? So everybody would rather visit the museum and pay less, same for airport
# No pretty outputs nowadays
```

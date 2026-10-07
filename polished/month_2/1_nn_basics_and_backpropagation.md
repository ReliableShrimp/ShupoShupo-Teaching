# Neural Network Basics & Backpropagation

> So, after a bit of pain, we're about to suffer a lot more — because apparently we like it! That's why this is going to be a super long chapter on neural network basics and backpropagation.
> 
> I don't know how I even ended up here, but I'm fine — nothing's wrong, other than the fact that this "12-month plan" looks more like a 120-month plan compressed by some low-quality AI on Google. But hey, we're fine.
> 
> That's why I'm going to teach you neural networks! (GL)

## Table of Contents

1. Neural Network Fundamentals
    1. Why Neural Networks?
    2. Biological vs. Artificial Neurons
    3. Weights, Biases & Dot Products
    4. Hidden Layers & Representation Learning

2. Activation Functions
    1. Why Activation Functions?
    2. Sigmoid
    3. tanh
    4. ReLU
    5. Softmax
    6. Choosing the Right Activation Function

3. Building Neural Networks
    1. Multiple Neurons
    2. Matrix Multiplication
    3. Tensor Shapes & Broadcasting
    4. The Forward Pass

4. Loss Functions
    1. Why We Need a Loss Function
    2. Regression Losses (MSE & MAE)
    3. Classification Losses (BCE & CCE)
    4. Multi-Class vs. Multi-Label Classification

5. Backpropagation
    1. How Neural Networks Learn
    2. Gradients
    3. The Chain Rule
    4. Computational Graphs
    5. Gradient Descent & Weight Updates

6. Building a Neural Network with NumPy
    1. Creating the Dataset
    2. Initializing Weights
    3. Implementing the Forward Pass
    4. Computing the Loss
    5. Backpropagation
    6. Updating the Parameters
    7. Complete Training Loop

---
## 1. What Is a Neural Network?

Before we answer that question, let's answer a different one first:

Why did somebody feel the urge to invent neural networks?

Imagine you're a scientist and you want to build a model that predicts whether a patient has the flu or not.

You're given a table:

|Temperature|Cough|Headache|Flu?|
|--:|--:|--:|:-:|
|36.6|1|0|No|
|37.1|2|1|No|
|38.8|8|9|Yes|
|39.2|9|10|Yes|

So now you have your inputs (features):

```
Temperature
Cough
Headache
```

And your output (label):

```
Flu?
```

Your goal is to learn a function:

```
inputs -> ???? -> prediction
```

Since we already know linear regression, we can try using it.

For one feature: 

```math
prediction = w \times x + b
```

For many features: 

```math
w_1x_1 + w_2x_2 + ... + w_nx_n + b
```

But sadly, the answer isn't always linear — it might look more like this:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/f36e5606-3efa-47c8-a1f9-7a5e4824eb75" />

and here, a linear function won't help at all. Even if you draw a line through this data, can it pass through all the points? No. No matter how hard you try to draw that line, it can't capture all of the data.

That's our first limitation.

Now suppose you have to classify a cat based on some features:

```
Ear Size
Tail Length
Whisker Length
```

Maybe we have rules like:

```
IF
long tail

AND
pointy ears

BUT
short whiskers

THEN
Cat
```

This isn't a straight equation anymore — it's a complex interaction.

**What's the big idea?**

The big idea is that instead of using one equation, we can chain many simple equations together:

```
Input -> Small Equation -> Small Equation -> Small Equation -> Prediction
```

One equation might not recognize the cat, but many small ones together will.

Another example. Input:

```
28 × 28 image
784 pixels
```

Can a single equation understand this? No, it can't.

But we can do:

```
Image -> Detect Edges -> Detect Corners -> Detect Curves -> Detect Shapes -> Digit
```

Now each layer picks up a small piece of useful information.

This idea is called **representation**.

Representation is simply a new way of describing the same data.

If we feed in an image, the computer just sees numbers — nothing more.

So what if we use small steps instead?

The first layer might transform it into:

```
Edges
```

The second layer:

```
Corners
```

The third layer:

```
Curves
```

The fourth layer:

```
Digit Shape
```

The image hasn't changed. But its representation has.

That's why they're called **hidden layers** — because you never explicitly tell them what to learn.

```
inputs -> ???? -> output
```

Even though we create the hidden layers ourselves — the first layer (edges), the second layer (corners), and so on — they're called "hidden" because we just feed in the input, and they handle all the internal processing (passing everything through all our layers) to form the output.

That's what a neural network is.

But why do people call it "neural"?

Because it was designed to function similarly to the human brain:

```
Numbers In -> Compute -> Numbers Out
```

Main ideas:

1. **A neural network is a function.**

It takes inputs and produces outputs.

```
Input -> f(x) -> Prediction
```

2. **A neural network is built from many small functions.**

Instead of being one giant equation, it's built from small ones:

```
Input -> Small Function -> Small Function -> Small Function -> Output
```

3. **It learns by adjusting numbers called weights.**

A neural network can have anywhere from a thousand to billions of weights.

Neural networks use neurons too — but what are they?

Let me explain them biologically first.

### What Is a Neuron? (Biology)

Imagine you put your hand on a hot stove.

What happens?

Do you think:

> "Wait, I placed my hand on the stove, so now I shall take it away, for it burns!"

or:

> "O most wise and discerning hand! Tell me, friend — does this black iron box truly possess the warmth of the gods, or have I merely initiated a profound dialectic between flesh and skin-melting agony?"

Nope. Your body reacts immediately to the heat, and you instantly pull your hand away.

How?

Because your brain has billions of tiny cells that communicate with each other — neurons.

A neuron is a specialized cell whose job is to receive, process, and transmit information. Think of it as the messenger of the nervous system. Instead of carrying packages like a mail carrier, it carries electrical signals.

They're structured something like this:

```
          Dendrites
        \   |   /
         \  |  /
          \ | /
        ┌────────┐
        │  Soma  │
        │ (Cell) │
        └────────┘
             │
             │ Axon
             │──────────────────────────────►
             │
        Axon Terminals
```

- **Dendrites** are the listeners. Their job is to receive signals from other neurons (through a synapse — that's how neurons transmit information to each other). Think of them as a microphone.
    
- **Soma (Cell body)** is the decision center. Think of it as the decision center of the neuron, since it's asking: "I've received a lot of signals — should I send a new one?" This is where the neuron combines all the incoming information.
    
- **Axon** is the electrical cable. If a neuron decides to send a new signal, it travels through the axon (some axons in our bodies are over a meter long!).
    
- **Axon terminals** sit at the end of the axon, where the signal reaches many other neurons — one neuron can connect to thousands of others, forming a massive network.
    

So when we put our hand on a hot stove, the neurons act like this:

```
Neuron A -> Neuron B -> Neuron C
```

Neuron A gets a signal from a nociceptor (a specialized sensory receptor in our skin. As soon as you touch a hot stove, the cells in your hand register the sudden thermal energy. The nociceptor converts this heat into an electrochemical signal, which gets transmitted to a neuron — in our case, Neuron A).

Then Neuron B receives the signal, and if it's strong enough, passes it on to Neuron C — and eventually your muscles receive the message and pull your hand off the stove.

But here's the interesting part: neurons don't always fire.

Imagine a neuron receiving signals like:

```
+2

+1

-3

+4

-1
```

It combines everything and gets +3 — not enough to fire, so no signal.

But maybe later it receives enough input to total +7, and now it decides to fire.

This simple idea — combine many inputs and decide whether to produce an output — became the inspiration for the artificial neuron.

Around 1940, a scientist wondered:

"Can we build a mathematical version of this?"

Not a cell, but a simplified model — something like:

```
Numbers -> Decision -> Number
```

And that's how we got the artificial neuron, modeled after the biological one:

```
Biological Neuron

Signals from other neurons
          │
          ▼
     Dendrites
          │
          ▼
       Soma
 (combines signals)
          │
          ▼
      Fires or not
          │
          ▼
        Axon
          │
          ▼
  Sends signal onward
```

```
Artificial Neuron

Input numbers
      │
      ▼
 Multiply by weights
      │
      ▼
 Add everything together
      │
      ▼
 Activation function
      │
      ▼
 Output number
```

So the neural network is inspired by the real human brain — but it would still be wrong to say it's identical.

Now let's talk about artificial neurons in a bit more depth.

### Artificial Neurons

Since we understand how a biological neuron works, let's talk about the artificial neuron.

Imagine a neuron whose job is to tell whether a patient is sick or not.

It might get 3 features:

```
Temperature : 38 C
Cough : 8
Headache : 7
```

Now the neuron has to figure out how important each feature is.

It does this with... weights!

#### Weights

Suppose you're a doctor evaluating which of these features matters most:

- Favorite color
- Favorite animal
- Body temperature

Obviously, the most important feature is body temperature — not every feature should influence the prediction equally. Scientists invented something called a **weight** for exactly this reason.

A weight is simply a number that tells us how much importance to give an input.

Imagine the weights are:

```
Temperature -> 5

Cough -> 2

Headache -> 1
```

Because everybody gets stressed sometimes and can get a headache, or inhale some dust and cough.

So this makes intuitive sense:

```
Temperature is very important.

Cough is somewhat important.

Headache is a little important.
```

Now we multiply each feature by its importance (weight):

```
Temperature × Weight

Cough × Weight

Headache × Weight
```

```
Temperature:
38 * 5 = 190

Cough:
8 * 2 = 16

Headache:
7 * 1 = 7
```

Why do we multiply instead of just adding?

Because if we just added everything, the result would barely change. Multiplying lets the result change drastically. For example, imagine trying to tell if a patient is sick using:

```
Temperature: 38.6 C
How much they like reading: 100
How many times per day they throw up: 2
```

If we just add, a useless feature just sits there doing nothing useful — but by multiplying, we can do:

```
Temperature: 38.6 C * 5
How much they like reading: 100 * 0
How many times per day they throw up: 2 * 4
```

Continuing on — what do we do with these numbers? We just add them:

```
190 + 16 + 7 = 213
```

This is called the **weighted sum**.

We can do all of this in one step using the dot product:

```python
np.dot(x, w)
```

The dot product is literally saying: "multiply all the elements together, then add up the results." Every fully connected neuron starts this way.

But what happens if all our feature values equal 0?

```
Temperature = 0

Cough = 0

Headache = 0
```

Now the weighted sum becomes 0 too.

But maybe we want the neuron to have a default tendency. Maybe this neuron should lean slightly toward saying "healthy" — or slightly toward saying "sick."

How do we do that? Enter the bias!

#### Bias

Scientists introduced a new number called the **bias**.

It's a simple addition:

```
weighted sum + bias
```

Suppose:

```
weighted sum = 122

bias = -12
```

We get:

```
122 + (-12) = 110
```

The bias shifts the neuron's output before the final decision.

Imagine a teacher grading tests who decides to curve the exam by +5 points. He does:

```
grade + b
```

to every upcoming grade. For example:

```
77 + 5 = 82
```

Nothing about the underlying test changed — the teacher simply shifted the final score. The bias plays a similar role: it shifts the neuron's output independently of the inputs.

So the full picture looks like:

```
Temperature ─┐
             │ × Weight
Cough ───────┼──────────────┐
             │              │
Headache ────┘              ▼
                     Add everything
                           │
                           ▼
                      + Bias
                           │
                           ▼
                      z (output)
```

The equation, as shown before, looks like: 

```math
z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
```

But you might wonder — why do we call it "z"? Why not "Funny_Dot_Product_14_July_of_2026_at_13h_and_20m"?

Because it's convenient. Watch:

- x = inputs
- w = weights
- b = bias
- z = weighted sum before activation (we'll get to activation soon, hang tight)
- a = output after activation

But is our neuron actually useful yet?

No. Because if we stack many of these neurons together, all we get is a linear transformation (a line, nothing more). And when we have clean, roughly linear data like:

```
x = np.array([1, 5, 10, 15, 18])
y = np.array([1, 2, 3, 4, 4.75])
```

which is nearly a straight line, why would we even need a neural network? We could just use a ruler.

But sadly, as we all know, the real world is nonlinear (kind of wiggly). Everything that exists bends, curves, rotates — while we're stuck trying to use a straight line to recognize a cat?

Nope. With a purely linear function, you can't solve:

- **Images:** Identifying a cat requires recognizing curves, shadows, and textures. A straight line can't "bend" to follow a cat's outline.
- **Language:** The meaning of a word depends on context in a complex, non-straight way.
- **Graphs:** As we'll see later with GNNs, the relationships between atoms in a molecule form a complex, web-like geometry.

So the question becomes:

"How do we make a neural network capable of learning nonlinear patterns like images, speech, or language?"

The answer is...

**Activation functions.**

That's what transforms a series of linear computations into a model that can learn complex, nonlinear relationships.

We'll use:

```python
A1 = np.tanh(Z1)
```

#### Activation Functions

Now let's talk about how our lovely models can tell the difference between a line and a cat.

Most people just memorize:

```
A = np.tanh(Z)
```

without ever asking: "Why do we do that? What does it actually mean?"

Let's go back to our neurons. We know that:

```
Temperature ─┐
             │
Cough ───────┼──► Weighted Sum ─► z
             │
Headache ────┘
```

For example:

```
Temperature = 38

Cough = 8

Headache = 7
```

Weights:

```
5

2

1
```

Bias:

```
-100
```

Now the neuron does its job: 

```math
38 \times 5 + 8 \times 2 + 7 \times 1 + (-100) = 113
```

That's the output of this neuron — but let's connect it to another neuron.

With two neurons, the whole process looks like:

```
Inputs -> Neuron 1 -> Neuron 2 -> Output
```

Neuron 1 computes: 

```math
z_1 = xW_1 + b_1
```

_which is exactly what we did to get 113._

Now neuron 2 computes: 

```math
z_2 = z_1W_2 + b_2
```

It has different weights and bias, since $`z_2`$ might care more about the cough than the fever — while a third neuron, $`z_3`$, might care mostly about the headache rather than the fever or the cough.

Let's go a bit deeper. We have: 

```math
z_2 = z_1W_2 + b_2
```

but what is $`z_1`$ equal to? 

```math
z_1 = xW_1 + b_1
```

Now let's substitute the equation (remember this from high school): 

```math
z_2 = (xW_1 + b_1)W_2 + b_2
```

and now let's expand it: 

```math
z_2 = xW_1W_2 + b_1W_2 + b_2
```

Now we use composition: 

```math
W' = W_1W_2
```

```math
b' = b_1W_2 + b_2
```

so now we have: 

```math
z_2 = xW' + b'
```

That's the exact same shape as neuron 1!

That means two linear neurons are mathematically the same as one linear neuron.

The comparison looks like this:

Network A (deep linear — meaning we've stacked 2+ layers of math):

```
Input

↓

Linear

↓

Linear

↓

Output
```

Network B (shallow — just 1 layer of math):

```
Input

↓

Linear

↓

Output
```

And that explains why the second layer didn't make the model more powerful. It only made it more complicated.

Let me explain why.

Imagine you want to calculate how much tax you owe on your income ($`X`$).

- **One Layer (Network B):** You have a tax rate of **20%**.
    
    - Formula: $`X \times 0.20 = \text{Total Tax}`$
- **Two Layers (Network A):** You hire a middleman accountant.
    
    - **Layer 1:** You give your income to the accountant. They multiply it by **2** (e.g., $`X \times 2`$).
        
    - **Layer 2:** You give that result to the tax office. They multiply it by **0.10** (e.g., $`(X \times 2) \times 0.10`$).
        

**The result:** $`X \times 0.20`$.

See? You get the same result both ways.

Another example:

- **One Layer (Network B):** You use a blue filter that transmits 50% of the light and adds a blue tint.
    
- **Two Layers (Network A):**
    
    - You use a filter that transmits 70% of the light.
        
    - Then you put another filter on top that further reduces the remaining light by a factor of 1.4 ($`0.7 \div 1.4 = 0.5`$).
        
    - You're using two filters, but you're just achieving the exact same "blocking power" as the first one.
        

Another example (this one is a big one):

Let's bring back Airi Sezaki. Say that lately she's been feeling unwell, and she's only been doing three things every day:

```
Feature 1 = Read CurePre manga
Feature 2 = Sleep
Feature 3 = Get headpats from Hinako
```

But how much does she read CurePre? How much does she sleep? And how many headpats does she get per day?

```
Hours reading manga      = 2
Hours sleeping           = 6
Number of headpats       = 10
```

Okay, so we have an input vector: 

```math
x = \begin{bmatrix} 2 & 6 & 10 \end{bmatrix}
```

---

Neuron 1: "How happy does manga make Airi?"

Weights: 

```math
W_1 = \begin{bmatrix} 2 \\ 0.5 \\ 1 \end{bmatrix}
```

Bias: 

```math
b_1 = 1
```

If all the features were 0, it's reasonable that her happiness would still be 1 — it can't be 0.

Now we use the dot product: 

```math
z_1 = 2(2) + 6(0.5) + 10(1) + 1 = 18
```

---

Neuron 2: Maybe this neuron likes sleep.

Weights: 

```math
W_2 = \begin{bmatrix} 0.2 \\ 3 \\ 0.1 \end{bmatrix}
```

Bias: 

```math
b_2 = -2
```

Maybe it's skeptical, and it initially doesn't like sleep.

Dot product: 

```math
z_2 = 2(0.2) + 6(3) + 10(0.1) - 2 = 17.4
```

---

Neuron 3: Maybe this neuron LOVES headpats.

Weights: 

```math
W_3 = \begin{bmatrix} 0.1 \\ 0.3 \\ 5 \end{bmatrix}
```

Bias: 

```math
b_3 = 0
```

Maybe she'll get 0 pats from Hinako one day.

Dot product: 

```math
z_3 = 2(0.1) + 6(0.3) + 10(5) = 52
```

---

Layer 1 produces: 

```math
\begin{bmatrix} 18 & 17.4 & 52 \end{bmatrix}
```

The original information is already mixed together.

If we used a second layer here, the first layer would collapse mathematically, and we'd gain no new information beyond what we already had.

So using a neural network purely for linear transformations is highly fruitless. So don't.

Even if you added 100 layers, nothing would change.

That's why the scientists thought:

"Somewhere inside the network, we need a mathematical operation that isn't linear."

That's why they invented the activation function.

So instead of:

```
Input -> Linear -> Linear -> Output
```

they did:

```
Input -> Linear -> Activation -> Linear -> Output
```

This changes everything.

(Keep in mind that an activation function follows every layer, like this: Input -> Linear -> Activation -> Linear -> Activation -> Linear -> Output)

Now let me explain what an activation function actually looks like. The simplest way to picture one is:

```
If z > 0

↓

Output 1

Else

↓

Output 0
```

This is called a **step function**.

If we have $`z_1 = 18`$, it gets checked against the rule, and since $`18 > 0`$ is true, it becomes 1 — so now $`a_1 = 1`$.

This is great for a "yes/no" game.

But it also means the function has almost no useful gradient for learning. Since backpropagation relies on gradients to tell us how to adjust weights, this makes the step function a poor choice for training with gradient descent.

Scientists needed activations that change smoothly, so that small changes in the weights produce small, measurable changes in the output. That's why I'd like to introduce $`tanh(z)`$ — but first, I should mention that there are several activation functions, so I'll go over when to use each one, and then how to use them.

#### When to Use Each Activation Function

There are several types of activation functions. The most common ones by far are:

1. Tanh (Hyperbolic Tangent)
2. ReLU (Rectified Linear Unit)
3. Sigmoid
4. Softmax

Let's walk through some examples.

Say your model computed: 

```math
z = xW_1 + b_1
```

and you got an output like:

```
z = -100
```

Should we leave it like this? Maybe, maybe not — who knows.

That's why I'm going to teach you how to choose between them, because there are quite a few.

##### Sigmoid

Set the formula aside for a moment and just think about this example:

```
A doctor wants to know if the patient has the flu or not.

YES

or

NO
```

If the model answered with:

```
Patient = 487
```

would that be useful? Nope.

Instead, we use Sigmoid as the activation function, which gives us a result like:

```
0.96
```

meaning the model is 96% confident the patient has the flu. That's already much better.

As we know, Sigmoid gives an output between 0 and 1. That's why we always use it for a binary answer, like:

```
Cancer?

Yes

or

No
```

```
Spam?

Yes

or

No
```

```
Fraud?

Yes

or

No
```

So we use Sigmoid whenever we need answers like these.

But Sigmoid has a problem. Imagine your neuron computes: 

```math
z = 100
```

output:

```
0.99999
```

Now the next iteration, the neuron computes: 

```math
z = 101
```

output:

```
0.99999
```

and so on...

The result barely changes, and it stays stuck near its "extremes." That's what a **vanishing gradient** is.

That's why we always use it in the output layer, like:

```
Input (Medical Data)
↓
Linear Layer
↓
ReLU
↓
Linear Layer
↓
ReLU
↓
Output Layer (Logits)
↓
Sigmoid
↓
Independent Probabilities (0 to 1)
```

##### tanh

This is one of the most popular activation functions, because it doesn't jump straight from 0 to 1 — it changes gradually.

The formula: 

```math
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```

But why would we need something to move gradually instead of jumping around?

Because after the discovery of the Sigmoid activation, scientists started using it everywhere — but there was a problem. Sigmoid's output was always positive; it never reached a negative number. Why is that a problem? Imagine giving somebody instructions like:

```
take a really big step ahead

take a step ahead

take a small step ahead
```

Notice something? It can never be negative, so you can only ever move forward. That's why scientists wanted an activation function that also allowed a bit of backward movement — and that's $`tanh(z)`$.

tanh produces a number between -1 and 1. That's all it does. For example:

```
tanh(z)

100 -> 1
1 -> 0.7616
0 -> 0
-1 -> -0.7616
-100 -> -1
```

How is it used? People used it in hidden layers because it's zero-centered. That made training more stable and faster, because we'd get results like:

```
100.0 -> 1.0000
83.9 -> 1.0000
2.5 -> 0.9866
0.5 -> 0.4621
0.0 -> 0.0000
-0.5 -> -0.4621
-2.5 -> -0.9866
-83.9 -> -1.0000
```

We can see the model understands negative answers too — it can represent what's "bad" as well as what's "good."

But it has a problem too — it still suffers from vanishing gradients. Watch:

```
tanh

100 -> 1

101 -> 1
```

No difference — still messy.

It solved one issue (zero-centered outputs), but not the saturation problem.

##### ReLU

Now imagine it's around 2010, and AI models are getting a lot more layers.

50 layers. 100 layers...

Scientists realized Sigmoid was slowing things down, and tanh was still slowing things down too. So they came up with a new activation function.

**ReLU.**

This function has a simple output: 

```math
ReLU(x) = max(0, x)
```

so the outputs look like:

```
-100 -> 0
-5 -> 0
0 -> 0
0.1 -> 0.1
12 -> 12
192 -> 192
1382 -> 1382
```

Negative values become 0, and positive values stay exactly the same.

Unlike Sigmoid and tanh, which squash positive numbers, ReLU leaves them untouched. And it's really fast, because the computer doesn't need to solve any exponentials — it just checks:

```python
if x > 0:
    return x
else:
    return 0
```

When you're training billions of neurons, saving even a tiny bit of computation per neuron becomes a massive win overall.

Where is ReLU used?

```
Image Classification

↓

ReLU

------------------

Object Detection

↓

ReLU

------------------

Speech Recognition

↓

ReLU

------------------

Medical Imaging

↓

ReLU
```

Does it have a problem?

Yup — if a neuron only ever receives negative inputs, its output is always 0. That's what people call a **dead ReLU**.

##### Softmax

Softmax is very different. It's not trying to improve on Sigmoid or ReLU — it solves a completely different problem. If we're building an AI that recognizes animals, we'd use Softmax. Why? Let's see.

```
Cat
Dog
Rabbit
```

The neural network's raw answer (often called **logits**) might be:

```
Cat      8.2

Dog      5.1

Rabbit   0.3
```

Can we call these probabilities? No — they don't even add up to 1.

But by applying softmax, we get:

```
Cat      0.96

Dog      0.04

Rabbit   0.00
```

Every value is between 0 and 1, and all the probabilities add up to 1.

When do we use softmax? Whenever exactly one class should be correct. For example:

```
                              Disease

Flu

Cold

COVID

Pneumonia
```

assuming only one of these is correct.

But careful — don't use it when there can be multiple correct answers, like reading an X-ray:

```
Pneumonia

Rib Fracture

Fluid in Lung
```

A patient might have all three, or just two of them — so here we'd use ReLU for the hidden layers and Sigmoid for the output layer instead.

We use Softmax as an output layer only when we need exactly one correct answer, like:

```
Fox

Wolf

Cat
```

In which case we'd do:

```
Input
↓
Linear
↓
ReLU
↓
Linear
↓
Output Layer (Logits)
↓
Softmax
↓
Probabilities (Sum to 1.0)
```

- Fox: $`\approx 0.17`$
- Wolf: $`\approx 0.77`$
- Cat: $`\approx 0.06`$

$`0.17 + 0.77 + 0.06 = 1.0`$ (100%).

That's how softmax works.

So, tanh is fairly old-fashioned at this point — stick to ReLU for now, since its derivative is constant (= 1) for positive inputs, meaning gradients can travel much farther through deep networks. (It's faster.)

But so far we've only worked with a single neuron (and a little bit with more). Can we work with an entire network of them?

Absolutely — and now the fun part starts.

## 2. Neural Networks

Okay, okay — so maybe some of you are now thinking: but why do we even need more than one neuron?

Think about it logically: can a single neuron figure out that there's a cat in an image?

No.

A model just sees:

```
2 8 1
4 5 9
3 7 6
```

a bunch of numbers.

Because:

```
Input
↓
One Neuron
↓
Output
```

The neuron does: 

```math
z = xW_1 + b_1
```

and then maybe applies ReLU...

But this is far too simple to understand an image.

Think about hiring a detective whose only job is: "Does this animal have whiskers?" He doesn't care about anything else.

```
Not ears.
Not eyes.
Not the tail.
Only whiskers.
```

But now imagine hiring more detectives — one for the ears, one for the eyes, one for the tail.

Now we have several specialized neurons instead of just one:

```
                Image

                  │

      ┌───────────┼───────────┐

      ▼           ▼           ▼

   Neuron 1   Neuron 2   Neuron 3

   Whiskers     Ears        Tail

      ▼           ▼           ▼

```

That's how we use features to describe an image of a cat (we already know what features are). The raw pixels of an image are also features — and stacking layers lets the network build better features out of them.

Imagine this progression:

```
Raw pixels (784 numbers)
↓
Edges (horizontal, vertical)
↓
Corners
↓
Curves
↓
Eyes
↓
Face
↓
Person
```

Every layer is built on top of the previous one.

That's what we call **feature representation**.

(I actually forgot to mention this earlier, so I'll add it here: information always flows)

```
Left

↓

Right
```

This direction is called the **forward direction**. Once we get to backpropagation, information will also travel backward — we'll see how soon.

We might also have a cute question about how many neurons to use per problem. And the answer is much simpler than you'd think: we don't know. Nobody knows the perfect number in advance — figuring that out is a big part of doing ML. Good luck.

### Neural Networks Are Matrix Multiplication

"A neural network is basically a sequence of matrix multiplications and activation functions."

But that sentence alone doesn't prove anything, so let's actually prove it.

Let's go back to our sick-patient example:

```
Temperature = 38

Cough = 8

Headache = 7
```

Weights:

```
5

2

1
```

Now we can compute: 

```math
38(5) + 8(2) + 7(1)
```

That's exactly the dot product.

But what if we turn these into vectors?

```math
X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}
```

```math
W = \begin{bmatrix} 5 \\ 2 \\ 1 \end{bmatrix}
```

Now we compute: 

```math
XW
```

What is that? It's the dot product we already talked about — we just wrote it using linear algebra notation.

So now we can see why a single neuron:

```
Feature

×

Weight

+

Feature

×

Weight

+

Feature

×

Weight
```

is identical to: 

```math
XW
```

That's why AI is so full of linear algebra.

But life isn't funny enough as it is, so let's add 3 more neurons. Now we have these weights:

Neuron 1

```
5

2

1
```

Neuron 2

```
3

8

4
```

Neuron 3

```
1

6

2
```

Neuron 4

```
9

0

5
```

Now we might wonder: do we need 4 separate dot products? Technically yes, but we can compute them all at once.

We can lay it out like this:

```
             Neuron
          1   2   3   4

Temp      5   3   1   9
Cough     2   8   6   0
Headache  1   4   2   5
```

```math
W = \begin{bmatrix} 5 & 3 & 1 & 9 \\ 2 & 8 & 6 & 0 \\ 1 & 4 & 2 & 5 \end{bmatrix}
```

```
Rows = inputs
Columns = neurons
```

So let's bring back our favorite input:

```math
X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}
```

We'll end up with a row of shape 1×4.

(I hope you already know broadcasting — but in case you don't:)

```
input is a vector of:
1x3 (1 row and 3 columns)

our weight is a matrix of:
3x4 (3 rows and 4 columns)

now let's check if they're compatible:
(rule: for A×B multiplied by C×D, B and C must always match, otherwise
the matrix multiplication is undefined)

1x3 · 3x4 (the inner 3s match)

(what's left is A×D)
1x4 (our output will be a vector of 1 row and 4 columns)
```

Let's compute: 

```math
XW = \begin{bmatrix} 213 & 206 & 100 & 377 \end{bmatrix}
```

That means we got:

```
Neuron 1 → 213

Neuron 2 → 206

Neuron 3 → 100

Neuron 4 → 377
```

We got all of that by building a matrix, instead of computing 4 separate dot products.

Why does that matter? Because we won't just have 4 neurons — we'll have thousands of them, or more! That's why we really don't want to be doing:

```python
.dot()

.dot()

.dot()

.dot()
```

thousands of times.

That's why we just do:

```python
X @ W
```

And that's the whole secret behind why this approach has always been used.

Where's the bias? What about the activation? Just add them:

```python
Z = X @ W + b

A = activation(Z)
```

But somebody might complain:

> "A hospital has thousands of patients — why are we learning to compute all of this for just one patient?"

We used: 

```math
X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}
```

but if you want more patients, just add more rows: 

```math
X = \begin{bmatrix} 38 & 8 & 7 \\ 37.2 & 2 & 1 \\ 39.5 & 4 & 9 \\ 36.6 & 0 & 0 \end{bmatrix}
```

Now we have 4 patients.

Let's work through another broadcasting problem:

```
Input Features: 784
        ↓
Hidden Layer: 128 neurons
        ↓
Hidden Layer: 64 neurons
        ↓
Output Layer: 10 neurons

We have 256 patients to start (that's 256 rows).

So now let's work out all the shapes:

 W1
 Z1
 A1
 W2
 Z2
 A2
 W3
 Z3

 Solve:
 W1 = 784x128 (we know this because we start with 256x784, and we need
 the result to end in x128 — so to make the multiplication work, W1's
 first dimension has to be 784, giving us:
 256x784 · 784x128 = 256x128)
 Z1 = 256 rows and 128 columns
 A1 = same shape, values change but not the shape
 W2 = 128x64
 Z2 = 256x64
 A2 = same
 W3 = 64x10
 Z3 = 256x10
```

### Forward Pass

This is a big concept, so it might be a little brain-melting.

But the idea itself is simple: "Given some input and some weights, what prediction does the network produce?" This is called the **forward pass**.

Here's what that looks like in a real-world situation.

Think about Airi — she started practicing piano a few days ago, and now we want to build a model that predicts, based on her daily habits:

```
Good pianist

Bad pianist
```

So she keeps a daily record of three things:

```
Hours Slept

↓

Hours Practiced Yesterday

↓

Mood
```

One row might look like:

```
Hours Slept = 8

Practice Yesterday = 3 hours

Mood = 9/10
```

Another might look like:

```
Hours Slept = 5

Practice Yesterday = 1 hour

Mood = 4/10
```

So now we put it all together:

```
             Sleep   Practice   Mood

Day 1          8         3         9

Day 2          6         1         5

Day 3          9         5        10

Day 4          5         0         3
```

As a matrix (and in Python too):

```python
X = np.array([
    [8,3,9],  # Day 1
    [6,1,5],  # Day 2
    [9,5,10], # Day 3
    [5,0,3]   # Day 4
]) # Shape of 4x3
```

```math
X = \begin{bmatrix} 8 & 3 & 9 \\ 6 & 1 & 5 \\ 9 & 5 & 10 \\ 5 & 0 & 3 \end{bmatrix}
```

Suppose our hidden layer has 4 neurons (layer 1) — that's 4 columns, so we'll need 3 rows in the weight matrix so the shapes line up.

```
X = 4x3
W = 3x4

X @ W = 4x4 (4 days, 4 outputs)
meaning: each day gets a result from all 4 neurons.
```

Nobody tells a neuron to focus only on mood or only on sleep. One neuron might end up caring much more about sleep than mood. Another might care much more about mood than practice. And so on.

Suppose:

```
Day 1

Neuron 1 → 12

Neuron 2 → -4

Neuron 3 → 7

Neuron 4 → -9
```

That's the result for:

```
Z1
```

Now we apply ReLU:

```python
ReLu(Z1)
```

which leaves us with:

```python
A1 = ReLu(Z1) # Shape: 4x4
```

Suppose the second layer has only 2 neurons, since there are only 2 possible answers:

```
Good pianist

Bad pianist
```

The shape changes to make that work:

```
A1 = 4x4
W2 = 4x2

A1 @ W2 = 4x2
```

Now we're left with 4 rows and 2 columns. Let's say Day 1 looks like:

```
Day 1

Great Practice = 4.8

Bad Practice = 1.2
```

Those aren't probabilities — they're logits (raw scores). But she can't have a bad day and a good day at the same time, so we use another activation function: softmax.

```
Day 1

Great Practice

0.97

Bad Practice

0.03
```

Now we can tell that Day 1 was a good day. Everything we just did can be summarized in 4 steps:

```python
Z1 = X @ W1 + b1

A1 = ReLu(Z1)

Z2 = A1 @ W2 + b2

A2 = softmax(Z2)
```

```
Day's Features
      │
      ▼
Weighted Sum (Z1)
      │
      ▼
Activation (A1)
      │
      ▼
Weighted Sum (Z2)
      │
      ▼
Softmax
      │
      ▼
Prediction
```

That's the forward pass — the data flows forward through the network in one direction, the same left-to-right flow we described earlier.

### Loss

We already know what a loss is: literally "how wrong was the prediction?"

It's the only way we can tell if the model is lying to us or not.

Maybe the model said:

> "PERFECT GOOD DAY, GO ALL IN ON RED."

But the loss tells us what actually happened:

> "This day was so bad it deserves a black mark."

Say the model output was:

```
0.99
```

```
Reality:
1
```

```
Loss:
0.01
```

That's really good!

Now say the model output was:

```
0.02
```

```
Reality:
1
```

```
Loss:
3.9
```

That's terrible.

The loss is simply:

```
Prediction

↓

Reality

↓

Compare

↓

One Number
```

The network doesn't hear "you are wrong." It hears "you are wrong by this much."

And as we already know (it feels like we keep repeating this), the first question to ask is always:

"What kind of prediction am I making?"

#### Regression-Family Losses

**MEAN SQUARED ERROR (MSE)**

You use this for regression — continuous numbers, not a yes-or-no answer. Something like temperature, which might be 38, or 38.2.

- House price
- Salary
- Temperature
- Battery life
- Stock demand
- Age
- Blood glucose level

So what's the formula?

```math
MSE = \frac{1}{N}\sum (y - \hat{y})^2
```

We already know the math here, but let's walk through an example.

One day, Airi wants to know how many hours until the school day ends, so she guesses: 4. Reality: 5.

So how did that go?

```
5 - 4 = 1
```

Now we square the result:

```
1**2 = 1
```

A normal loss — a bit big, but nothing crazy.

Another day, she guesses: 3. Reality: 8.

```
8 - 3 = 5
5**2 = 25
```

That's a much bigger number. But why do we do this? Because huge mistakes should hurt a lot more than tiny ones. We punish bigger mistakes on purpose — imagine a self-driving car: if you tell it to park and it's off by 0.1 m, that's fine. But if it's off by 2 m? That's genuinely bad. Squaring the error makes those catastrophic mistakes dominate the loss.

**MEAN ABSOLUTE ERROR (MAE)**

MAE uses the same basic idea as MSE, except it doesn't punish bigger mistakes as harshly — it just uses the absolute value to turn negative errors positive.

It's mainly used when mistakes aren't "that big a deal" in a runaway sense. If the model is off by 1, it's off by 1. If it's off by 10, it's off by 10 — same for 20.

The model treats a mistake of 20 as exactly 20 times worse than a mistake of 1. It doesn't "panic" over big mistakes the way MSE does.

#### Classification Losses

Now we're no longer predicting numbers — we're predicting classes.

**BINARY CROSS ENTROPY (BCE)**

Suppose we're predicting:

```
Fraud?

Yes

No
```

That's why we use BCE for this kind of work. If we get:

```
0.99
```

that's good, so the loss is very small.

```
0.51
```

not great — that's uncertainty, but it's fine. Loss is moderate.

```
0.02
```

terrifying. Loss is huge.

It punishes confidently wrong outputs much more harshly, like:

```
Cat

0.02

Dog

0.95

Rabbit

0.03
```

Result:

```
Cat
```

The loss will be huge, because the model was way too confident about the wrong answer.

It won't punish nearly as much if the model says:

```
Cat

0.10

Dog

0.81

Rabbit

0.09
```

Result:

```
Dog
```

A small punishment for a bit of uncertainty.

(Categorical Cross Entropy always shows up together with softmax.)

**CATEGORICAL CROSS-ENTROPY (CCE)**

This is the same basic idea as BCE, but instead of classifying just 2 classes (Yes/No, Spam/Not Spam, Healthy/Sick), it classifies 3 or more (Yes/Maybe/No, Healthy/Unsure/Sick...).

BCE pairs with sigmoid, while CCE pairs with softmax.

For example — BCE (sigmoid), multi-label:

```python
import numpy as np

classes = ["Cancer", "Pneumonia", "Paraphilia Disorder", "Depressed"]
# Let's look at some patients (I won't build the X here, just the labels)

Y = np.array([
    [1, 0, 0, 1],  # Patient 0: Cancer, Depressed - can have more than one
    [0, 1, 0, 0],  # Patient 1: Pneumonia
    [0, 0, 1, 1],  # Patient 2: Paraphilia, Depressed - Airi
    [1, 1, 0, 0],  # Patient 3: Cancer, Pneumonia
    [0, 0, 0, 0]   # Patient 4: Healthy (no classified conditions)
], dtype=float)

# This is the sigmoid output:
Y_hat = np.array([
    [0.90, 0.10, 0.05, 0.85],  # Patient 0: strong predictions for Cancer & Depressed (good)
    [0.15, 0.80, 0.10, 0.20],  # Patient 1: strong prediction for Pneumonia (good)
    [0.05, 0.05, 0.70, 0.30],  # Patient 2: missed the Depression prediction (higher loss)
    [0.85, 0.20, 0.05, 0.10],  # Patient 3: missed the Pneumonia prediction (higher loss)
    [0.05, 0.10, 0.05, 0.15]   # Patient 4: correctly predicts low chances (good)
])

def multi_label_bce_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate element-wise loss
    loss_matrix = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Return both the detailed matrix and the mean loss
    return loss_matrix, np.mean(loss_matrix)

# Run computation
loss_matrix, overall_loss = multi_label_bce_loss(Y, Y_hat)

# Display raw results
print("--- TARGET MATRIX (Y) ---")
print(Y)
print("\n--- PREDICTION MATRIX (Y_HAT) ---")
print(Y_hat)
print("\n--- LOSS PER ELEMENT MATRIX ---")
print(np.round(loss_matrix, 4))
print(f"\nOverall Multi-Label BCE Loss: {overall_loss:.6f}")

"""
Output:

--- TARGET MATRIX (Y) ---
[[1. 0. 0. 1.]
 [0. 1. 0. 0.]
 [0. 0. 1. 1.]
 [1. 1. 0. 0.]
 [0. 0. 0. 0.]]

--- PREDICTION MATRIX (Y_HAT) ---
[[0.9  0.1  0.05 0.85]
 [0.15 0.8  0.1  0.2 ]
 [0.05 0.05 0.7  0.3 ]
 [0.85 0.2  0.05 0.1 ]
 [0.05 0.1  0.05 0.15]]

--- LOSS PER ELEMENT MATRIX ---
[[0.1054 0.1054 0.0513 0.1625]
 [0.1625 0.2231 0.1054 0.2231]
 [0.0513 0.0513 0.3567 1.204 ]
 [0.1625 1.6094 0.0513 0.1054]
 [0.0513 0.1054 0.0513 0.1625]]

Overall Multi-Label BCE Loss: 0.255051
"""
```

Now you might ask yourself:

> "How does BCE handle more than a yes-or-no answer, if it's supposed to be binary?"

It's a fair question — what it's actually doing is checking the percentage after the sigmoid, independently, for each class:

```
classes = ["Cancer", "Pneumonia", "Paraphilia Disorder", "Depressed"]

patient_0 = [0.90, 0.10, 0.05, 0.85]
0.90 -> Cancer? Yes/No -> Yes
0.10 -> Pneumonia? Yes/No -> No
0.05 -> Paraphilia Disorder? Yes/No -> No
0.85 -> Depressed? Yes/No -> Yes

patient_0 -> [1, 0, 0, 1]

and so on...
```

Meanwhile, CCE and SCCE (Sparse Categorical Cross-Entropy) look like this:

```python
import numpy as np

# Same idea as before, just:

# X Matrix: [Temperature, Cough intensity, Headache intensity]
X = np.array([
    [38.0,  8, 7],  # Patient 0: high fever, bad cough/headache -> maybe?
    [37.2,  2, 1],  # Patient 1: mild symptoms -> No
    [39.5,  4, 9],  # Patient 2: severe fever and headache -> Yes
    [36.6,  0, 0]   # Patient 3: completely normal -> No
])

# Classes: Index 0 = "Yes", Index 1 = "Maybe", Index 2 = "No"
# Only one can be true per patient
Y = np.array([
    [0, 1, 0],  # Patient 0 target: Maybe
    [0, 0, 1],  # Patient 1 target: No
    [1, 0, 0],  # Patient 2 target: Yes
    [0, 0, 1]   # Patient 3 target: No
], dtype=float)

# This is the result of the softmax:
Y_hat = np.array([
    [0.20, 0.70, 0.10],  # Patient 0: 70% chance of "Maybe" (good prediction)
    [0.05, 0.15, 0.80],  # Patient 1: 80% chance of "No" (good prediction)
    [0.40, 0.50, 0.10],  # Patient 2: model guessed "Maybe" (50%) instead of "Yes" (higher loss)
    [0.01, 0.04, 0.95]   # Patient 3: 95% confident "No" (excellent prediction)
])
# ...and so on.
```

### Backpropagation

Now we get to the interesting part, because we're about to learn backpropagation (and by the end, we'll work through plenty of coded real-life examples).

#### How Does a Neural Network Learn?

Imagine Airi is trying to learn a hard piano piece, so she gives it a few tries.

Attempt #1:

```
42 mistakes
```

On her first attempt, she made 42 mistakes.

Attempt #2:

```
31 mistakes
```

On her second attempt, just 31 mistakes...

Attempt #3:

```
17 mistakes
```

On her third attempt, she already has a better result — 17 mistakes.

Attempt #4:

```
13 mistakes
```

Now she's down to just 13 mistakes!

Attempt #31:

```
Perfect performance
```

Now she makes no mistakes at all. But...

How did she improve so much? Was it because someone replaced her piano before every attempt? No. She improved because she adjusted what she was doing after each attempt.

That's the exact same thing a neural network does.

Initially it tries:

```
Random Weights

↓

Terrible Prediction

↓

Large Loss
```

Then it:

```
Adjust the Weights

↓

Better Prediction

↓

Smaller Loss
```

And it does it again:

```
Adjust Again

↓

Even Better Prediction

↓

Even Smaller Loss
```

Eventually:

```
Very Good Weights

↓

Very Small Loss
```

That's what we call **training**!

The whole process looks like:

```
Input Data
      │
      ▼
Forward Pass
      │
      ▼
Prediction
      │
      ▼
Loss Function
      │
      ▼
How Wrong?
      │
      ▼
Backpropagation
      │
      ▼
Which Weights Caused It?
      │
      ▼
Gradient Descent
      │
      ▼
Update Weights
      │
      ▼
Repeat...
```

Every neural network architecture follows this same strategy:

- MLP
- CNN
- Transformer
- Graph Neural Network

> **Note from Shrimp:** GNNs and Transformers are the main priorities in this whole plan, so give them special attention — especially GNNs.

But we have a small problem.

Suppose we have:

```
128 neurons
```

and

```
21,000 weights
```

Output:

```
Loss = 5.12
```

Okay! So now we apply the strategy we just talked about — tweak one of the weights, tweak another one of the weights... but which of the many?

Imagine trying to tune a piano with 21,000 strings. We're not going to tune every single string one by one hoping for a better outcome. We need to answer a much more important question first:

Which exact string do I need to tune to actually get a better result?

And that's where backpropagation comes in.

#### What Is Backpropagation?

> **Story mode**
> 
> Imagine you're at school in PE class. The teacher says to you: "Go play basketball with Airi. You'll throw the ball, and Airi will tell you by how much you were wrong."
> 
> So you throw the ball (the forward pass), but it doesn't go in. So you start wondering what went wrong...
> 
> But Airi doesn't know exactly what went wrong either — she just knows one thing for sure. So she tells you: "You missed by 5 feet to the right." (the loss)
> 
> After hearing that, instead of guessing blindly on your next throw, you work backward from your mistake:
> 
> You look at your hand release: "Did my wrist flick too hard to the right?"
> 
> You look at your arm: "Was my elbow pushed out?"
> 
> You look at your stance: "Was my weight on the wrong foot?"

To figure this out, we use a bit of calculus (the chain rule).

Backpropagation doesn't say: "Your neural network is wrong."

It says: "This specific weight contributed a lot to the loss."

Maybe:

```
W1[5,12]
```

was responsible for a big chunk of the mistake.

Maybe:

```
W2[18,3]
```

barely mattered at all.

Backpropagation measures how much each weight influences the loss. That "influence" is called the **gradient**.

#### Gradient

The gradient is literally asking: "If I tweak this exact weight, how much does the loss change?"

It looks at the weight, nudges it slightly, and then checks:

```
Did the loss go:

Up?
Down?
Stay the same?
```

If the loss went up, it makes the weight smaller instead of bigger.

But like I said, we're going to use calculus for this. How does it actually help us? We use this: 

```math
\frac{\partial L}{\partial W}
```

We read this as: "the partial derivative of L with respect to W."

In plain English, that's: "how does the loss (L) change if I slightly change the weight (W)?"

If the result is positive, increasing the weight increases the mistake — we need to lower the weight.

If the result is negative, increasing the weight decreases the mistake — we need to increase the weight.

The thing that helps us track down the culprit among these 21,000 weights is the **chain rule**.

Yup, the famous chain rule — let's get into it.

#### Chain Rule

Imagine Airi has a piano concert tomorrow. Will she get an S-rank?

Obviously that doesn't depend on a single factor — it depends on a chain of factors:

```
Hours Slept
        │
        ▼
Energy
        │
        ▼
Focus
        │
        ▼
Practice Quality
        │
        ▼
Concert Performance
        │
        ▼
Judge's Score
```

Notice something? Each step depends on the one before it. For example:

```
Hours Slept = 3
```

and then everything goes wrong:

```
Less Sleep

↓

Less Energy

↓

Harder to Focus

↓

More Mistakes

↓

Lower Score
```

So if we ask: "How much did sleep affect the score?" — the answer is that it didn't affect the score directly. It affected energy. Energy affected focus. Focus affected quality. Quality affected the concert performance. And the poor performance affected the final score.

It's a connected chain. That's the chain rule: "If A affects B, B affects C, and C affects D — how much did A end up affecting D?"

The entire neural network is one giant chain.

```
Input

↓

Linear Layer

↓

tanh

↓

Linear Layer

↓

Softmax

↓

Loss
```

Say we tweak $`W_1`$ — does the loss just change, and that's it? No, because it sets off an entire chain of dominoes.

First it changes:

```
W1

↓

Z1
```

then:

```
Z1

↓

A1
```

then:

```
A1

↓

Z2
```

then:

```
Z2

↓

Softmax
```

then:

```
Softmax

↓

Loss
```

A whole chain that changes the final result.

---

The chain rule even has its own magical formula. Imagine this:

```
A
↓
B
↓
C
```

How much does A change C?

The formula is: 

```math
\frac{dC}{dA} = \frac{dC}{dB} \times \frac{dB}{dA}
```

The effect of A on C = (effect of B on C) $`\times`$ (effect of A on B).

---

Let's use a clear example:

```
Sleep
↓
Energy
↓
Performance
```

This is a simplified version of what Airi needs for a good performance. Now let's think this through.

Every extra hour of sleep improves energy by +3.

```
When Sleep is 1
Energy is 3.
```

Every point of energy improves performance by 2 points.

How much does sleep affect performance?

Easy: 

```math
\frac{dC}{dA} = \frac{dC}{dB} \times \frac{dB}{dA}
```

becomes: 

```math
\frac{dC}{dA} = \frac{2}{1} \times \frac{3}{1}
```

```math
\frac{dC}{dA} = 6
```

Every hour of sleep improves performance by 6 points.

But why do we say it goes "backward"? Because we first have to run the full forward pass and compute the loss — only then can we start backpropagation. When we apply the chain rule, we start calculating from the very last step and work our way back to $`W_1`$.

Forward pass:

```
Input
  │
  ▼
Z1
  │
  ▼
A1
  │
  ▼
Z2
  │
  ▼
A2
  │
  ▼
Loss
```

Backpropagation: 

```math
\frac{\partial \text{Loss}}{\partial W_1} = \frac{\partial \text{Loss}}{\partial A_2} \times \frac{\partial A_2}{\partial Z_2} \times \frac{\partial Z_2}{\partial A_1} \times \frac{\partial A_1}{\partial Z_1} \times \frac{\partial Z_1}{\partial W_1}
```

See? We start calculating from the last step, $`\frac{\partial \text{Loss}}{\partial A_2}`$, and work our way down to $`\frac{\partial Z_1}{\partial W_1}`$.

That's why it's called **backpropagation**:

- **Back** — we move from the loss back toward the inputs.
- **Propagation** — we pass information about the error backward through the network.

We'll come back to this in a bit, but before that, let's learn a trick that will help us.

#### Computational Graph

Now imagine we're building an AI that predicts how likely a user is to like a newly released piano piece.

Today's user is Airi.

Our AI receives many features, such as:

```
User Embedding
Recent Listening Time
Number of Piano Songs Played
Average Completion Rate
Similarity to Previous Songs
Time Since Last Session
... and many more
```

But for simplicity, let's just use 2:

```
Listening Time

Similarity Score
```

and set them to:

```
Listening Time = 8

Similarity Score = 0.9
```

Let's give Listening Time a weight of 0.7 and a bias of 2.

Instead of just writing this in Python as:

```python
z = X * W + b
```

we're going to split the operation into pieces.

```
Listening Time = 8
↓
Weight = 0.7
↓
Multiply
(8 × 0.7)
↓
5.6
```

Notice something? The computer doesn't do everything in one single step and hand you the output — it works through small steps to get there.

Let's say the weight for similarity is 5.

```
Similarity
↓
0.9
×
5
↓
4.5
```

Then:

```
5.6
+
4.5
+
2 (bias)
↓
12.1
```

Visually, it looks like this:

```
Listening Time = 8 ──×0.7──┐
                           │
                           ▼
Similarity = 0.9 ──×5────► (+) ──► +2 ──► z = 12.1
```

Let's expand the idea a bit further:

```
Listening Time
↓
Similarity
↓
Linear Layer
↓
z = 12.1
↓
tanh
↓
a = 0.9999
↓
Next Layer
```

So now our graph becomes:

```
Features
↓
Multiply
↓
Add
↓
tanh
↓
Multiply
↓
Add
↓
Softmax
↓
Loss
```

That's literally our neural network. You'll see why this view is useful a bit later.

#### NumPy Implementation

Now let's do a NumPy implementation, since it'll really help this click.

Airi is analyzing transaction data (her own transaction data, since she's trying to figure out which ones to delete without getting caught) using three features to produce three separate financial predictions.

---

- 3 Features (inputs):
    1. $`x_1`$: Account Balance (in thousands)
    2. $`x_2`$: Daily Transaction Count
    3. $`x_3`$: Risk Score (0 to 1)
- 3 Neurons (outputs):
    1. Neuron 1: Fraud Likelihood
    2. Neuron 2: Loan Approval Score
    3. Neuron 3: VIP Status Probability

---

```math
X = \begin{bmatrix} 2.5 & 12 & 0.1 \\ 0.8 & 45 & 0.9 \\ 15.0 & 2 & 0.05 \\ 1.2 & 18 & 0.4 \\ 5.4 & 8 & 0.2 \end{bmatrix}
```

On the first day, Airi had an account balance of 2,500 (thousandths aside, ¥2,500), made 12 transactions, and had a low risk score (10%). On the second day, she had just 800, made 45 transactions, and had a high risk score (90%)... poor soul.

The weights: 

```math
W = \begin{bmatrix} \mathbf{-0.2} & 0.8 & 1.5 \\ \mathbf{0.5} & -0.3 & 0.1 \\ \mathbf{1.2} & -0.9 & -0.4 \end{bmatrix}
```

The bias: 

```math
b = \begin{bmatrix} -0.5 & 0.1 & -0.2 \end{bmatrix}
```

---

Let's break this down column by column.

**Column 1 (Neuron 1 — Fraud Likelihood)**

bias = -0.5 (meaning: "I assume everything is fine unless proven otherwise") 

```math
\begin{bmatrix} \mathbf{-0.2} \\ \mathbf{0.5} \\ \mathbf{1.2} \end{bmatrix}
```

- Row 1 (Account Balance weight = -0.2): having more money makes her look less suspicious.
- Row 2 (Transaction Count weight = 0.5): making a lot of transactions makes her look a bit suspicious.
- Row 3 (Risk Score weight = 1.2): a high risk score makes her look like an outright villain!

Let's use Day 2 as an example:

- The math: $`(0.8 \times -0.2) + (45 \times 0.5) + (0.9 \times 1.2) - 0.5`$
- The result: $`-0.16 + 22.5 + 1.08 - 0.5 = 22.92`$

> What Neuron 1 says: "A score of 22.92??? Freeze the account! This is definitely fraud!"

---

**Column 2 (Neuron 2 — Loan Approval Score)**

bias = 0.1 (meaning: "I'm slightly generous by default") 

```math
\begin{bmatrix} \mathbf{0.8} \\ \mathbf{-0.3} \\ \mathbf{-0.9} \end{bmatrix}
```

- Row 1 (Balance weight = 0.8): having more money makes her look responsible, raising her approval odds.
- Row 2 (Transaction weight = -0.3): too many daily transactions make her look financially unstable (poor Airi), dragging her score down.
- Row 3 (Risk weight = -0.9): a high risk score tanks her chances completely — the bank manager wants nothing to do with her! (poor soul, once again)

Using Day 2 again as an example:

- The math: $`(0.8 \times 0.8) + (45 \times -0.3) + (0.9 \times -0.9) + 0.1`$
- The result: $`0.64 - 13.5 - 0.81 + 0.1 = -13.57`$

> What Neuron 2 says: "Absolutely not. She's completely broke and spending wildly! Nahhh — reject."

---

**Column 3 (Neuron 3 — VIP Status Probability)**

bias = -0.2 (meaning: "you aren't a VIP unless you prove it") 

```math
\begin{bmatrix} \mathbf{1.5} \\ \mathbf{0.1} \\ \mathbf{-0.4} \end{bmatrix}
```

- Row 1 (Balance weight = 1.5): a huge account balance is basically a golden ticket to the VIP club — it dominates everything else.
- Row 2 (Transaction weight = 0.1): swiping her card often makes her look like an active premium customer, but only by a small amount.
- Row 3 (Risk weight = -0.4): a bad risk score hurts her VIP chances, since the bank doesn't want trouble in its exclusive club.

As usual, let's use Day 2 as our example — Airi is broke, making lots of transactions, and carrying a bad risk score.

- The math: $`(0.8 \times 1.5) + (45 \times 0.1) + (0.9 \times -0.4) - 0.2`$
- The result: $`1.2 + 4.5 - 0.36 - 0.2 = 5.14`$

> What Neuron 3 says: "Wow — despite being broke, she still scored high enough to look like a positive VIP client. Well... once we run this through ReLU or softmax, everything gets normalized anyway."

Now, `Y` tells us what was actually true about her each day: 

```math
Y = \begin{bmatrix} \mathbf{0} & \mathbf{1} & \mathbf{0} \\ 1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
```

On the first day, she:

- 0 — wasn't committing fraud
- 1 — got her loan approved
- 0 — didn't qualify as a VIP

On the second day (the broke version of her), she:

- 1 — was flagged for fraud
- 0 — didn't get her loan approved
- 0 — didn't qualify as a VIP

Now let's run it in Python (full NumPy — still no PyTorch!):

```python
# ==== Input Layer ====

import numpy as np

# Inputs (X): 5 days of data, 3 features (Account Balance, Transactions, Risk)
X = np.array([
    [2.5,  12,  0.1 ],  # Day 1
    [0.8,  45,  0.9 ],  # Day 2
    [15.0,  2,  0.05],  # Day 3
    [1.2,  18,  0.4 ],  # Day 4
    [5.4,   8,  0.2 ]   # Day 5
])

# Weights (W): 3 features (rows) x 3 neurons (columns)
# Column 1 = Fraud, Column 2 = Loan, Column 3 = VIP
W = np.array([
    [-0.2,  0.8,  1.5],  # Weights for Account Balance
    [ 0.5, -0.3,  0.1],  # Weights for Transaction Count
    [ 1.2, -0.9, -0.4]   # Weights for Risk Score
])

# Biases (b): 1 bias for each of the 3 neurons
b = np.array([-0.5, 0.1, -0.2])

# Targets (Y): the truth about her own transaction data
Y = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
    [0, 1, 0]
])

epsilon = 1e-15
lr = 0.01
names = ["Fraud", "Loan approval", "Vip"]

# ==== Hidden layer ====
for epoch in range(2000):
	Z = X @ W + b
	A = 1/(1 + np.exp(-Z))
	
	A_clipped = np.clip(A, epsilon, 1 - epsilon)
	loss = -np.mean(Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped))
	
	# --------------------------------------------------------
	dZ = (A - Y) / X.shape[0]
	dW = X.T @ dZ
	db = np.sum(dZ, axis=0)
	# We'll explain how we got this idea below, so don't panic (yet)
	
	W -= lr * dW
	b -= lr * db  
	
	if epoch % 500 == 0:
		print(f"loop number {epoch}: loss = {loss}")

  
# ==== Output layer ====
print("FINAL PREDICTION")
print(f"loss: {loss}")
print("")

for day, prediction in enumerate(A, start=1):
	print(f"Day {day}")
	
	for name, value in zip(names, prediction):
		print(f"{name:1}: {value: .4f}")
		
		if name == "Vip":
			print("")

"""
Output:
----

loop number 0: loss = 2.69151691314985
loop number 500: loss = 0.0696149604094053
loop number 1000: loss = 0.05030490237868697
loop number 1500: loss = 0.039744957896778434
FINAL PREDICTION
loss: 0.03291615834620523

Day 1
Fraud:  0.0082
Loan approval:  0.8938
Vip:  0.0000

Day 2
Fraud:  0.9209
Loan approval:  0.0000
Vip:  0.0000

Day 3
Fraud:  0.0000
Loan approval:  1.0000
Vip:  0.9999

Day 4
Fraud:  0.1917
Loan approval:  0.0722
Vip:  0.0000

Day 5
Fraud:  0.0000
Loan approval:  0.9999
Vip:  0.0028
"""
```

Before we move on and explain a few more concepts, I'm going to derive your very soul, so be ready...

---

#### SECRET DLC — Mini Calculus Prep

##### e

We've all seen this number a thousand times... yet most of us don't really know what it means.

Picture this: in year one, you have

```
100 dollars
```

and year by year it goes:

```
100
↓
200
↓
400
↓
800
```

This becomes: 

```math
2^x
```

What if it tripled instead?

```
100
↓
300
↓
900
```

So we can see that different bases grow at different rates.

But now imagine something continuous:

```
Population.

Radioactivity.

Bacteria.

Neural networks.

Compound interest.
```

These don't grow "once a year" — they're continuous, growing at every single instant. And the magical base behind that kind of growth is: 

```math
e = 2.718281828...
```

It's the only base where growth rate = current value.

For example, let's say we have a firefly whose growth rate depends on its own weight in grams:

- If the firefly weighs 1 gram, it's growing at a rate of 1 gram per second.
- If it eats and grows to 5 grams, its growth rate instantly speeds up to 5 grams per second.
- If it grows to a whopping 100 grams, its growth rate rockets up to 100 grams per second.

So now we can see it: growth rate = current value.

That means the derivative of this beautiful base is: 

```math
\frac{\partial}{\partial x}e^x = e^x
```

and it always stays the same: 

```math
\frac{\partial}{\partial x}e^{5x} = 5e^{5x}
```

```math
\frac{\partial}{\partial x}7e^{2x} = 14e^{2x}
```

Why? Because there's something inside the parentheses: 

```math
e^{(\text{inside})}
```

so we find the derivative of what's inside (chain rule) and bring it down front. Take: 

```math
e^{(12x^2)}
```

We look at what's inside: 

```math
\frac{\partial}{\partial x}12x^2 = 24x
```

and now we bring it to the front: 

```math
\frac{\partial}{\partial x}e^{(12x^2)} = 24xe^{(12x^2)}
```

But don't get fooled — $`e`$ itself is still just a constant, since it always equals 2.718281828...

So if there's no variable attached to it, its derivative is 0: 

```math
\frac{\partial}{\partial x}e^3 = 0
```

```math
\frac{\partial}{\partial x}e = 0
```

```math
\frac{\partial}{\partial x}7e = 0
```

A trickier one: 

```math
\frac{\partial}{\partial x}\frac{e^{2x} + 12}{e^{y + 12}} = ?
```

Let's use a simple algebraic trick (e.g., $`\frac{1}{x^2} = x^{-2}`$): 

```math
\frac{e^{2x} + 12}{e^{y + 12}} = \big(e^{2x} + 12\big) \cdot e^{-(y + 12)}
```

Now let's find the derivative "with respect to x":

```math
\frac{\partial}{\partial x}\Big[\big(e^{2x} + 12\big) \cdot e^{-(y + 12)}\Big] = \left(\frac{\partial}{\partial x}\big(e^{2x} + 12\big)\right) \cdot e^{-(y + 12)}
```

We can see that "+ 12" is just a constant, so its derivative is 0. Looking inside $`e^{2x}`$, the result is: 

```math
\frac{\partial}{\partial x} = 2e^{2x} \cdot e^{-(y+12)}
```

Now let's find the derivative "with respect to y": 

```math
\frac{\partial}{\partial y} \Big[ \big(e^{2x} + 12\big) \cdot e^{-(y + 12)} \Big] = \big(e^{2x} + 12\big) \cdot \left( \frac{\partial}{\partial y} e^{-(y + 12)} \right)
```

Everything on the outside stays the same, but on the inside we have $`-(y + 12)`$, whose derivative is just $`-1`$, so we bring that to the front: 

```math
\frac{\partial}{\partial y} = -\big(e^{2x} + 12\big) \cdot e^{-(y+12)}
```

Now you might be wondering: why didn't we treat the second part as a constant too, even though it's being multiplied?

Because one of the rules of calculus says: if a constant is multiplied by a variable expression, you leave the constant alone and differentiate only the variable part. For example: 

```math
\frac{\partial}{\partial x} \big( 5 \cdot x^2 \big) = 5 \cdot \left( \frac{\partial}{\partial x} x^2 \right) = 5 \cdot 2x = 10x
```

Long story short — we're about to traumatize ourselves with logarithms, because apparently we're masochists.

##### log, ln

The most important piece here is clearly: 

```math
ln(x)
```

because it's the undo button for: 

```math
e^x
```

```
eˣ
↓
ln
↓
x
```

The derivative of $`ln(x)`$: 

```math
\frac{\partial}{\partial x}\ln(x) = \frac{1}{x}
```

Since $`e^x`$ keeps growing faster and faster, $`ln(x)`$ has to flatten out more and more — which is why its derivative is $`\frac{1}{x}`$.

A couple of examples: 

```math
5\ln(x) = 5 \times \frac{1}{x} = \frac{5}{x}
```

```math
12\ln(x) = 12 \times \frac{1}{x} = \frac{12}{x}
```

---

What happens if there's something inside the parentheses? 

```math
\ln(5x)
```

On the outside, we get: 

```math
\frac{1}{5x}
```

Now we check what's inside the parentheses: 

```math
\frac{\partial}{\partial x}5x = 5
```

so now we have: 

```math
5 \times \frac{1}{5x} = \frac{1}{x}
```

We simplified away the 5s.

---

```math
\ln(x^2 + 1)
```

- Outside derivative: $`\frac{1}{x^2 + 1}`$
- Inside derivative: $`2x`$

```math
\frac{2x}{x^2 + 1}
```

A trickier idea: $`\ln(15)`$ is just a fixed number, not an expression involving $`x`$ — so its derivative with respect to $`x`$ is simply: 

```math
\frac{\partial}{\partial x}\ln(15) = 0
```

It's a constant, so it disappears entirely. (Don't be tempted to write $`\frac{1}{15}`$ here — that shortcut only applies when the thing inside the parentheses is the actual variable, $`x`$.)

---

What about exponentials?

```math
\frac{\partial}{\partial x}2^x = 2^x\ln(2)
```

The general formula is: 

```math
\frac{\partial}{\partial x}a^x = a^x\ln(a)
```

What if $`a = e`$? 

```math
\ln(e) = 1
```

So: 

```math
e^x \times 1 = e^x
```

---

What about logarithms with other bases? Suppose: 

```math
\log_2(x)
```

Derivative: 

```math
\frac{1}{x\ln 2}
```

The general formula: 

```math
\frac{\partial}{\partial x}\log_a(x) = \frac{1}{x \ln(a)}
```

---

The main rules we'll use daily:

- **Exponential + chain rule**

```math
\frac{d}{dx}e^{f(x)} = e^{f(x)} \cdot f'(x)
```

Example: 

```math
\frac{d}{dx}e^{5x} = e^{5x} \cdot 5 = 5e^{5x}
```

Example 2: 

```math
\frac{d}{dx}e^{12x^2} = e^{12x^2} \cdot 24x = 24xe^{12x^2}
```

- **Natural logarithm + chain rule**

```math
\frac{d}{dx}\ln(f(x)) = \frac{f'(x)}{f(x)}
```

Example: 

```math
\frac{d}{dx}\ln(5x) = \frac{5}{5x} = \frac{1}{x}
```

Example 2: 

```math
\frac{d}{dx}\ln(x^2 + 1) = \frac{2x}{x^2 + 1}
```

- **Sigmoid derivative**

```math
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)
```

Example: if the current activation of a neuron is $`\sigma(z) = 0.8`$, the rate of change at that point is: 

```math
\sigma'(z) = 0.8 \cdot (1 - 0.8) = 0.8 \cdot 0.2 = 0.16
```

Example 2: 

```math
\sigma'(z) = 0.99 \cdot (1 - 0.99) = 0.99 \cdot 0.01 = 0.0099
```

And now, back to the main story.

### ReLU, Revisited

Yes, I know — we already covered this, but now we're going to understand the full logic behind it. So... brace yourself, but that's how it goes.

#### ReLU and Leaky ReLU

We already know the basic idea behind ReLU: it helps us avoid vanishing gradients. Back in the day, engineers reached for Tanh and Sigmoid for pretty much everything, but as we've already seen, that comes with problems.

**1. Saturation**

Sigmoid and tanh have a saturation problem. During backpropagation, each step multiplies by another derivative, and since the maximum value of the sigmoid derivative is only 0.25...

```math
\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)
```

...we end up multiplying together a chain of small numbers, like:

```
0.2
×0.1
×0.05
×0.02
×0.01
...
```

and pretty soon our gradient looks like:

```
Gradient
↓
0.0000000001
```

At that point, the early layers basically stop learning (their updates become vanishingly small) — and that's the vanishing gradient problem.

**2. Computation**

As we already know, sigmoid needs exponentials and division, which hits GPUs hard — computing millions of exponentials gets expensive fast.

ReLU's derivative, by comparison, is almost embarrassingly simple:

```math
\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x < 0 \end{cases}
```

It's simply: 

```math
1
```

But why does that matter?

Because with sigmoid, a chain of gradients looks like this:

```
0.25
↓
0.06
↓
0.015
↓
0.003
↓
0.0008
```

The gradient vanishes.

Meanwhile, with ReLU:

```
1
↓
1
↓
1
↓
1
↓
1
```

Why is that useful?

Suppose your network has 1,000 neurons. With sigmoid, every single neuron is always "doing work" — even if its output is 0.00001, it's still contributing.

ReLU doesn't let that happen — it drives a lot of neurons straight to 0. Those neurons simply don't participate in the forward computation for that particular input. The network becomes **sparse**.

So instead of:

```
1000 active neurons
```

we get something like:

```
470 neurons

530 inactive
```

If we're training a model to recognize a cat in an image, one neuron might be looking for:

```
whiskers
```

another might be looking at:

```
tail
```

and another at:

```
Blue sky
```

If the image has no sky in it, why should the "blue sky" neuron get to influence the prediction at all? ReLU simply switches it off — keeping only the useful "detectives" active.

Since ReLU is a piecewise function, its derivative in Python looks like:

```python
(Z > 0).astype(float)
```

But of course, everything has a price. When $`z < 0`$, the derivative quietly becomes 0 — and if a neuron keeps producing negative pre-activations for every training example, its weights simply stop changing. That's the **dead ReLU** problem.

```python
import numpy as np

# Columns: [Income, Hours, Purchases, Age]
X = np.array(
[[4.0, 8.0, 2.0, 30.0],   # Customer #1
[10.0, 12.0, 4.0, 28.0],  # Customer #2 (very rich)
[7.0, 5.0, 1.0, 35.0],    # Customer #3
])

W = np.array([[0.5], [0.2], [1.0], [-0.1]])
b = -12.0

Z = X @ W + b
A = np.maximum(0, Z)

for customer, value in enumerate(A, start=1):
	print(f"Raw ReLu output for Customer {customer}: {value}")

"""
Raw ReLu output for Customer 1: [0.]
Raw ReLu output for Customer 2: [0.]
Raw ReLu output for Customer 3: [0.]
"""
```

That's what a dead ReLU looks like. So what do we do — switch back to tanh? Maybe, but we don't really want to go back to the old stuff, so instead we use **Leaky ReLU**, which is literally:

```math
\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0.01, & x < 0 \end{cases}
```

So instead of flattening everything to 0, we use Leaky ReLU:

```python
import numpy as np

# 10 Customers with 6 features:
# [Income (k$), Hours in app, Purchases, Age, Device (1=iOS, 0=Android), Referral (1=Yes, 0=No)]
X = np.array([
[4.0, 8.0, 2.0, 30.0, 1.0, 0.0],   # Customer 1
[10.0, 12.0, 4.0, 28.0, 1.0, 1.0], # Customer 2 (very rich)
[7.0, 5.0, 1.0, 35.0, 0.0, 0.0],   # Customer 3
[12.0, 20.0, 8.0, 25.0, 1.0, 1.0], # Customer 4 (high value)
[3.0, 1.0, 0.0, 19.0, 0.0, 0.0],   # Customer 5
[8.5, 15.0, 5.0, 42.0, 1.0, 0.0],  # Customer 6
[2.5, 3.0, 0.0, 50.0, 0.0, 0.0],   # Customer 7
[15.0, 2.0, 1.0, 38.0, 1.0, 1.0],  # Customer 8
[6.0, 10.0, 3.0, 29.0, 0.0, 1.0],  # Customer 9
[11.0, 18.0, 6.0, 31.0, 1.0, 0.0]  # Customer 10
])

# Ground truth: 1.0 = Subscribed, 0.0 = Not Subscribed
y = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0]])

# Since we have 6 columns, we need 6 weights
W = np.array([[0.1], [0.15], [0.3], [-0.05], [0.2], [0.1]])
b = -3.5
lr = 0.001

for epoch in range(2000):
	Z = X @ W + b
	A = np.where(Z > 0, Z, Z * 0.01)
	# np.where is just: np.where(condition, value_if_true, value_if_false)
	error = A - y
	
	loss = np.mean(error**2)
	
	dL_dA = 2 * (A - y) / X.shape[0]
	dZ = dL_dA * np.where(Z > 0, 1, 0.01)
	db = np.sum(dZ, axis=0)
	dL_dW = X.T @ dZ
	
	W -= dL_dW * lr
	b -= db * lr

	if epoch % 500 == 0:
		print(f"Loop N {epoch}, loss of: {loss}")
		print("")

  
for customer, value in enumerate(A, start=1):
	print(f"Leaky ReLu output for Customer {customer}: {value[0]: .4f}")

"""
Output:

Loop N 0, loss of: 0.34206522499999986

Loop N 500, loss of: 0.07061126948025641

Loop N 1000, loss of: 0.06503041715406149

Loop N 1500, loss of: 0.05989998340536873

Leaky ReLu output for Customer 1: -0.0057
Leaky ReLu output for Customer 2:  0.3356
Leaky ReLu output for Customer 3: -0.0067
Leaky ReLu output for Customer 4:  1.2553
Leaky ReLu output for Customer 5: -0.0217
Leaky ReLu output for Customer 6:  1.1522
Leaky ReLu output for Customer 7: -0.0035
Leaky ReLu output for Customer 8:  0.1463
Leaky ReLu output for Customer 9: -0.0032
Leaky ReLu output for Customer 10:  1.0093
"""
```

> Keep in mind — this is just a practice run. In real production, you'd pair sigmoid with BCE, not ReLU with MSE, for a binary classification problem like this one.

## Progetto Finale

```python
import numpy as np

# X = [Uppercase Ratio, Profanity Count, Threat Words, Number of Links, Previous Reports, Comment Length]
X = np.array(
[[0.00, 0, 0, 0, 0, 6],
[0.10, 1, 0, 0, 2, 8],
[0.05, 0, 1, 0, 1, 5],
[0.40, 3, 2, 0, 5, 14],
[0.02, 0, 0, 1, 0, 10],
[0.18, 2, 0, 3, 3, 12],
[0.01, 0, 0, 0, 0, 4],
[0.35, 2, 3, 0, 6, 16],
[0.00, 0, 0, 2, 0, 11],
[0.22, 1, 1, 1, 4, 13],
], dtype=np.float64,)

# Labels = [Threat, Spam, Profanity, Harassment]
Y = np.array(
[[0, 0, 0, 0],
[0, 0, 1, 0],
[1, 0, 0, 0],
[1, 0, 1, 1],
[0, 1, 0, 0],
[0, 1, 1, 0],
[0, 0, 0, 0],
[1, 0, 1, 1],
[0, 1, 0, 0],
[1, 1, 1, 0],
], dtype=np.float64,)

names = ["Threat", "Spam", "Profanity", "Harassment"]

W = np.zeros((6, 4)) # 6 rows, since we have 6 features
b = np.array([[0.0, 0.0, 0.0, 0.0]]) # One bias per neuron
lr = 0.01
epsilon = 1e-15

for epoch in range(2000):
	Z = X @ W + b
	A = 1/(1 + np.exp(-Z))
	A_clipped = np.clip(A, epsilon, 1 - epsilon)
	
	loss = -np.mean(Y * np.log(A_clipped) + (1 - Y) * np.log(1 - A_clipped))
	
	dZ = (A - Y) / X.shape[0]
	db = np.sum(dZ, axis=0)
	dL_dW = X.T @ dZ
	
	W -= dL_dW * lr
	b -= db * lr

	if epoch % 500 == 0:
		print(f"Loop N {epoch}, loss of: {loss}")
		print("")

for num, prediction in enumerate(A, start=1):
	print(f"User {num}:")
	for name, guess in zip(names, prediction):
		print(f"{name:3}: {guess: .4f}")
		if name == "Harassment":
			print("")

"""
Loop N 0, loss of: 0.6931471805599453

Loop N 500, loss of: 0.2285337999143049

Loop N 1000, loss of: 0.16136886060708183

Loop N 1500, loss of: 0.1269960308851768

User 1:
Threat:  0.0517
Spam:  0.2520
Profanity:  0.0482
Harassment:  0.0057

User 2:
Threat:  0.3066
Spam:  0.2220
Profanity:  0.8094
Harassment:  0.0826

User 3:
Threat:  0.7130
Spam:  0.1639
Profanity:  0.2634
Harassment:  0.1469

User 4:
Threat:  0.9885
Spam:  0.0529
Profanity:  0.9990
Harassment:  0.9128

User 5:
Threat:  0.0058
Spam:  0.7866
Profanity:  0.0145
Harassment:  0.0002

User 6:
Threat:  0.0583
Spam:  0.9955
Profanity:  0.9805
Harassment:  0.0087

User 7:
Threat:  0.1138
Spam:  0.2425
Profanity:  0.0963
Harassment:  0.0240

User 8:
Threat:  0.9993
Spam:  0.0511
Profanity:  0.9984
Harassment:  0.9210

User 9:
Threat:  0.0022
Spam:  0.9739
Profanity:  0.0128
Harassment:  0.0000

User 10:
Threat:  0.8215
Spam:  0.6797
Profanity:  0.9672
Harassment:  0.0756
"""

# The loss looks reasonable — we started from all-zero weights, so this is
# about what we'd expect, especially with such a small handful of users.
```

Another one:

```python
import numpy as np

# Set seed for identical initializations
np.random.seed(9)

# 10 Days with 6 features:
# [Headpats, Hugs, Kisses, Smiles, Minutes Together, Gifts]
X = np.array([
[15.0, 6.0, 3.0, 20.0, 480.0, 4.0],  # Day 1
[2.0, 0.0, 0.0, 1.0, 50.0, 0.0],     # Day 2
[8.0, 4.0, 1.0, 12.0, 240.0, 2.0],   # Day 3
[25.0, 12.0, 8.0, 35.0, 720.0, 6.0], # Day 4
[0.0, 0.0, 0.0, 0.0, 15.0, 0.0],     # Day 5
[10.0, 5.0, 2.0, 15.0, 300.0, 1.0],  # Day 6
[4.0, 1.0, 0.0, 5.0, 90.0, 0.0],     # Day 7
[18.0, 9.0, 4.0, 22.0, 540.0, 5.0],  # Day 8
[1.0, 0.0, 0.0, 2.0, 40.0, 0.0],     # Day 9
[14.0, 7.0, 2.0, 18.0, 420.0, 3.0],  # Day 10
], dtype=np.float64)

# One-hot encoded targets: [Yes, Maybe, No]
Y = np.array([
[1.0, 0.0, 0.0], # Day 1 -> Yes
[0.0, 0.0, 1.0], # Day 2 -> No
[0.0, 1.0, 0.0], # Day 3 -> Maybe
[1.0, 0.0, 0.0], # Day 4 -> Yes
[0.0, 0.0, 1.0], # Day 5 -> No
[0.0, 1.0, 0.0], # Day 6 -> Maybe
[0.0, 0.0, 1.0], # Day 7 -> No
[1.0, 0.0, 0.0], # Day 8 -> Yes
[0.0, 0.0, 1.0], # Day 9 -> No
[1.0, 0.0, 0.0], # Day 10 -> Yes
], dtype=np.float64)

# 32 neurons
W1 = np.random.randn(6, 32) * 0.01
b1 = np.zeros((1, 32))

# 16 neurons
W2 = np.random.randn(32, 16) * 0.01
b2 = np.zeros((1, 16))

# 3 neurons
W3 = np.random.randn(16, 3) * 0.01
b3 = np.zeros((1, 3))

lr = 0.05
epsilon = 1e-15

# Hidden layer
for epoch in range(2500):
    # Training
    Z1 = X @ W1 + b1
    A1 = np.where(Z1 > 0, Z1, Z1 * 0.01)
    Z2 = A1 @ W2 + b2
    A2 = np.where(Z2 > 0, Z2, Z2 * 0.01)
    Z3 = A2 @ W3 + b3

    # Softmax
    Z = Z3 - np.max(Z3, axis=1, keepdims=True)  # keepdims preserves the row shape
    exp = np.exp(Z)
    A3 = exp / (np.sum(exp, axis=1, keepdims=True))

    # Categorical Cross Entropy
    A3_clipped = np.clip(A3, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(Y * np.log(A3_clipped), axis=1))

    # The backpropagation
    m = X.shape[0]
    dZ3 = (A3 - Y) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    dW3 = A2.T @ dZ3

    dA2 = dZ3 @ W3.T
    dZ2 = np.where(Z2 > 0, dA2, dA2 * 0.01)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dW2 = A1.T @ dZ2

    dA1 = dZ2 @ W2.T
    dZ1 = np.where(Z1 > 0, dA1, dA1 * 0.01)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    dW1 = X.T @ dZ1

    W3 -= dW3 * lr
    b3 -= db3 * lr

    W2 -= dW2 * lr
    b2 -= db2 * lr

    W1 -= dW1 * lr
    b1 -= db1 * lr

    if epoch % 500 == 0:
        print(f"Loops {epoch}, loss: {loss}")

classes = ["Yes", "Maybe", "No"]
print(f"Current loss: {loss}")

for num, prediction in enumerate(A3):
    print(f"Day {num + 1}:")
    for name, guess in zip(classes, prediction):
        print(f"{name:6}: {guess: .4f}")
    print("")

"""
Loops 0, loss: 1.0968720248744117
Loops 500, loss: 1.0947842314788425
Loops 1000, loss: 0.3922458565356477
Loops 1500, loss: 0.17582517180578713
Loops 2000, loss: 0.09990237575476019
Current loss: 0.060951108601330996
Day 1:
Yes   :  0.9746
Maybe :  0.0254
No    :  0.0000

Day 2:
Yes   :  0.0010
Maybe :  0.0330
No    :  0.9660

Day 3:
Yes   :  0.0494
Maybe :  0.9416
No    :  0.0090

Day 4:
Yes   :  1.0000
Maybe :  0.0000
No    :  0.0000

Day 5:
Yes   :  0.0009
Maybe :  0.0317
No    :  0.9673

Day 6:
Yes   :  0.2135
Maybe :  0.7860
No    :  0.0005

Day 7:
Yes   :  0.0012
Maybe :  0.0464
No    :  0.9524

Day 8:
Yes   :  0.9950
Maybe :  0.0050
No    :  0.0000

Day 9:
Yes   :  0.0010
Maybe :  0.0323
No    :  0.9667

Day 10:
Yes   :  0.8805
Maybe :  0.1195
No    :  0.0000
"""
```

> You need to be pretty comfortable with derivatives to follow backpropagation — but remember, almost nobody actually computes the full softmax Jacobian by hand. We can jump straight to $`\partial L / \partial Z_3`$ instead, which saves us from ever having to build that Jacobian matrix directly... and trust me, that thing is scary.

And with that, we arrive at a genuinely important next step: **PyTorch**, which will let us use these same ideas in far fewer lines of code — and make us look like the caffeinated ML engineers craving kudos that we secretly are (we are, and that's exactly why we're going to master it!).

So let's get into PyTorch, before the geometric hell that awaits us...

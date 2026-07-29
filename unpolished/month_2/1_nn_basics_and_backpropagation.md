
Sooo, after a bit of pain, we will suffer much more, because we like it! That why I will make a super long chapter about Neural Network basics + Backpropagation
Idk how I even ended up here, yet I am fineee, nothing is bad, beside the 12 month plan that looks mostly like a 120 month plan compressed by a low quality AI on google, but we are fine. 

That why I will learn you neural network! (Gl)
## 1. What is Neural Network?

Before we answer this question, we will answer another question:

Why did somebody felt the urge to invent the neural networks?

Imagine you are a scientist and you want to build a model that predicts if the patient has flu or it doesn't.

You are given a table:

| Temperature | Cough | Headache | Flu? |
| ----------: | ----: | -------: | :--: |
|        36.6 |     1 |        0 |  No  |
|        37.1 |     2 |        1 |  No  |
|        38.8 |     8 |        9 | Yes  |
|        39.2 |     9 |       10 | Yes  |

So now you have as inputs (Features):

```
Temperature
Cough
Headache
```

And as output (label):

```
Flu?
```

Your goal is to learn a function:

```
inputs -> ???? -> prediction
```

Since we know already the linear regression, we can try to use:
For one feature:
$$prediction = w \times x + b$$
For many features:
$$w_1x_1 + w_2x_2 +... w_nx_n + b$$

But sadly, the answer is not always linear, it may be even like this:

![[Figure_1111.png|457]]

and here a linear function may not help at all. Because even if you draw a line, can it pass through all the points? No. No matter how hard you draw that line, it can't catch all the data.

That our first limitation

Then suppose you have to rank a cat by some features:

```
Ear Size
Tail Length
Whisker Length
```

But maybe we have rules as:

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

Now this is not a straight equation, it is a complex interaction.


**What is the big idea?**

The big idea is that instead of using equation.
We may chain many simple equations together.

as:

```
Input -> Small Equation -> Small Equation -> Small Equation -> Prediction
```

One equation may not recognize the cat, but many small will.


Another example:
Input:

```
28 × 28 image
784 pixels
```

Can a single equation understand this?
No, it can't.

But we can do:
```
Image -> Detect Edges -> Detect Corners -> Detect Curves -> Detect Shapes Digit
```

And now each layer took a small useful information

This idea is called representation
Representation - is simply a new way of describing the same data

If we place an image, the computer will just see numbers, and nothing more.

What about if we use small steps?

First layer might transform it into
```
Edges
```

Second layer
```
Corners
```

Third layer
```
Curves
```

Fourth layer
```
Digit Shape
```

The image hasn't changed. But its representation did.

That why they are called hidden layers, because you never tell them what to learn

```
inputs -> ???? -> output
```

Even if we create the hidden layers, as the first layer (edges), second layers (corners), and etc...
They are called hidden because we just put the input and they do all the internal processes (Passing through all our layers) and forming our output x

That what a neural network is.

But why people call it 'Neural'?
Because it was made to function similarly to the human brain:

```
Numbers In -> Compute -> Numbers Out
```

Main ideas:

1. A neural network is a function:

It takes inputs and produces outputs.

```
Input -> f(x) -> Prediction
```

2. The neural networks is built from many small functions

Instead of being built from a really big equation, it is built from small ones:

```
Input -> Small Function -> Small Function -> Small Function -> Output
```

3. It learns by changing those numbers called weights 

A neural network may have thousand to billions of weights.


Neural network uses neurons too... what are they?

Firstly let me explain them in a biological way:

1) What is neuron? (Bio)

Imagine that you put your hand on the hot stove

What happens?
Do you think:

"Wait, I placed my hand on the stove, so now I shall take it away, cuz it burns!"
or
"O, most wise and discerning hand! Tell me, friend, does this black iron box truly possess the warmth of the gods, or have I merely initiated a profound dialectic between flesh and skin-melting agony?"

Nope
Your body reacts immediately to the heat and you immediately take away your hand
How?

Because you brain has billions of tiny cells that communicates with each other - neurons.

A neuron is a specialized cell whose job is to receive, process, and transmit the information.
Think of it as the messenger of the nervous system. Instead of carrying packages like a mail carrier, it carries electrical signals.

And they go something like:

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

- The Dendrites are the listeners. Their job is to receive the signal by other neurons (synapse - that how neurons transmit information to each other). Think of them as a microphone

- Soma (Cell) - the decision center. Think about it as the decision center of the neuron, since it asks:
  "I have received lots of signals" 
  "Should I send a new signal?"

  This is where the neurons combine all the information

- Axon - this is the electric cable. If a neuron decides to send a new signal, it will travel through the Axon (Some Axons in our body are over a meter!)

- Axon Terminals - at the end of the axon, the signal reaches many others neurons, and one neuron can connect to thousands of neurons. Forming a massive network

So when we put our hand on the hot stove, the neurons act like this:

```
Neuron A -> Neuron B -> Neuron C
```

The neuron A gets a signal by the nociceptor (A specialized sensory receptor in our skin. As soon as you put your hand on hot stove - the cells in our hands move and make and we get hit with a thermal energy. The nociceptor makes this heat into a electrochemical signal which will be transited to the neuron - In our case it is the neuron A). 

Then neuron B gets the signal, if the signal is strong enough, it sends it to the neuron C, and eventually your muscles receive the message and take your hand off the stove.

But there is an interesting part; the neurons doesn't always fire.

Imagine a neuron getting a signal like:

```
+2

+1

-3

+4

-1
```

It will combine everything and get +3; which is not enough - no signal

But maybe later it will get +7, now it decides to fire.

This simple idea - combine many inputs and decide whether to produce an output - became the inspiration for artificial neurons.

Around 1940 a scientist wondered ...

"Can we build a mathematical version of this?"

Not a cell, but a simplified model, something as:

```
Numbers -> Decision -> Number
```

And that how we get an artificial neuron instead of a biological neuron:

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

So neural network is inspired by the real human brain, but it would still be wrong to say: "It is identical"

Now we will talk about the artificial neurons a bit more.

### 1. Artificial neurons

Since we understand how a biological neuron works, now we will talk about the artificial neuron. 

Imagine a neurons gets the job is to tell whenever a patient is sick or no.

He may get 3 features as:

```
Temperature : 38 C
Cough : 8
Headache : 7
```

Now the neuron has to ask how important is each feature.

It will use.... weights!

#### 1. weights

Suppose you are a doctor and now you have to evaluate the most important features... so which is the most important:

- Favorite color
- Favorite animal
- Body temperature

Obviously the most important feature is Body temperature. So not every feature should influence the prediction equally.
Scientists invented something called a weight.

A weight is simply a number that tells us how much importance to give an input.


Imagine the weights are:

```
Temperature -> 5

Cough -> 2

Headache -> 1
```

Because everybody can get stressed and have headache or inhale some dust and cough.

So we understand this immediately.

```
Temperature is very important.

Cough is somewhat important.

Headache is a little important.
```

So now we will multiply the current features by their importance (weights):

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

Why we multiply? Why don't just add?

Because if we just add then the result barely changes.
While by multiplying the result can change drastically, because if we have:

tell if the patient is sick or no
```
Temperature: 38.6 C
How much they like reading: 100
How many times per day they throw up: 2
```

Just by adding we will let a useless feature just sit there while it means nothing, but by multiplying you can do:

```
Temperature: 38.6 C * 5
How much they like reading: 100 * 0
How many times per day they throw up: 2 * 4
```


By continuing, what do we do with the numbers?
We just add them:

```
190 + 16 + 7 = 213
```

this is called weighted sum.

We can do all of this in one step. By using the dot product we will do the same operations:

```python
np.dot(x, w)
```

because the dot product is literally saying: "I will multiply all the elements and then add the results"
Every fully connected neuron starts this way.

But what do we do if we have all the features element that equals to 0?

As:
```
Temperature = 0

Cough = 0

Headache = 0
```

Now the weighted sum will become 0 too

But maybe we want the neuron to have a default tendency

Maybe this neuron should be slightly biased toward saying "healthy."
Or slightly biased toward saying "sick".

How we can do that?...
Here eneters the bias!

#### 2. bias

The scientists introduced a new number called bias.

It will be a simple addition as

```
weighted sum + bias
```

suppose:

```
weighted sum = 122

bias = -12
```

we will do

```
122 + (-12) = 110 
```

the bias changes the neuron output before the final decision

Imagine a teacher, he is grading some tests, but he decided to curve the exam by +5 points.

so he did:

```
grade + b
```

to all the upcoming grades, for example:

```
77 + 5 = 82
```

nothing changed.
The teacher simply shifted the final score.
The bias plays a similar role: it shifts the neuron's output independently of the inputs.

So the scheme is actually:

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

The equation as I already showed, will look like:
$$z=w_1​x_1​+w_2​x_2​+w_3​x_3​+b$$

But you may wonder, why we call it 'z'? why not 'Funny_Dot_Product_14_July_of_2026_at_13h_and_20m'

because it is convenient, watch:

- x = inputs
- w = weights
- b = bias
- z = weighted sum before activation (Now we will talk about activation, wait a tad)
- a = output after activation

But is our neuron actually useful? 

No.
Because we stacked many neurons together and just got a linear transformation (A line, nothing more).
But when we have clear data as:

```
x = np.array([1, 5, 10, 15, 18])
y = np.array([1, 2, 3, 4, 4.75])
```

That almost placed in a line, why would we use neural network?? We can use a ruler.

But sadly, as we all noticed, the real world is non-linear (Kinda wiggly).
The items that exist bend, rotate, do everything existing! While we? Will we be stuck to use a linear function to identify a cat?

Nope. Sadly by using the linear function you can't solve any:

- Images: Identifying a cat requires recognizing curves, shadows, and textures. A straight line cannot "bend" to follow the outline of a cat.

- Language: The meaning of a word depends on context in a complex, non-straight way.

- Graphs: In the future GNNs, the relationship between atoms in a molecule is a complex, web-like geometry.

So now we have a question...

- "How do we make a neural network capable of learning nonlinear patterns like images, speech, or language?"

The answer is...

**Activation functions**

That what transforms a series of linear computations into a model that can learn complex, nonlinear relationships.

we will use:

```python
A1 = np.tanh(Z1)
```

#### 3. Activation functions

Now we will talk how our lovely models can see the difference between a line and a cat.

Most people will memorize what 

```
A = np.tanh(Z)
```

Without asking: "Why do we do that? what is its meaning"

Let's go back to our neurons, we know that:

```
Temperature ─┐
             │
Cough ───────┼──► Weighted Sum ─► z
             │
Headache ────┘
```

for example:
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

Now the neuron do all the job:
$$38 \times 4 + 8 \times 2 + 7 \times 1 + (-100) = 113$$

That the output of this neuron, but let's connect to another neuron

with two neurons the whole process will look like:

```
Inputs -> Neuron 1 -> Neuron 2 -> Output
```

Neuron 1 computes:
$$z_1 = xW_1 + b_1$$
     _which is exactly the stuff we did to get 113._

Now neuron 2 computes:
$$z_2 = z_1W_2 + b_2$$
It will have different weights and bias, because $z_2$ maybe cares more about the cough instead of fever, then the $z_3$ (neuron 3) will care mostly about headache rather than fever or cough

Now let's go a bit deeper, we have:
$$z_2 = z_1W_2 + b_2$$
but what is $z_1$ equal to?
$$z_1 = xW_1 + b_1$$

now lets just substitute the equation (Remember what you are learning/learnt in high school):
$$z_2 = (xW_1 + b_1)W_2 + b_2$$
and now let's just expand it:
$$z_2 = xW_1W_2 + b_1W_2 + b_2$$

and now we will use composition...
$$W' = W_1W_2$$

$$b' = b_1W_2 + b_2$$
so now we have:
$$z_2 = xW' + b'$$

That the exact same thing as neuron 1! 

That means that two linear neurons are mathematically the same as one linear neuron.

the process looks like:

Network A (Deep linear - this means that we have stacked 2+ layers of math):
```
Input

↓

Linear

↓

Linear

↓

Output
```

Network B (Shallow - that just 1 layer of math):
```
Input

↓

Linear

↓

Output
```

And that explain us the second layer didn't make the model more powerful.
It only made it more complicated.

Let me explain why.

Imagine you want to calculate how much tax you owe on your income ($X$).

- **One Layer (Network B):** You have a tax rate of **20%**.

    - Formula: $X \times 0.20 = \text{Total Tax}$

- **Two Layers (Network A):** You hire a middleman accountant.

    - **Layer 1:** You give your income to the accountant. They multiply it by **2** (e.g., $X \times 2$).

    - **Layer 2:** You give that result to the tax office. They multiply it by **0.10** (e.g., $(X \times 2) \times 0.10$).


**The result:** $X \times 0.20$.

See? You will get the same result both ways

Another example:

- One Layer (Network B): You use a blue filter that blocks 50% of light and adds a blue tint.

- Two Layers (Network A):

    - You use a filter that blocks 70% of light.

    - Then, you put another filter on top that _amplifies_ the remaining light by 1.4x ($0.7 \times 1.4 \approx 0.5$).

    - You are using two filters, but you are just achieving the exact same "blocking power" as the first one.

Another example (this one is big):

Let's remember about Airi Sezaki... Let's say that lately she was feeling unwell and she has been doing just three activity everyday, as:

```
Feature 1 = Read CurePre manga
Feature 2 = Sleep
Feature 3 = Headpats from Hinako
```

But how much does she reads CurePre? how much she sleeps? And how many headpats does she gets per day.

```
Hours reading manga      = 2
Hours sleeping           = 6
Number of headpats       = 10
```

Okay, so we have an input vector:
$$x =\begin{bmatrix} 2 & 6 & 10 \end{bmatrix}$$

---
Neuron 1:
"How happy does manga make Airi?"
Weights:
$$W_1 =\begin{bmatrix} 2 \\ 0.5 \\ 1 \end{bmatrix}$$
bias:
$$b_1 = 1$$
If the features were 0, it is normal that her happiness will be of 1, since it can't be 0

Now we will use the dot product:
$$z_1=2(2)+6(0.5)+10(1)+1=18$$
---
Neuron 2:
Maybe this neuron likes sleeping.
Weights:
$$W_2 =\begin{bmatrix} 0.2 \\ 3 \\ 0.1 \end{bmatrix}$$
bias:
$$b_2 = -2$$
maybe she is skeptical and initially she didn't like to sleep

Dot product:
$$z_2​=2(0.2)+6(3)+10(0.1)−2=17.4$$
---
Neuron 3:
Maybe this neuron LOVES headpats.
Weights:
$$W_3 =\begin{bmatrix} 0.1 \\ 0.3 \\ 5 \end{bmatrix}$$
bias:
$$b_3 = 0$$
Maybe she will get 0 pats from Hinako one day.

Dot product:
$$z_3​=2(0.1)+6(0.3)+10(5)=52$$
---
the layer 1 produces:
$$\begin{bmatrix} 18 & 17.4 & 52 \end{bmatrix}$$
The original information already mixed.

If we would use a second layer the first layer collapses, and we gain no new information beside the one we already knew.

So using neural network for linear transformations is highly fruitless. So don't.

Even if you would add 100 layers there, this will not change anything

That why the scientists though:

"Somewhere inside the network we need a mathematical operation that is not linear."

So that why... they invented the activation function.

So that why instead of:

```
Input -> Linear -> Linear -> Output
```

they did:

```
Input -> Linear -> Activation -> Linear -> Output
```

This changes everything. 
(Keep in mind that the activation functions start after every layer, as:
Inpute -> Linear -> Activation -> Linear -> Activation -> Linear -> Output)

Now I'll explain how an activation is like, the simplest way to explain it is:

```
If z > 0

↓

Output 1

Else

↓

Output 0
```
This is called a step function

if we have:
$z_1 = 18$, it will be checked by the rules and... since 18 > 0 = True, it will change it to 1, so now $z_1 = 1$ 

This is great for a "Yes/No" game.

But that means the function has almost no useful gradient for learning. Since backpropagation relies on gradients to tell us how to adjust weights, this makes the step function a poor choice for training with gradient descent.

Scientists needed activations that change smoothly, so that small changes in the weights produce small, measurable changes in the output. That why I would like to introduce $tanh(z)$, but I have to say firstly that there are some activation functions, that why I will explain in which situation to use them and then I'll explain how to use them.


### When to use each activation function

We have different types of activation functions. The most common ones are by far:

1. Tanh (Hyperbolic Tangent)
2. ReLu (Rectified Linear Unit)
3. Sigmoid
4. Softmax

So we will give some examples.
Let's say your model computed
$$z = xW_1 + b_1$$
And you got an output as:
```
z = -100
```

Shall we leave it like this? maybe yes, maybe no, who knows.

That why I am going to teach you how to choose through them, because there are many.

#### 1. Sigmoid

Leave the formula behind and think just about this example:

```
A doctor wants to know if the patient has flu or it does not.

YES

or

NO
```

If the model would answer with:

```
Patient = 487
```

Would that be useful? Nope.

Instead we will use the Sigmoid as activation function, which will give us a result as:

```
0.96
```

Meaning that the model is confident at 96% that the patient has flu. That already much better.

As we already know SIgmoid gives an output between 0 and 1. That why we always use it for a binary answer, like:

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

So we use Sigmoid just when we need answers like this.

But Sigmoid had a problem...
Imagine you neuron computes
$$z = 100$$
output:
```
0.99999
```

Now next iteration the neuron computes
$$z = 101$$
output
```
0.99999
```

and so on...

The result barely changes, and it is always stuck near it 'extremes'. So that what a vanishing gradient is.

That why use it always in the output layer as

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
#### 2. tanh

This activation is one of the most popular activations, because it doesn't jump from 1 and 0, it changes gradually.
The formula:
$$f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

But when would need it to move gradually instead of jumping from 1 to 0 and so on?
Because after the discover of the Sigmoid activation, the scientists were spamming it everywhere... but there was a problem. The Sigmoid activation was always positive, it never reached a negative number, but why is it a problem? Imagine...

You are giving to somebody instructions:

```
take a really big step ahead

take a step ahead

take a small step ahead
```

Noticed something? It can't be negative, so we can only move forward, that why scientists wanted an activation function that permits them to go even a bit backwards, and that was $tanh(z)$

tanh can produce a number between -1 -> 1. That all it can do. For example

```
tanh(z)

100 -> 1
1 -> 0.7616
0 -> 0
-1 -> -0.7616
-100 -> -1
```

How is it used?
People used it in the hidden layers, because it was zero-centered. 
It meant that it was more stable and faster at training the model, because if we got an answer as:

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

We can see that the model understand even negative answers. Seeing what is bad and good.

But it has a problem... again.
It still has the vanishing gradient problem, watch:

```
tanh

100 -> 1

101 -> 1
```

No difference, so it is still messy.

It solved one issue (zero-centered outputs), but not the saturation problem.

#### 3. ReLu

Now imagine we are in the 2010, the ai gets much more layers.

50 layers
100 layers...

That why the scientists said that the Sigmoid slows them down, the tanh still slows them down...
So they though about a new activation function. 

**ReLu**

This function had such an output:
$$ReLu(x) = max(0, x)$$

so the outputs were like:

```
-100 -> 0
-5 -> 0
0 -> 0
0.1 -> 0.1
12 -> 12
192 -> 192
1382 -> 1382
```

Negative values become 0 and the positive stay always the same.

Unlike Sigmoid and tanh that squashes the positive number, ReLu doesn't touch it.
And it is really fast, because the computer doesn't solve exponents it only checks:

```python
if x > 0:
    return x
else:
    return 0
```

because when you're training billions of neurons, saving a tiny amount of computation per neuron becomes a huge win.

Where is ReLu used?

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

Does it has a problem?

Yup, if we have negative inputs only the value of the result will be of 0. That what people calls a Dead ReLu.


#### 4. Softmax

Softmax is very different. It is not trying to improve sigmoid or ReLU. It solves a completely different problem. If we build an Ai that can recognize animals, we would use Softmax, why? let's see.

```
Cat
Dog
Rabbit
```

The neural network raw answer (The raw answer is called logits many times) will be:

```
Cat      8.2

Dog      5.1

Rabbit   0.3
```
Can we call them probability? No. They don't even add up to 1.

But by using softmax we get:

```
Cat      0.95

Dog      0.04

Rabbit   0.01
```

Every probability is between 0 and 1.
All probabilities add up to 1

When do we use softmax?
Whenever exactly one class should be correct.

for example:

```
                              Disease

Flu

Cold

COVID

Pneumonia
```

Supposing that only one is correct.

But wait! Don't use it when there are many correct answers, for example a X-Ray:

```
Pneumonia

Rib Fracture

Fluid in Lung
```

A patient may have all three or just 2 of them, so that why here we will use ReLu as hidden layer and Sigmoid as output layer.

We use Softmax as an output layer and only when we need to get one correct answer.

As we have just:

```
Fox 

Wolf

Cat
```

Then we will do:

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

- Fox: $\approx 0.17$
- Wolf: $\approx 0.77$
- Cat: $\approx 0.06$

$0.17 + 0.77 + 0.06 = 1.0$ (100%).

That how softmax work.

So tanh is pretty... old fashion, so stick to ReLu for now. Because Its derivative is constant (=1) for positive inputs, so gradients can travel much farther through deep networks. (It is faster)

But for now we worked with a single neuron (And a bit with more)...
Can we work with an entire network of them? 

Absolutely, and now the funny part starts.

## 2. Neural network

Okay, okay, so maybe one of you now thinks... But why shall we even work with more neurons??

Think logically, can a single neuron figure out that there is a cat on the image?
No.

A model can see just:
```
2 8 1
4 5 9
3 7 6
```
a amount of numbers

Because:
```
Input 
↓ 
One Neuron 
↓ 
Output
```

The neuron does:
$$z = xW_1 + b_1$$
and then maybe applies ReLu...

But this is far too simple to understand an image...

Think about hiring a detective, but his job is only: 
"Does this animal has whiskers?"
He doesn't care about anything else.

```
Not ears.
Not eyes.
Not the tail.
Only whiskers.
```

But imagine hiring more detectives, one for the ears, one for the eyes, one for the tail.

And now we have more specialized neurons instead one:

```
                Image

                  │

      ┌───────────┼───────────┐

      ▼           ▼           ▼

   Neuron 1   Neuron 2   Neuron 3

   Whiskers     Ears        Tail

      ▼           ▼           ▼

```

That how we used our features for describing an image of a cat (We already know what features are)
the raw pixels of the image are also features. 
They create better features.

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

Every layer is built on the previous one.

That what we call: Feature Representation.

(I actually forgot to say it before, so I'll just add it here)
Sooooo, I wanted to say that the information flows:

Always
```
Left

↓

Right
```

This direction is called the forward direction. But when we will learn backpropagation, the information will also travel backwards. We will see how.

And we may have a cute question about the number of neurons... How many do we add per problem?
And the answer is much simpler than you may think! 
We don't know. Because nobody knows the perfect number in advance... that a great part of ML, so good luck.

### 1. Neural Network is matrix multiplication 

"A neural network is basically a sequence of matrix multiplications and activation functions."

But that sentence alone will discover nothing, that why I am going to prove it.

Let's go back to our idea of the sick patient:

```
Temperature = 38

Cough = 8

Headache = 7
```

weights:
```
5

2

1
```

now we can do:
$$38(5) + 8(2) + 7(1)$$
That exactly the dot product.

But what if we make a vector out of them:

$$X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}$$

$$W = \begin{bmatrix} 5 \\ 2 \\ 1 \end{bmatrix}$$
Now we compute:
$$XW$$
What is it?
It is the dot product that we talked about. 
We just wrote them according to linear algebra.

So now we can understand why one neuron:

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

Is identical to:
$$XW$$
That why AI is full of linear algebra.

But imagine that life is not funny enough, so what do we do? We add another 3 neurons.
Now we have as weights:

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

But now we may have a question... Do we need 4 different dot products? Yes, but we can do them all at once.

We can do like this:

```
             Neuron
          1   2   3   4

Temp      5   3   1   9
Cough     2   8   6   0
Headache  1   4   2   5
```
$$W = \begin{bmatrix} 5 & 3 & 1 & 9 \\ 2 & 8 & 6 & 0 \\ 1 & 4 & 2 & 5 \end{bmatrix}$$
```
Rows = inputs
Columns = neurons
```

So we have to remember about our lovely import:

$$X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}$$

We will get a row of 1x4 
(I hope you know how to do broadcasting... yet if you don't know...)

```
input is a vector of:
1x3 (1 row and 3 columns)

our weight is a matrix of:
3x4 (3 rows and 4 columns)

now let's see if they are compactible:
(formula:
AxB  CxD, B and C must be always equal, otherwise we can't do a matrix multiplication)

1x3 3x4 (B and C cease)

(Now we have left AxD)
1x4 (our outpt will be a vector of 1 row and 4 columns)
```

Suppose we compute:
$$XW = \begin{bmatrix} 213 & 174 & 96 & 377 \end{bmatrix}$$

That mean that we got:
```
Neuron 1 → 213

Neuron 2 → 174

Neuron 3 → 96

Neuron 4 → 377
```

We did all of that by making a matrix, instead of computing 4 different dot products.

Why do we do that? We do that because we will not have 4 neurons, we will thousands of them or more!
That why we don't really feel like doing

```python
.dot()

.dot()

.dot()

.dot()
```
Thousands of times.

That why we do just

```python
X @ W
```

And that the whole secret why we used this even in the past.

Where is the bias?? What about the activation??
Just add the
```python
Z = X @ W + b

A = activation(Z)
```

But somebody may complain as: 
"But a hospital has thousands of patients, so why would we learn how to compute all of this just for a patient??"

We used:
$$X = \begin{bmatrix} 38 & 8 & 7 \end{bmatrix}$$
But if you want more patients, just add more rows, as:
$$X = \begin{bmatrix} 38 & 8 & 7 \\ 37.2 & 2 & 1 \\ 39.5 & 4 & 9 \\ 36.6 & 0 & 0 \end{bmatrix}$$

Now we have 4 patients.

Let's solve another broadcasting problem:

```
Input Features: 784
        ↓
Hidden Layer: 128 neurons
        ↓
Hidden Layer: 64 neurons
        ↓
Output Layer: 10 neurons

We have 256 patients at the start (that 256 rows)

So now we will discover all of this:

 W1
 Z1
 A1
 W2
 Z2
 A2
 W3
 Z3
 
 Solve:
 w1 = 784x128 (We discovered that because we had from the start 256x784 and we knew that we have ''x128, that why to make it work you will add 784, so now we had:  
 256x784 784x128 = 256x128)
 Z1 = 256 rows and 128 columns 
 A1 = same shape, change values not shape 
 w2 = 128x64 
 z2 = 256x64 
 a2 = same 
 w3 = 64x10 
 z3 = 256x10
```

### 2. Forward pass

This is a big concept, soooo, it may be a tad brain melting.

But the idea is simple:
"Given some input and some weights, what prediction does the network produce?"
This is called the forward pass.

In a real world situation is like this:

Think about Airi, she started to practice the piano some days ago. And now we want to build a model for her that will say (Based on the days): 

```
Good pianits

Bad painist
```

So she keep record for us everyday of three things:

```
Hours Slept

↓

Hours Practiced Yesterday

↓

Mood
```

One row may look like:

```
Hours Slept = 8

Practice Yesterday = 3 hours

Mood = 9/10
```

The other like:

```
Hours Slept = 5

Practice Yesterday = 1 hour

Mood = 4/10
```

So now we put them together~

```
             Sleep   Practice   Mood

Day 1          8         3         9

Day 2          6         1         5

Day 3          9         5        10

Day 4          5         0         3
```

matrix look (in python too):

```python
X = np.array([
    [8,3,9], # That day 1
    [6,1,5], # That day 2
    [9,5,10], # That day 3
    [5,0,3] # That day 4
]) # Shape of 4x3
```

$$X = \begin{bmatrix} 8 & 3 & 9 \\ 6 & 1 & 5 \\ 9 & 5 & 10 \\ 5 & 0 & 3 \end{bmatrix}$$

Suppose that our hidden layer has 4 neurons (layer 1)  (4 columns, and we will make 3 rows, so they match and make it possible)

```
X = 4x3
W = 3x4

X @ W = 4x4 (4 days 4 outputs)
means: 1 day will have the result of 4 neurons.
```

Nobody tells to the neuron that we will focus just on mood or sleep.
A neuron may prefer much more the sleep instead of mood
Another may prefer much more the mood instead of of practice
and so on...

Presume:
```
Day 1

Neuron 1 → 12

Neuron 2 → -4

Neuron 3 → 7

Neuron 4 → -9
```

That the result for:
```
Z1
```
Now we will use the ReLu:

```python
ReLu(Z1)
```

Now we are left with

```python
A1 = ReLu(Z1) # Shape: 4x4
```

Suppose that the second layer has only 2 neurons, because there are only 2 possible answers:

```
Good pianits

Bad painist
```

The shape will change, since we will make it work:

```
A1 = 4x4
Z2 = 4x2

A1 @ Z2 = 4x2
```

Now we are left with 4 rows and 2 columns.
Let's say that our day 1 looks like:

```
Day1

Great Practice = 4.8

Bad Practice = 1.2
```

Those are not probabilities, those are logits (Raw score). But you can't have a bad and a good day at the same time, that why we will use another activation function - the softmax:

```
Day1

Great Practice

0.97

Bad Practice

0.03
```

Now we can understand that in the first day she had a good day. All we did can be summarized in 4 steps:

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

That the forward pass, because the data flows from right to left (In a sequence like the one you see)

### 3. Loss

We already know what a loss is, it is literally 
"How wrong was the prediction?"

It is the only way we can understand if the model is lying or no.

Because maybe the model said: "PERFECT GOOD DAY, GO ALL IN ON RED"
But by the loss we understand that in reality it was: "This day was so bad that I shall put a black mark on it."

Let's say that the model gave as output:

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

Now if the model output is

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

That's terrible

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

The network doesn't hear:
"You are wrong"

It hears:
"You are wrong by this much"

And as we already know (It looks like we are repeating everything we already learnt...)

The first question anyone has to give is:

"What kind of prediction am I making?"

#### 1. Regression family losses

**MEAN SQUARE ERROR (MSE)**

You use them for regressions (Continuous numbers: not a yes or no. It is something as the temperature that it is 38, and then 38.2)

- House price
- Salary
- Temperature
- Battery life
- Stock demand
- Age
- Blood glucose level

So what is the formula?
The formula:
$$MSE = \frac{1}N\sum (y - \hat{y})^2$$

We already know the math for this one, but I'll give an example:

One day:

Suppose Airi wants to know after how many hours the school day ends, so she tries to guess by saying: 4
Reality: 5

So how it was?

```
5 - 4 = 1
```

now we square the output:
```
1**2 = 1
```

A normal loss, pretty big-ish.

Another day:

She tries to predict by saying: 3
Reality: 8

```
8 - 3 = 5
5**2 = 25
```

That a really big number there. But why do we do it?
Because huge mistakes should hurt much more than tiny ones.
So we just punish the bigger mistakes because imagine using a self driving car, if you commanded your car to park and it was off by 0.1m it is fine, but if it is off by 2m? That really bad, That why we square it, because squaring makes those catastrophic mistakes dominate the loss.


**MEAN ABSOLUTE ERROR (MAE)** 

The MAE uses the same idea as the MSE, the problem is that we don't penalize the bigger mistakes so badly, we just use the absolute value to make from negative outputs positive.

But it is mainly used when the mistakes are not 'such a big deal'.

For example if the model is off by 1 it is off by 1, if it is off by 10, it is off by 10, same for 20.

Because the model treats a mistake of 20 as exactly 20 times worse than a mistake of 1. It doesn't "panic" about big mistakes as much as MSE does.

#### 2. Classification Losses

Now we're no longer predicting numbers.
We're predicting classes.

**BINARY CROSS ENTROPY (BCE)** 

Supose we are predicting if:

```
Fraud?

Yes

No
```

That why we use the BCE for such a work, because if we get

```
0.99
```
This is good, so the loss is very small.

```
0.51
```
Not really okay, it is uncertainty. But fine. Loss is moderate

```
0.02
```
Terrifying. Loss is huge.

It punishes much more the confidently wrong outputs as:

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

The loss will be huge, because it is way too confident on what it shouldn't.

He will not punish really much if the model says:

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
Congrats, a small punishment for uncertainty.

(Categorical Cross Entropy appears always with softmax)

**CATEGORICAL CROSS-ENTROPY**

This is the same idea as the BCE, but instead of classifying just 2 classes (Yes/No, Spam/NotSpam/, Healthy/Sick) It will classify from 3+ classes (Yes/Maybe/No, Healthy/Unsure/Sick...).

For BCE you will use sigmoid while for CCE you will use softmax.

For example:
BCE (Sigmoid) - multiple answer:
```python
import numpy as np

classes = ["Cancer", "Pneumonia", "Paraphilia Disorder", "Depressed"]
# We will look at some patients (The X which I will not make, but I will just give the labels)

Y = np.array([
    [1, 0, 0, 1],  # Patient 0: Cancer, Depressed - they can have more than one
    [0, 1, 0, 0],  # Patient 1: Pneumonia
    [0, 0, 1, 1],  # Patient 2: Paraphilia, Depressed - Airi
    [1, 1, 0, 0],  # Patient 3: Cancer, Pneumonia
    [0, 0, 0, 0]   # Patient 4: Healthy (no classified conditions)
], dtype=float)

# 3. That the sigmoid output:
Y_hat = np.array([
    [0.90, 0.10, 0.05, 0.85],  # Patient 0: strong predictions for Cancer &                                                                     Depressed (Good)
    [0.15, 0.80, 0.10, 0.20],  # Patient 1: strong prediction for Pneumonia                                                                           (Good)
    [0.05, 0.05, 0.70, 0.30],  # Patient 2: missed Depression prediction (Higher                                                                       Loss)
    [0.85, 0.20, 0.05, 0.10],  # Patient 3: missed Pneumonia prediction (Higher                                                                        Loss)
    [0.05, 0.10, 0.05, 0.15]   # Patient 4: correctly predicts low chances (Good)
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

Now you may ask yourself "HOW IN THIS ABYSMALL WORLD DID BCE GOT MORE IDEAS THAN ONE SINCE IT CAN DO JUST YES OR NO"

Maybe unrelated but that what it did, It checked the percentages after the Sigmoid.

```

classes = ["Cancer", "Pneumonia", "Paraphilia Disorder", "Depressed"]

patient_0 = [0.90, 0.10, 0.05, 0.85]
0.90 -> Cancer? Yes/No -> Yes
0.10 -> Pneumonia? Yes/No -> No
0.05 -> Paraphilia Disorder? Yes/No -> No
0.85 -> Depressed? Yes/No -> Yes

patient_0 = [1, 0, 0, 1]

and so on...
```

While CCE and SCCE will look like this:

```python
import numpy as np

#Same stuff from before, just:

# X Matrix: [Temperature, Cough intensity, Headache intensity]
X = np.array([
    [38.0,  8, 7],  # Patient 0: High fever, bad cough/headache -> Maybe?
    [37.2,  2, 1],  # Patient 1: Mild symptoms -> No
    [39.5,  4, 9],  # Patient 2: Severe fever and headache -> Yes
    [36.6,  0, 0]   # Patient 3: Completely normal -> No
])

# Classes: Index 0 = "Yes", Index 1 = "Maybe", Index 2 = "No"
# They can have just 
Y = np.array([
    [0, 1, 0],  # Patient 0 Target: Maybe
    [0, 0, 1],  # Patient 1 Target: No
    [1, 0, 0],  # Patient 2 Target: Yes
    [0, 0, 1]   # Patient 3 Target: No
], dtype=float)


# This is the result of the softmax:
Y_hat = np.array([
    [0.20, 0.70, 0.10],  # Patient 0: 70% chance of "Maybe" (Good prediction)
    [0.05, 0.15, 0.80],  # Patient 1: 80% chance of "No" (Good prediction)
    [0.40, 0.50, 0.10],  # Patient 2: Model guessed "Maybe" (50%) instead of                                                         "Yes" (Higher Loss)
    [0.01, 0.04, 0.95]   # Patient 3: 95% confident "No" (Excellent prediction)
])
# And so on...
```

### 4. Backpropagation

Now here we have the interesting part. Because now we will learn about backpropagation (In the end we will do give many coded real life situations).

#### 1. How does a Neural Network learn?

Imagine Airi is trying to learn a hard piano piece, so she give it some tries.

#1 Attempt:
```
42 mistakes
```

On her first attempt she made 42 mistakes.

#2 Attempt:
```
31 mistakes
```

On her second attempt she made just 31 mistakes...

#3 Attempt:
```
17 mistakes
```

On her third attempt she has already a better result - 17 mistakes

#4 Attempt:
```
13 mistakes
```

Now she has only 4 mistakes!

#31 Attempt
```
Perfect performance
```

Now she had no mistakes, but...

How did she improved so much?
Did she improved so much because someone replaced her piano every attempt? No.
She improved because she adjusted what she was doing after each attempt.

That the same stuff that a neural network does:

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

That what we call Training!

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

Every existing Neural Network follow this strategy:
- MLP
- CNN
- Transformer
- Graph Neural Network

(Note by author - Shrimp: The GNN and Transformer are the main priorities in this whole plan, so give them special attention. Especially to GNNs)


But we have a small problem... 

Presume we have:
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

Okay! So now we use the strategy we talked about, we will tweak one of the weights... tweak one of the weights.... which of the many?

Imagine trying to tune a piano with 21k strings, we will not tune all the strings one by one just to get a better outcome, we have to understand a really important thing:

Which exact string I have to tune so I get a bettwe answer?

And that where backpropagation comes in.

#### 2. What is backpropagation?

```Storymode

Imagine that you are at school at the PE lesson. The teacher says to you:

"Go and play basketball with Airi. You will throw the ball and Airi will tell you by how much you were wrong"

So you go and throw the ball (Forward pass), but it doesn't enter the net. 
So you think what was wrong...

But Airi also doesn't know what was wrong, she just knows one thing for sure. That why she says to you: "You missed by 5 feet to the right" (Loss)

And then after you heard the news, instead of guessing blindly for the next throw, you work backward from your mistake:

You look at your hand release:
"Did my wrist flick too hard to the right?"

You look at your arm:
"Was my elbow pushed out?"   

You look at your stance:
"Was my weight on the wrong foot?"
```

To figure it out we will use a bit of calculus (Chain rule) to figure out what was wrong.

The backpropagation will not say:
"Your Neural Network is wrong."

It says:
"This specific weights contributes a lot to the loss."

Maybe:
```
W1[5,12]
```
was responsible for a large part of the mistake.

Maybe:
```
W2[18,3]
```
barely mattered 

Backpropagation measures how much each weight influences the loss. 
This "influence" is called the gradient.

#### 3. Gradient

The gradient literally asks:
"If I tweak this exact weight, by how much the loss will change"

It will look at the weight, maybe raise it by a bit and then monitor:

```
Did the weight went:

Up?
Down?
Stayed the same?
```

Because if the loss went up, then it will make the weight smaller instead of bigger.

But as I said before, we will use calculus. But.. how will it help us???  Why would we use it?? We will use this:
$$\frac{\partial{L}}{\partial{W}}$$

We are going to read it as:
"The partial derivative of L with respect to W"

But in plain english it sounds like:
"How does the Loss (L) change if I slightly change the Weight (W)?"

So if its result is positive. It means increasing the weight increases our mistake. We need to lower the weight.

If the result is negative. It means increasing the weight decreases our mistake. We need to increase the weights.

The one that will help us finding the culprit through this 21k weights it's the Chain Rule.

Yup, the famous chain rule, so let us start.

#### 4. Chain Rule

Imagine that Airi has a piano concert tomorrow...
But will she get an S-Rank? 

Obviously that doesn't depend on a single factor, it depends on more factors, as:

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

What we can notice?
We can clearly notice that each step depends on the previous, for example:

```
Hours Slept = 3
```

Then everything will go wrong:
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

So now if we ask is:
"How much the sleep did affect the score?"
It didn't directly affect the score, it affected Energy.
Energy affected focus.
Focus affected quality.
Quality affected concert performance.
and the poor concert performance affected the final score.

They are a connected chain.
That is the chain rule, it works as:
"If A affects B, B affects C, C affects D... How much did A affect D.

The entire Neural Network is a gigantic chain.

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

Let's say we tweak $W_1$ , does the loss changes and that it? No. Because it becomes a entire domino:

It first changes
```
W1

↓

Z1
```

Then
```
Z1

↓

A1
```

Then
```
A1

↓

Z2
```

Then
```
Z2

↓

Softmax
```

Then
```
Softmax

↓

Loss
```

A big chain that changes the result.

---
Even the chain rule has a magical formula that we will talk about, imagine this:
```
A
↓
B
↓
C
```

How much does A changes C?

The formula is:
$$\frac{dC}{dA} = \frac{dC}{dB} \times \frac{dB}{dA}$$

The effect of A on C =
(Effect of B on C) $\times$ (Effect of A on B)

---
Let's use a clear example:

```
Sleep
↓
Energy
↓
Performance
```

This is a simplified version of Airi's needs for the good performance. Now we will think about this...

Every extra hour of sleep improve energy by +3

```
When Sleep is 1
Energy is 3.
```

Every point of energy improve performance by 2 points:

How much does sleep affect performance?

Easy:
$$\frac{dC}{dA} = \frac{dC}{dB} \times \frac{dB}{dA}$$
is going to become:
$$\frac{dC}{dA} = \frac{2}{1} \times \frac{3}{1}$$
$$\frac{dC}{dA} = 6$$
Every hour of sleep improves performance by 6 points.

But why we say that it goes backwards? We say that because we firstly need to pass through all the forward pass, compute the loss, and only then we can start with the backpropagation.
Because when we use the chain rule we will start to calculate from the last step and slowly reach the $W_1$. 

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
$$\frac{\partial \text{Loss}}{\partial W_1} = \frac{\partial \text{Loss}}{\partial A_2} \times \frac{\partial A_2}{\partial Z_2} \times \frac{\partial Z_2}{\partial A_1} \times \frac{\partial A_1}{\partial Z_1} \times \frac{\partial Z_1}{\partial W_1}$$
See? We start calculating from the last step of: $\frac{\partial \text{Loss}}{\partial A_2}$ → $....$ → $\frac{\partial Z_1}{\partial W_1}$

That's why it's called backpropagation:
- Back = we move from the loss back toward the inputs.
- Propagation = we pass information about the error backward through the network.

We will return to this step after a bit, but before this, we will learn a trick that will help us!

#### 5. Computational graph

So now imagine we are building an AI that predicts how likely an user will like the new released piano piece.

Today's user is Airi.

Our AI will receive many features as:

```
User Embedding
Recent Listening Time
Number of Piano Songs Played
Average Completion Rate
Similarity to Previous Songs
Time Since Last Session
... and many more
```

But for simplicity I will use just 2

```
Listening Time

Similarity Score
```

And make them equal to:
```
Listening Time = 8

Similarity Score = 0.9
```

Let's give to Listening time a weight of 0.7 and a bias of 2

So now we got:
```
8 × 0.7 = 5.6

5.6 + 2 = 7.6
```

But this time instead of just writing on python:
```python
z = X * W + b
```
 We will split the operation.

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

Noticed something? 
The computer didn't do everything in a step and give the output, it did small steps and give us the output.

Let's say the weight of similarity is of 5.

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

It went like:

```
Listening Time = 8 ──×0.7──┐
                           │
                           ▼
Similarity = 0.9 ──×5────► (+) ──► +2 ──► z = 12.1
```

Let's expand a bit the idea:

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

So now our graph became:

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

It is literally our neural network.
Later on you will understand why this is useful.

#### 6. Numpy implementation.

Now we will do numpy implementation, because it will be useful to understand!
Okay, now Airi is analyzing transaction data (Her own transaction data, since she thinks which to delete and don't get caught) with three features to output three distinct financial predictions.

---
- 3 Features (Inputs):
    1. $x_1$: Account Balance (in thousands)
    2. $x_2$: Daily Transaction Count
    3. $x_3$: Risk Score (0 to 1)
        
- 3 Neurons (Outputs):
    1. Neuron 1: Fraud Likelihood
    2. Neuron 2: Loan Approval Score
    3. Neuron 3: VIP Status Probability
---

$$X = \begin{bmatrix} 2.5 & 12 & 0.1 \\ 0.8 & 45 & 0.9 \\ 15.0 & 2 & 0.05 \\ 1.2 & 18 & 0.4 \\ 5.4 & 8 & 0.2 \end{bmatrix}$$

We can notice that first day, Airi had an account balance of 2500 yen, she made 12 transactions, and had a low risk (10%)
In the second day she had just 800 yen, 45 transactions, and a high risk (90%)... Poor soul.

The weights:
$$W = \begin{bmatrix} \mathbf{-0.2} & 0.8 & 1.5 \\ \mathbf{0.5} & -0.3 & 0.1 \\ \mathbf{1.2} & -0.9 & -0.4 \end{bmatrix}$$
The bias:
$$b = \begin{bmatrix} -0.5 & 0.1 & -0.2 \end{bmatrix}$$
---

We break the columns to pieces:
1 columns (Neuron 1 - Fraud Likelihood)
bias = -0.5 (meaning: "I assume everything is fine unless proven otherwise")
$$\begin{bmatrix} \mathbf{-0.2} \\ \mathbf{0.5} \\ \mathbf{1.2} \end{bmatrix}$$

- Row 1 (Account Balance Weight = -0.2): Having more money makes her seem less suspicious.
    
- Row 2 (Transaction Count Weight = 0.5): Making a lot of transactions makes her look a tad suspicious.
    
- Row 3 (Risk Score Weight = 1.2): Having a high security risk score makes her look like a villain!

We will use day 2 as an example:
- The Math: $(0.8 \times -0.2) + (45 \times 0.5) + (0.9 \times 1.2) - 0.5$
- The Result: $-0.16 + 22.5 + 1.08 - 0.5 = 22.92$
- What Neuron 1 says: "A score of 22.92??? Freeze the account! This is definitely fraud!"
---

2 column (Neuron 2 - Loan Approval Score)
bias = 0.1 (meaning: "I'm slightly generous by default")
$$\begin{bmatrix} \mathbf{0.8} \\ \mathbf{-0.3} \\ \mathbf{-0.9} \end{bmatrix}$$
- Row 1 (Balance Weight = 0.8): Having more money makes her look incredibly cute and innocent, that why her chances of getting the loan approved rise.
    
- Row 2 (Transaction Weight = -0.3): Making too many daily transactions makes her look financially unstable (Poor Airi), dragging her score down.
    
- Row 3 (Risk Weight = -0.9): Having a high risk score completely ruins her chances—the bank manager wants nothing to do with her! (Poor soul, once again)

We will use day 2 as example (again):
- The Math: $(0.8 \times 0.8) + (45 \times -0.3) + (0.9 \times -0.9) + 0.1$
- The Result: $0.64 - 13.5 - 0.81 + 0.1 = -13.57$
- What Neuron 2 says: "Absolutely not. She is completely broke, spending wildly! Nahhh, NEGATE."

---
3 columns  (Neuron 3 - VIP Status Probability)
bias = -0.2 (meaning: "You aren't a VIP unless you prove it")
$$\begin{bmatrix} \mathbf{1.5} \\ \mathbf{0.1} \\ \mathbf{-0.4} \end{bmatrix}$$

- Row 1 (Balance Weight = 1.5): Having a huge account balance is the ultimate ticket to the VIP club! It dominates everything else.
    
- Row 2 (Transaction Weight = 0.1): Swiping her card often helps her look like an active premium customer, but only by a tiny bit.
    
- Row 3 (Risk Weight = -0.4): Having a bad security rating hurts her VIP chances because the bank doesn't want trouble in their exclusive club

As usual, we will use day 2 as example:
Airi is broke, running around making transactions, with a bad risk score.

- **The Math:** $(0.8 \times 1.5) + (45 \times 0.1) + (0.9 \times -0.4) - 0.2$
- **The Result:** $1.2 + 4.5 - 0.36 - 0.2 = 5.14$
- What neuron 3 says: "Wow, despite being broke she raised her score enough to be a positive VIP client, well... if we use ReLu or Softmax everything will normalize"


Now Y will tell everything that is true or no about her:
$$Y = \begin{bmatrix} \mathbf{0} & \mathbf{1} & \mathbf{0} \\ 1 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$

First day she:
0 - Wasn't a fraud
1 - Got her loan approved
0 - Didn't qualify as a VIP

Second day she (Broke version):
1 - She was a fraud
0 - Didn't get her loan approved
0 - Didn't qualify as a VIP

Now we will use it on python (Full Numpy! Still no PyTorch!):
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

# Targets (Y): the thruth about her own transaction data
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
	# I will explain how we got this idea here, so don't panic (yet)
	
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

Before procceding and explaining some concpets I am going to derive your existing soul, so be ready...

---
#### SECRET DLC - Mini preparation:

##### 1. e

We saw this number thousands of times... yet we don't understand what it means...

Imagine this.
The first year you have
```
100 dolars
```

Year by year it goes as:
```
100
↓
200
↓
400
↓
800
```

This become:
$$2^x$$

What if it would triple?
```
100
↓
300
↓
900
```

So we can understand that different bases grow differently.

But now imagine something continuous:

```
Population.

Radioactivity.

Bacteria.

Neural networks.

Compound interest.
```

This doesn't like growing "Once a year". This is continuous, it grows every time instant. And this magical base is:
$$e = 2.718281828...$$
It is the only exponent where
growth rate = current value

For example...
Let's say we have a firefly that grows depending of its grams:

- If the firefly weighs 1 gram, it is growing at a speed of 1 gram per second.
- If it eats and grows to weigh 5 grams, its growth rate instantly speeds up to 5 grams per second.
- If it grows to a whopping 100 grams, its growth rate rockets to 100 grams per second.

so now we realize that 
growth rate = current value.

So the derivative of this beautiful base is...
$$\frac{\partial{}}{\partial{x}}e^x = e^x$$
this remains always the same:
$$\frac{\partial{}}{\partial{x}}e^{5x} = 5e^{5x}$$
$$\frac{\partial{}}{\partial{x}}7e^{2x} = 14e^{2x}$$
WHY???
Because there is something inside (Parentheses - kinky)
$$e^{(inside)}$$
So in this case we find the derivative of the inside (CHAIN RULE) and bring it down:
$$e^{(12x^2)}$$
We look at what is inside:
$$\frac{\partial{}}{\partial{x}}12x^2 = 24x$$
so now we bring it in front:
$$\frac{\partial{}}{\partial{x}}e^{(12x^2)} = 24xe^{(12x^2)}$$

But don't get fooled, because this is still a constant! Because it stills equals 2.182818284590...
So if it has do variable next to it, it will become 0.
$$\frac{\partial{}}{\partial{x}}e^3 = 0$$
$$\frac{\partial{}}{\partial{x}}e = 0$$
$$\frac{\partial{}}{\partial{x}}7e = 0$$

Tricky ideas:
$$\frac{\partial{}}{\partial{x}}\frac{e^{2x} + 12}{e^{y + 12}} = ?$$

We will do a simple algebraic trick. (e.g... $\frac{1}{x^2} = x^-2$):
$$\frac{e^{2x} + 12}{e^{y + 12}} = \big(e^{2x} + 12\big) \cdot e^{-(y + 12)}$$
now we find the 'with respect to x':

$$\frac{\partial{}}{\partial{x}}[\big(e^{2x} + 12\big) \cdot e^{-(y + 12)}] =(\frac{\partial{}}{\partial{x}}\big(e^{2x} + 12\big)) \cdot e^{-(y + 12)}$$

We can see that '+ 12' is just a constant, so it will equal to 0.
We look at the inside of $e^{2x}$ (Which equals to)
result:
$$\frac{\partial}{\partial x} = 2e^{2x} \cdot e^{-(y+12)}$$
Now we find the 'with respect to y':
$$\frac{\partial}{\partial y} \Big[ \big(e^{2x} + 12\big) \cdot e^{-(y + 12)} \Big] = \big(e^{2x} + 12\big) \cdot \left( \frac{\partial}{\partial y} e^{-(y + 12)} \right)$$
Everything stays the same from the outside, beside from the inside, because we have there 
-(y + 12), which derivative will equal just to -1. so we bring it in the most front part:
$$\frac{\partial}{\partial y} = -\big(e^{2x} + 12\big) \cdot e^{-(y+12)}$$

Now you may have a question...
WHY DIDN'T WE TREAT THE SECND PART AS A CONSTANT EVEN IF THERE IS MULTIPLICATION?

Because one of the rules of calculus says:
if you have a constant  multiplied by a variable expression, you simply leave the constant alone and differentiate the variable part. For example:
$$\frac{\partial}{\partial{x}} \big( 5 \cdot x^2 \big) = 5 \cdot \left( \frac{\partial}{\partial{x}} x^2 \right) = 5 \cdot 2x = 10x$$

Long story short. We traumatize with logarithms, because we are masochists.

##### 2. log, ln

The most important part will clearly be 
$$ln(x)$$
Because it is the undo of:
$$e^x$$

```
eˣ
↓
ln
↓
x
```

The formula of $ln(x)$:
$${\frac{\partial}{\partial{x}}\ln(x) = \frac{1}{x}}$$
Since $e^x$ grows and grows, $ln(x)$ must flatten more and more, that why the result is $\frac{1}x$

Some examples would be:
$$5\ln(x) = 5 \times \frac{1}x = \frac{5}x$$
$$12\ln(x) = 12 \times \frac{1}x = \frac{12}x$$
---
What will happen if we have something inside...? (Again, parentheses):
$$\ln(5x)$$
In the outside we will get:
$$\frac{1}{5x}$$
now we check inside the parentheses:
$$\frac{\partial}{\partial{x}}5x = 5$$
now we have:
$$5 \times \frac{1}{5x} = \frac{1}x$$
We simplified the 5s. 

---
$$\ln(x^2 + 1)$$
- Outside derivative: $\frac{1}{x^2 + 1}$
- Inside derivative: $2x$

$$\frac{2x}{x^2 + 1}$$

Tricky ideas:
$$\ln(15) = \frac{1}{15} - \text{That a costant, so it will equal to 0.}$$
---
What about exponential?

$$\frac{\partial}{\partial{x}}2^x = 2^x\ln(2)$$
their general formula is literally:
General formula
$${ \frac{\partial}{\partial{x}}a^x = a^x\ln(a) }$$

What if a = $e$
$$\ln(e) = 1$$
So:
$$e^x \times 1 = e^x$$
---
What about logarithms? 
Suppose
$$log_2(x)$$

Derivative
$$\frac1{x\ln2}​​$$

The general formula:
$${\frac{\partial}{\partial{x}}\log_a(x) = \frac{1}{x \ln(a)}}$$
---
Main rule we will use daily:

- **Exponential + chain rule**
$$\frac{d}{dx}e^{f(x)} = e^{f(x)} \cdot f'(x)$$
Example:
$$\frac{d}{dx}e^{5x} = e^{5x} \cdot 5 = 5e^{5x}$$
Example 2:
$$\frac{d}{dx}e^{12x^2} = e^{12x^2} \cdot 24x = 24xe^{12x^2}$$
- **Natural Logarithm + Chain Rule**
$$\frac{d}{dx}\ln(f(x)) = \frac{f'(x)}{f(x)}$$
Example:
$$\frac{d}{dx}\ln(5x) = \frac{5}{5x} = \frac{1}{x}$$
Example 2:
$$\frac{d}{dx}\ln(x^2 + 1) = \frac{2x}{x^2 + 1}$$
- **Sigmoid Derivative**
$$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$
Example:
If the current activation of a neuron is $\sigma(z) = 0.8$:
The rate of change at this point is:
$$\sigma'(z) = 0.8 \cdot (1 - 0.8) = 0.8 \cdot 0.2 = 0.16$$
Example 2:
$$\sigma'(z) = 0.99 \cdot (1 - 0.99) = 0.99 \cdot 0.01 = 0.0099$$
And now we make a comeback...

---
### 5. ReLu

Yes, I know, we learned them, yet we will understand the whole logic behind them. Sooo... we need to be sad, yet that how life work.

#### 1. ReLu and Leaky ReLu

Okay, so we already know the ReLu idea, it will help us from vanishing gradients. Because back in times, engineers spammed Tanh and Sigmoid after every inconvenience, but as we already explored...

1. Saturation
They have a saturation problem, because when we do the backpropagation, each step multiplies by another derivative, and since the maximum value from the sigmoid derivative is 0.25...
$$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$
It will just multiply by small numbers, like:
```
0.2
×0.1
×0.05
×0.02
×0.01
...
```

and soon our gradient will look like:
```
Gradient
↓
0.0000000001
```

and so, our early layer stops learning at all (improvement is abysmally small)
and that how we get a vanishing gradient problem.


2. Computation

As we already know, sigmoid requires exponentials and division, and this hits hard the GPU, so computing millions of exponentials was expensive.

Its derivative is almost embarrassingly simple.

$$\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0, & x < 0 \end{cases}$$
It is simply:
$$1$$

But why is it important?

Because with Sigmoid we went like this:
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

meanwhile, with ReLu:
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

Why is it useful?

Suppose your network has 1000 neurons. With sigmoid, every neuron is doing work. Even if its output is 0.00001 it's still involved.

Meanwhile ReLu doesn't let this to happen, it will make many neuron output the value of 0.
Those neurons don't participate in the forward computation for that input. The network becomes sparse.

So instead of getting 
```
1000 active neurons
```

we will get:
```
470 neurons

530 inactive
```

So if we train a model to recognize a image of a cat, a neuron may look for:

```
whiskers
```

another neuron may look at:

```
tail
```

another may look at:

```
Blue sky
```

If the image has no sky, why should the "blue sky" neuron influence the prediction?
ReLU simply turns it off.
So it let just the useful detectives.
Since ReLu is a piecewise function, its derivative on python looks like:

```python
(Z > 0).astype(float)
```

But anyway... everything has a price...

if the z < 0, our derivative turns cutely into 0... and If a neuron keeps producing negative pre-activations for every training example, its weights stop changing. That's the dead ReLU problem.

```python:
import numpy as np

# Columns: [Income, Hours, Purchases, Age]
X = np.array(
[[4.0, 8.0, 2.0, 30.0], # Customer #1
[10.0, 12.0, 4.0, 28.0], # Customer #2 (Very rich)
[7.0, 5.0, 1.0, 35.0], # Customer #3
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

That what a dead ReLu looks like... so what do we do? Do we use tanh? Maybe, but we don't really wanna stick with oldies, so we will use the Leaky ReLu, that is literally:
$$\text{ReLU}'(x) = \begin{cases} 1, & x > 0 \\ 0.01, & x < 0 \end{cases}$$

So we don't make all the outputs 0, we use the leaky ReLu:

```python
import numpy as np

# 10 Customers with 6 features:
# [Income (k$), Hours in app, Purchases, Age, Device (1=iOS, 0=Android), Referral (1=Yes, 0=No)]
X = np.array([
[4.0, 8.0, 2.0, 30.0, 1.0, 0.0], # Customer 1
[10.0, 12.0, 4.0, 28.0, 1.0, 1.0], # Customer 2 (Very rich)
[7.0, 5.0, 1.0, 35.0, 0.0, 0.0], # Customer 3
[12.0, 20.0, 8.0, 25.0, 1.0, 1.0], # Customer 4 (High value)
[3.0, 1.0, 0.0, 19.0, 0.0, 0.0], # Customer 5
[8.5, 15.0, 5.0, 42.0, 1.0, 0.0], # Customer 6
[2.5, 3.0, 0.0, 50.0, 0.0, 0.0], # Customer 7
[15.0, 2.0, 1.0, 38.0, 1.0, 1.0], # Customer 8
[6.0, 10.0, 3.0, 29.0, 0.0, 1.0], # Customer 9
[11.0, 18.0, 6.0, 31.0, 1.0, 0.0] # Customer 10
])

# Ground Truth: 1.0 = Subscribed, 0.0 = Not Subscribed
y = np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0]])
  
# SInce we have 6 collumns we need 6 neurons
W = np.array([[0.1], [0.15], [0.3], [-0.05], [0.2], [0.1]])
b = -3.5
lr = 0.001
  
for epoch in range(2000):
	Z = X @ W + b
	A = np.where(Z > 0, Z , Z * 0.01)
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

Remeber, this is just a trial, because in real production you would mix sigmoid with BCE, not ReLu with MSE.
```

# Progetto finale

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
  
W = np.zeros((6, 4)) # We do 6 rows, since we have 6 features
b = np.array([[0.0, 0.0, 0.0, 0.0]]) # A bias per neuron
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

The loss is pretty normal, but we started from full 0. So that was expected, beside we have just a few users.
```

Another

```python
import numpy as np

# Set seed for identical initializations
np.random.seed(9)
  
# 10 Days with 6 features:
# [Headpats, Hugs, Kisses, Smiles, Minutes Together, Gifts]
X = np.array([
[15.0, 6.0, 3.0, 20.0, 480.0, 4.0], # Day 1
[2.0, 0.0, 0.0, 1.0, 50.0, 0.0], # Day 2
[8.0, 4.0, 1.0, 12.0, 240.0, 2.0], # Day 3
[25.0, 12.0, 8.0, 35.0, 720.0, 6.0], # Day 4
[0.0, 0.0, 0.0, 0.0, 15.0, 0.0], # Day 5
[10.0, 5.0, 2.0, 15.0, 300.0, 1.0], # Day 6
[4.0, 1.0, 0.0, 5.0, 90.0, 0.0], # Day 7
[18.0, 9.0, 4.0, 22.0, 540.0, 5.0], # Day 8
[1.0, 0.0, 0.0, 2.0, 40.0, 0.0], # Day 9
[14.0, 7.0, 2.0, 18.0, 420.0, 3.0], # Day 10
], dtype=np.float64)

# One-Hot Encoded Targets: [Yes, Maybe, No]
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

# SOFTMAX
Z = Z3 - np.max(Z3, axis=1, keepdims=True) # Keepdims will just
exp = np.exp(Z)
A3 = exp/(np.sum(exp, axis=1, keepdims=True))

# Categorical Cross Entropy
A3_clipped = np.clip(A3, epsilon, 1 - epsilon)
loss = -np.mean(np.sum(Y * np.log(A3_clipped), axis=1))

# The Backpropagation
m = X.shape[0]
dZ3 = (A3 - Y) / m
db3 = np.sum(dZ3, axis=0, keepdims=True)
dW3 = A2.T @ dZ3
  
dA2 = dZ3 @ W3.T
dZ2 = np.where(Z2 > 0, dA2, dA2 * 0.01)
db2 = np.sum(dZ2, axis=0, keepdims=True)
dW2 = A1.T @ dZ2

dA1 = dZ2 @ W2.T
dZ1 = np.where(Z1 > 0, dA1, dA1* 0.01)
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
Loops 500, loss: 1.0101906211147633
Loops 1000, loss: 0.21910083921119955
Loops 1500, loss: 0.12007220523774306
Loops 2000, loss: 0.06542320199081539
Current loss: 0.03891651681436877
Day 1:
Yes   :  0.9885
Maybe :  0.0115
No    :  0.0000

Day 2:
Yes   :  0.0003
Maybe :  0.0186
No    :  0.9811

Day 3:
Yes   :  0.0195
Maybe :  0.9661
No    :  0.0144

Day 4:
Yes   :  1.0000
Maybe :  0.0000
No    :  0.0000

Day 5:
Yes   :  0.0004
Maybe :  0.0185
No    :  0.9811

Day 6:
Yes   :  0.1395
Maybe :  0.8598
No    :  0.0007

Day 7:
Yes   :  0.0004
Maybe :  0.0429
No    :  0.9567

Day 8:
Yes   :  0.9986
Maybe :  0.0014
No    :  0.0000

Day 9:
Yes   :  0.0003
Maybe :  0.0176
No    :  0.9821

Day 10:
Yes   :  0.9138
Maybe :  0.0862
No    :  0.0000
"""

You need to be a good at derivatives to understand the backpropagation, yet remember, almost nobody, finds dL_dA3/A2... because we can instantly do dL/Z3, which will not force us to do a Jacobian matrix... which is scary, trust me.
```

And like this... we go toward a really important step... PyTorch, which will actually help us to use this ideas in fewer lines and make us look like starved ML engineers that want kudos (We do, that why we will master it!).

So let's start with PyTorch before the geometric hell we will experience...

# ML BASICS

Hello, explorer.
Idk from where you started, maybe this is your first topic? MAYBE?? Because I didn't mark them, so I don't even know from where to start, yet I'll start from here... and I have some lovely guesses how you ended up:

The most probable one:

- You are an interviewer, and you’re checking this out to see if it’s really worth hiring people in the big 2026/2027 era. 
- You like the idea of automating half of everything and being hated by people like a tax collector because you automated their jobs. No excuses like, _"I'm just making jobs easier!"_
- Saw a video on tiktok (I don't even post videos, maybe I'll be so desperate to start in the future)

Also highly possible:

- You love IT and Math, so you thought there was no better field to connect the two. (And yeah, that’s exactly why I’ll teach you to write hundreds of lines of code using pure math, only to later exchange all that effort for the sad reality of `scikit-learn` and `pyspark.ml`. But hey, at least you’ll actually understand bugs instead of hitting a `ValueError: Found input variables with inconsistent numbers of samples` and burning 2.5 million tokens on Claude hoping it fixes it for you.)

- You found a secure and satisfying career path, because the further we move into the future, the more [people will "boo" us when we talk about AI](https://www.youtube.com/watch?v=tNH43a1EI7s&t=2s). But just remember one thing: they hate us 'cause they ain't us. So keep studying. 

Too rare to be likely:

- You just like learning.

Sadly no vibecoding... we will mentally tackle demons, so we will start. 

This first month will be all about ML basics and even some math. Even if I suppose you know how to use chain rules, partial derivatives, and gradient descent, I will still explain them, so maybe they will tattoo in your brain. 

We will start in a easy way.... Linear regression, because life is easy:

## 1. Linear Regression

Why do we use linear regression? why do we even need it?? 
Luckily for you, I will not say something as "linear regression is a fundamental mathematical tool to establish a direct, quantitative...", because who will even understand it beside a senior engineer and AI?

So firstly, I will give you an example, tell you why do we use it, and in the end compute it on python.

But before that!
we will use numpy for many stuff, so firstly I'll tell you a basic idea of numpy.
You will see many times `np.array`, what does it mean? Think about it as a python list, but made only for math and operations like this. And actually you will see many times:

`np.array([1, 2, 3, 4])` < --- this is a 1D vector, nothing more, just a 1D vector:

$$\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}$$

`np.array([[1, 2], [3, 4]])` < --- this is already 2D:

$$ \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

and the more you add the more columns or rows you get:

`np.array([[1, 2, 3], [4, 5, 6]])` < --- this is still 2D:

$$ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} $$
and the last:

`np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])` < --- still a 2D, because to do a 3D we will need to make it a tensor, yet we don't work with them (yet):

$$ \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} $$

Example 1:

Imagine this... You are driving home from work, but you have many red lights in front, so you want to predict them, so you take a list with all the information about the red lights and see:

1 red light -> 6 minutes  
2 red lights -> 11 minutes  
3 red lights -> 12 minutes  
4 red lights -> 17 minutes  
5 red lights -> 20 minutes  
6 red lights -> 23 minutes  

You stare at the screen and think: "That not even close how a real red light works!" - maybe, but we care less about it right now, the only thing we care about is to predict how much we will stay when we have to wait at red lights of which we don't have the information about, as for: 7 red lights? 10 red lights? 25 red lights?

That why we will use linear regression, to predict the next red lights (That how google maps actually work, but google maps looks at many factors, of which one of them is this).
But how we can predict the red lights?

We will firstly write our `x` and `y` on python:

```python
import numpy as np

# We will make a numpy array with 7 rows and 1 column
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])
```

The matrix 'X' will always be written in more rows, later I will explain what to do if there will be more features.

![[Figure_1 2 1.png]]

Now that we have our X and y, we will need to give a first guess... which will be the line that passes through the dots... `w` and `b` will help with this, even if we will place them to 0. and even the lr (learning rate. This will be placed on 0.01 - most of the times, but when the data is small you can raise it):

```python

import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])

w = 0
b = 0
lr = 0.01 
# We will make w equal to 0 because X doesn't have more features (Don't worry for now)
```

now we will make the learning algorithm (using gradient descent). The algorithm will help us understand the exact value that w has to be and b too.

```python
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])

w = 0  # weight
b = 0  # bias
lr = 0.01  # learning rate

for epoch in range(1500):
    # We will do 1500 loops, since the gradient descent tweaks a bit the cute w        value. And epoch is just a fancy place holder, so no worry, write even 'i'
    prediction = X * w + b    # this is our beautiful linear expression
    error = prediction - y    # we will beautifully see by how far the model is                                  from guessing  
    loss = np.mean(error**2)  # this is the loss of the model. I will explain                                    all better
    
    nabla_w = (1/len(X)) * np.mean(error * X)
    nabla_b = (1/len(X)) * np.mean(error)
    
    w = w - nabla_w * lr
    b = b - nabla_b * lr
    
    # The code below means: every 300 loops show the result
    if i % 300 == 0:
		print(f"Step {i}: Loss = {loss} | w = {w} | b = {b}")
		print("")

print("----FINAL RESULT----")
print("")
print(f"The loss of the model is of: {loss}")
print(f"The weight of the model is of: {w}")
print(f"The bias of the model is of: {b}")
```

Before printing the result i'll explain the jargon:

First, let's understand `np.mean()` before anything else, cuz you gonna see it twice in here and if you don't get it, the rest just gonna look like chinese. `np.mean()` literally just take a bunch of numbers and give you back the average, that's it, nothing more. So if you do: 
`np.mean([2, 4, 6])`
it just add them up (2+4+6 = 12) and divide by how many numbers there was (3), so you get `4`. That's the whole point of np.mean.

Now `for epoch in range(1500):` - we do this so we make the model loop 1500 times and make the loss and guess better (don't put 10000 or smth as this, because  the model will start tweaking the one quadrillionth, so just don't do it.)

`prediction = X * w + b` - this the actual line equation we trying to build, remember from school `y = mx + b`? Same exact thing here, just written with different letter names. At the start `w = 0` and `b = 0`, so at loop 1, prediction for EVERY single X value is literally just 0, cuz anything times 0 plus 0 is still 0. Bad prediction, I know, but it's the starting point, gotta start somewhere, if you are too annoyed with the '0', we will learn much later the Xavier (Glorot) Initialization and the Kaiming Initialization (boo hoo to the poor soul who can't endure easy starts)

`error = prediction - y` - this just check "how far off was my guess from the real answer". If `prediction` is 0 and real `y` is 6, error is `0 - 6 = -6`, meaning we way under-guessed. This get done for all 6 rows at once, cuz numpy is smart like that, it don't make you loop manually, it does it on the whole array in one shot.

`loss = np.mean(error**2)` - here we square every single error first (so negative errors become positive, and big errors get punished way more than small ones), then take the average of all them squared errors using `np.mean` like we just learned. This number, `loss`, is basically a single score that tell you "how bad is my model doing right now", the smaller this number, the better the model guessing, if you will see answers as loss = 500, then maybe it is time to check the data or prediction... because I could guess better without ever seeing the numbers and no information. This type of loss is called the 'Mean Square Error (MSE)'

Now the scary part, `nabla_w` and `nabla_b` - don't panic, "nabla" just a fancy greek letter people use to mean "gradient", which just mean "which direction, and how much, should I nudge my `w` and `b` to make the loss smaller". Think of it like you standing on a hill in the fog, you can't see the bottom, but you can feel which way the ground slope down under your feet, so you take a small step that direction. That's literally what these two lines do.

`nabla_w = (1/len(X)) * np.mean(error * X)` - this multiply the error by the X value for each row (so bigger X values get blamed more if there's error), then average all that using `np.mean` again, then we divide once more by `len(X)` (how many rows we got, which is 6 here). This whole thing just tell us "how much and which direction we gotta adjust `w`".

`nabla_b = (1/len(X)) * np.mean(error)` - same idea, but simpler, since bias `b` don't get multiplied by X, we just average the error directly (and divide by len(X) again) but that's just how the formula written here, no big deal.

`w = w - nabla_w * lr` and `b = b - nabla_b * lr` - and here's where the actual "learning" happens. We take our current `w`, and nudge it a tiny bit in the direction that reduce the loss, and how big that nudge is depends on `lr` (learning rate, remember, set to 0.01). If `lr` was bigger, like 0.5, the steps would be huge and reckless (imagine this as taking small steps to get to the bottom of the hill... putting lr = 0.5 you are running down the hill or directly jumping from it). If `lr` too small, like 0.00001 it would be an eternal waiting (imagine waiting in a place and thw wind to move you slowly... way too slowly, till you don't get to the bottom). 0.01 usually a nice middle ground for small data like this.

Then this whole block repeat itself 1500 times, each time getting a slightly better `w` and `b`, slightly smaller loss, like the model slowly "waking up" and realizing "oh wait, the pattern is actually more like +4ish minutes per red light, plus some starting offset".

```python
"""
Output:

Step 0: Loss = 300.5 | w = 0.11222222222222222 | b = 0.02638888888888889

Step 300: Loss = 1.1658176891773773 | w = 4.201494306378685 | b = 1.0320377806554515

Step 600: Loss = 1.15772799442481 | w = 4.192532609157757 | b = 1.0760211827106234

Step 900: Loss = 1.151010738627913 | w = 4.183217288762125 | b = 1.1159036224474193

Step 1200: Loss = 1.1454140224030314 | w = 4.174714016447933 | b = 1.1523078196388303

----FINAL RESULT----

The loss of the model is of: 1.140765095584983
The weight of the model is of: 4.166977031120705
The bias of the model is of: 1.1854313896144373
"""
```

The loss is not really big, yet of. That is why we will run the same code but change just `w` and `b`

```python
w = 4.166977031120705
b = 1.1854313896144373

"""
Output:

Step 0: Loss = 1.1407509228485735 | w = 4.166952317505737 | b = 1.1855371935057646

Step 300: Loss = 1.1368656986240355 | w = 4.159867519156049 | b = 1.2158686215714196

Step 600: Loss = 1.133628588393243 | w = 4.153400588651619 | b = 1.243554834744929

Step 900: Loss = 1.1309314769678283 | w = 4.147497641512551 | b = 1.268826523180763

Step 1200: Loss = 1.1286842844033285 | w = 4.14210949254408 | b = 1.2918942584609654

----FINAL RESULT----

The loss of the model is of: 1.1268176480832708
The weight of the model is of: 4.1372069059387835
The bias of the model is of: 1.3128832048804213
"""
```

That how our model guessed: 

![[Figure_2.png]]

the loss fell by a small amount, but let us try and see how bad the model guesses are:

When there will be 4, 5, 6, 10 red lights, it will say:

- 4 red lights -> 17.86 minutes (Actual value in your dataset was 17)
- 5 red lights  -> 22.00 minutes (Actual value in your dataset was 20)
- 6 red lights -> 26.14 minutes (Actual value in your dataset was 23)
- 10 red lights -> 42.68 minutes (This is a brand new unseen value! - probably off, yet good try)

That how the linear regression works, another example could be... 
The taxi driver example...

we know just this:

2 km, 1 red light -> 9 minutes 
3 km, 2 red lights -> 14 minutes 
4 km, 1 red light -> 15 minutes 
5 km, 3 red lights -> 22 minutes 
6 km, 2 red lights -> 24 minutes 
8 km, 4 red lights -> 34 minutes

Here we can see the distance and the amount of red lights (for example the first dot (the one in the corner), it has 2 km and 1 red light, the last dot is clearly 8 km and 4 red lights.)
![[Figure_3 1.png]]

here we can see the total amount of time traveled by each point.
![[Figure_4.png]]


The one above were for learning purpose, it looks like this actually:

![[A new one.png]]
 
And as usual, we stare blankly at the screen, thinking that nobody will tell us when we will arrive home... since we still have 9 km and 5 red lights... we took our beautiful manga of "I love your Cruddy" and start reading it, but then we remember that the model can help us to predict it, so what do we do? Continue reading it, and after crying like babies, we start writing the algorithm:

```python
import numpy as np

X = np.array([[2, 1], [3, 2], [4, 1], [5, 3], [6, 2], [8, 4]])

y = np.array([[9], [14], [15], [22], [24], [34]])

w = np.zeros((2, 1))  # it is the same as writing 'w = np.array([0, 0])'. It is just you can always write w = np.zeros((the_total_ammount_of_features, 1))
# since we have two features; X = km and red lights, we will write np.zero((2,1))
b = 0
lr = 0.01

for epoch in range(1500):
	prediction = X @ w + b  # this time we do the matrix multiplication, because                               we have more features (this work for 10k+ features)
	error = prediction - y
	
	loss = np.mean(error**2)
	
	nabla_w = (1/len(X)) * (X.T @ error) # We usually do it for more                                                        features, so we get the right                                                    shape
	nabla_b = (1/len(X)) * np.sum(error)
	
	w = w - nabla_w * lr
	b = b - nabla_b * lr
	
	if epoch % 500 == 0:
		print(f"Loop {epoch}, loss: {loss}, w: {w}, b: {b}")
		print("")
		
print("----FINAL RESULT----")

print(f"The loss of the model is of: {loss}") 
print(f"The weight of the model is of: {w}")
print(f"The bias of the model is of: {b}")

"""
Output:

Loop 0, loss: 453.0, w: [[1.07666667]
 [0.50333333]], b: 0.19666666666666666

Loop 500, loss: 0.057043825490346554, w: [[3.28721594]
 [1.76879512]], b: 0.5077036270753326

Loop 1000, loss: 0.0551605963821002, w: [[3.27619854]
 [1.81110907]], b: 0.45998788483373343

----FINAL RESULT----
The loss of the model is of: 0.054914888850511386
The weight of the model is of: [[3.27602155]
 [1.82016599]]
The bias of the model is of: 0.4378494071196286
"""
```

the results are really good, so now we can try to run the code again after putting the new weight and bias:

```python
"""
Output:

Loop 0, loss: 0.05491469870828385, w: [[3.2760234 ]
 [1.82017369]], b: 0.437819644265099

Loop 500, loss: 0.0548663198652986, w: [[3.27701465]
 [1.82216066]], b: 0.4273310052033038

Loop 1000, loss: 0.05485553180380848, w: [[3.27771669]
 [1.82261747]], b: 0.42232725530473153

----FINAL RESULT----
The loss of the model is of: 0.054853062566064766
The weight of the model is of: [[3.2781008 ]
 [1.82273217]]
The bias of the model is of: 0.41993460958581014
"""
```

Our model predicted:
Learning purpose: 
![[Figure_5.png]]

reality:

![[A new one12.png]]

let's guess some minutes:

The model predicts:

- 2 km and 1 red light -> 8.80 minutes (Actual value in your dataset was 9)
- 5 km and 3 red lights -> 22.28 minutes (Actual value in your dataset was 22)
- 6 km and 2 red lights -> 23.73 minutes (Actual value in your dataset was 24)
- 8 km and 4 red lights -> 33.55 minutes (Actual value in your dataset was 34)
- 9 km and 5 red lights -> 39.92 minutes (This is a brand new unseen trip!)

That how the linear regression works, but sadly, In a real job, we usually use libraries such as `scikit-learn`, which already contain highly optimized implementations of Linear Regression. Instead of writing dozens of lines of code for Gradient Descent, calculating gradients, updating weights, and monitoring the loss ourselves, we simply create the model, train it with our data, and let the library handle all the mathematics behind the scenes.

We will do this:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# We use the same idea of the taxi driver
X = np.array([[2, 1], [3, 2], [4, 1], [5, 3], [6, 2], [8, 4]])

# Labels
y = np.array([9, 14, 15, 22, 24, 34])  # this is now a 1D vector, because scikit-learn is too much of a weakling, and will chock otherwise.

# Create the model
model = LinearRegression()

# Train (fit) the model
model.fit(X, y)

# Show what the model learned
print(f"Weights: {model.coef_}")  # coeficients = weight
print(f"Bias: {model.intercept_}") # intercepts = bias

# Make a prediction
prediction = model.predict([[9, 5]]) 

print(f"\nPrediction for 9 km and 5 red lights: {prediction[0]:.2f} minutes")

"""

Weights: [3.27848101 1.82278481]
Bias: 0.4177215189873458

Prediction for 9 km and 5 red lights: 39.04 minutes
"""
```


Since - I hope - you understood something out of this, we will learn the logistic regression. Because life is long, depends by how much you expect to live.

## 2. Logistic Regression

So far, we have used Linear Regression.
Its job was simple:
Predict any real number.

For example,
4 red lights  → 17 minutes
8 km taxi trip → 34 minutes
House size → $250,000

But now imagine a simple idea, you want to know:
Will the email be Spam/Not Spam?, will the patient have cancer - Yes/No?
We will need just two answers (Binary logic), a result that will tell the probability of it being a spam or not... and here will help us the Logistic Regression!

Why Logistic Regression when a Linear Regression model can predict the same:
No. Because the Linear Regression will answer with negative numbers and numbers above 1, you can't have negative possibilities or 101+% of something, that why we want a probability. Probabilities are locked by reality into a strict range:

- 0.00: 0% chance (Impossible)
- 0.50: 50% chance (A perfect coin flip) 
- 1.00: 100% chance (Guaranteed certainty)

Anything above 1 or below 0 breaks the math. 
That why we need a saver that will take the raw numbers of our linear equation and turn it into a probability, but who can do that?! **Blaise Pascal** and **Pierre de Fermat**? maybe, but today we will use the Sigmoid function.

Think of the Sigmoid function as a visual "S-curve" that acts as a trash compactor for numbers. No matter how massive or microscopic your input $z$ is, the output cannot escape the boundaries of 0 and 1:

![[image (10).png|545]]

The formula for it:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Massive Positive Scores:** If the model is incredibly confident (z = 5000), 
  $e^{-5000}$ becomes effectively 0. The formula becomes $1 / (1 + 0)$, which equals 1.00 (Guaranteed).
    
- **Massive Negative Scores:** If the model is completely convinced against it ($z = -500$),
  $e^{-(-500)}$ becomes an huge number. The formula becomes $1 / \text{infinity}$, which collapses down to 0.00 (Impossible).
    
- **The Dead Center:** If the model is completely unsure and outputs a raw score of exactly $z = 0$, $e^0 = 1$. The formula becomes $1 / (1 + 1)$, which is exactly 0.50.

in python it looks like:

```python
import numpy as np

def sigmoid(z):
	1/(1 + np.exp(-z))
```

That the main part, because the only thing that will change is even the loss function, which will not be the MSE anymore but the BCE (Binary Cross Entropy).

The formula of the BCE:

$$J(w, b) = - \frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

But don't get to scared, because you will never solve it by hand... only if all your machines are broke... a terminal work and you are left with a pen and notebook... but I calculated the change of this happening, at least 0. So no big troubles. We need to understand just that if you take the derivative of this entire messy function 'J(w, b)' with respect to the weights 'w' (You will just do some partial derivative), the calculus matches up with the Sigmoid function perfectly. All the straining logs and painful fractions completely cancel out, leaving you with just this for the gradient:

$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (A - Y)$$

It will simply take our predictions (we mark it usually as 'A'), subtracting the true labels (our beautiful y), and multiplying by your features ('X').
But we will compute it as:

```python
import numpy as np

def safe_log(y_true, p_pred):
	epsilon = 1e-15
	p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
	return -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
```

why do we write it as this?! Well, let me explain:

We do it to prevent our code to crash into a NaN (Not a Number) and Inf (Infinity)
epsilon = 0.000000000000001, so if that happens, we see immediately that there is something wrong.

Why np.clip? because we wrote:
`p_pred = np.clip(p_pred, epsilon, 1 - epsilon)` - this means that if the number is too small, np.clip will push it to the minimum value allowed, and if the number is too big, it will push the number to the maximum alowed as 0.9999... (99.999...%)

and the rest is just the formula, but used in python.

our graph will look like:

![[Figure_111.png|648]]

```python
import numpy as np

# 1. Setup Data: 2 features
# Feature columns: [Body Temperature, Cough Severity]
X = np.array([
[36.5, 2], # Patient 1 (Healthy)
[38.2, 7], # Patient 2 (Sick)
[36.8, 1], # Patient 3 (Healthy)
[39.1, 9], # Patient 4 (Sick)
[37.0, 3], # Patient 5 (Healthy)
[38.5, 6], # Patient 6 (Sick)
[36.2, 2], # Patient 7 (Healthy)
[37.8, 5], # Patient 8 (Sick)
[39.5, 8], # Patient 9 (Sick)
[36.7, 4] # Patient 10 (Healthy)
])

y_true = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [1], [0]])
# 1 = sick and 0 = healthy

w = np.zeros((2, 1))
b = 0
lr = 0.01


def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def safe_log(y_true, p_pred):
	epsilon = 1e-15
	p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
	return -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))

  
  

for epoch in range(1500):
	p = sigmoid(X @ w + b)
	error = p - y_true
	
	loss = safe_log(y_true, p)
	  
	nabla_w = (1 / len(X)) * (X.T @ error)
	nabla_b = (1 / len(X)) * np.sum(error)
	
	w = w - (nabla_w * lr)
	b = b - (nabla_b * lr)
	
  

print("--- Training Complete ---")
print("")
print(f"The current loss: {loss}")
print("")
print(f"Optimized weights:\n{w}")
print("")
print(f"Optimized bias: {b}\n")

"""
Output:

--- Training Complete ---

The current loss: 0.0831128739867651

Optimized weights:
[[-0.24324553]
 [ 2.03034046]]
 
Optimized bias: -0.04131919500891589

```

Now we update the weight and bias and see the result:

```python
"""
Output:

--- Training Complete ---

The current loss: 0.0624768838188333

Optimized weights:
[[-0.30891162]
 [ 2.57792074]]
 
Optimized bias: -0.0578637572773884
```

Model prediction:

![[Figure_112.png]]

What do we understand? We understand that all the dots that will be above the line are sick, while all the other bellow are healthy.

Tries:

Temperature = 36.5 °C, Cough Severity = 2 -> Healthy (Confidence: 2.18%)
Temperature = 38.4 °C, Cough Severity = 7 -> Sick (Confidence: 99.99%)
Temperature = 37.2 °C, Cough Severity = 4 -> Sick (Confidence: 92.34%)
Temperature = 39.0 °C, Cough Severity = 9 -> Sick (Confidence: 100.00%)
Temperature = 36.4 °C, Cough Severity = 1 -> Healthy (Confidence: 0.31%)

Now the industry level is incoming:

```python

import numpy as np

from sklearn.linear_model import LogisticRegression

X = np.array([
[36.5, 2], # Patient 1 (Healthy)
[38.2, 7], # Patient 2 (Sick)
[36.8, 1], # Patient 3 (Healthy)
[39.1, 9], # Patient 4 (Sick)
[37.0, 3], # Patient 5 (Healthy)
[38.5, 6], # Patient 6 (Sick)
[36.2, 2], # Patient 7 (Healthy)
[37.8, 5], # Patient 8 (Sick)
[39.5, 8], # Patient 9 (Sick)
[36.7, 4] # Patient 10 (Healthy)
])
  
# Labels: 1 = Sick, 0 = Healthy (Shape: 10 rows, 1 column)
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])

model = LogisticRegression()
model.fit(X, y_true)

print(f"Model weight is of: {model.coef_}")
print(f"Model bias is of: {model.intercept_}")

prediction = model.predict([[38.1, 6]])

print(f"prediction for a patient with 38.1 fever and 6 cough severity: {prediction[0]: .2f}")

"""
Output:

Model weight is of: [[0.62351466 1.04001144]]
Model bias is of: [-28.05834858]
prediction for a patient with 38.1 fever and 6 cough severity:  1.00
"""
```

We will continue with the decision tree, since we already understand the Logistic regression.
But as usual, the creator of the repo will complain about something as: Why do I have to learn?? I am sooo tired, every day, x/7 I learn, I don't even play games, but that fine, since I keep reading "I love your Cruddy" over and over.

## 3. Decision Tree

Now, here the idea goes much more into something more human readable and less algebra. The decision tree is a game of 20 questions in which we participate to create a tree of decisions that humans can read. 

Let's imagine this, your boss just said to you to make a something that are not a bunch of algorithms, but give the perfect answer to the questions they have. A doctor will never understand a logistic regression that says "Everything is cool and fine, just add to me the new values and make me predict the result."

If you spear our 10-patient dataset at a Decision Tree, it does not look for weights (w) or a bias (b), instead it will just follow the logic we gave to it, for example:

Is Body Temperature < 37.5°C?
	Yes: Classify as Healthy (Green).
	No: Go to the next question.

Is Cough Severity < 5?
	Yes: Classify as Healthy (Green).
	No: Classify as Sick (Red).


But before just computing it we have to understand it, so I'll give some examples:

| **Patient** | **Body Temperature** | **Status**  |
| ----------- | -------------------- | ----------- |
| A           | 36.5                 | 0 (Healthy) |
| B           | 37.0                 | 0 (Healthy) |
| C           | 38.0                 | 1 (Sick)    |
| D           | 39.0                 | 1 (Sick)    |

Let's look at this table, we have 2 healthy patients and 2 sick. 
Here we will see... how filthy (messy) is our dataset right now. By using the Gini impurity, which goes from 0 to 0.5. 

What is the Gini impurity?
Imagine this:
You close your eyes. You randomly grab one animal.

The question is...
"How difficult is it to correctly guess what I picked?"

That what exactly is the Gini impurity.

so it has a formula, the formula is literally:
$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
First, square each probability $p_{i}^{2}$ ,and add them together, then subtract that total from 1.
 (1 - sum)

Let's try it on the table we have:

$$p_i = \frac{n_i}{N}$$
which is just 
- $p_{i}$ is the probability of class \(i\).
- $n_{i}$ is the number of items belonging to class \(i\).
- $N$ is the total number of items in the group.


In our case we will check what we have:
we have 2 healthy patients and 2 sick patients.
- Healthy Count: 2
- Sick Count: 2
- Total Patients: 4

let us see what is the Gini of this list:
$$I_G(p) = 1 - ((\frac{1}2)^2 + (\frac{1}2)^2)) $$ $$I_G(p) = 1 - (0.25 + 0.25)$$$$I_G(p) = 1 - 0.5$$$$I_G(p) = 0.5$$
This the Gini of this list... it is of 0.5, which actually is the max of cruddiness of this list, that why we will help it.

Now we search, which would be our best x?
(for this:   x < body temperature), so we can split them, but how we find that exact number? It is easy!
Look at the feature (X) and find their best "threshold", i'll show you how:

| **Patient** | **Body Temperature** | **Status**  |
| ----------- | -------------------- | ----------- |
| A           | 36.5                 | 0 (Healthy) |
| B           | 37.0                 | 0 (Healthy) |
| C           | 38.0                 | 1 (Sick)    |
| D           | 39.0                 | 1 (Sick)    |
36.5 -> 37.0,  what is their midpoint?  (36.5 + 37.0)/2 = 36.75
37.0 -> 38.0 = 37.5
38.0 -> 39.0 = 38.5

so which we choose? The best strategy we use is deleting those in the same class as:

36.5 -> 37.0 (They are both healthy) -> 0
37.0 -> 38.0 (Different groups, we keep)
38.0 -> 39.0 = 38.5 (Same group, both sick) -> 0

so the perfect threshold is of 37.5

	is body temperature > 37.5
		No? We put them in healthy
		Yes? We put them in sick.

The structure is of:

Root:
	Internal nodes:
		Leaf nodes.

Something as:

![[image (11).png|347]]

That how a tree look.

So now that we understand how Gini impurity (cruddiness) works. I will give another example, with different data, so it will be easier to understand: 

| **Individual** | **Introversion Score (1-100)** | **Prefers Remote Work (0 = No, 1 = Yes)** |
| -------------- | ------------------------------ | ----------------------------------------- |
| A              | 22                             | 0                                         |
| B              | 35                             | 0                                         |
| C              | 48                             | 1                                         |
| D              | 50                             | 0                                         |
| E              | 63                             | 1                                         |
| F              | 71                             | 1                                         |
| G              | 78                             | 0                                         |
| H              | 85                             | 1                                         |
| I              | 92                             | 1                                         |
| J              | 97                             | 1                                         |
Now we will look at the classes... 
4 prefer remote work, and 6 don't .

We check the midpoints of different classes!

22 -> 35 (Same class (Both don't like remoteness)) = 0
35 -> 48 (Different classes) = 41.5
48 -> 50 (Different classes) = 49
50 -> 63 (Different classes) = 56.5
63 -> 71 (Same class (Both like remoteness)) = 0
71 -> 78 (Different class) = 74.5
78 -> 85 (Different class) = 81.5
the rest are of the same class.

now we have to take on by one...

	is Introversion Score > 41.5
		No? put them in the don't prefer remote work class
		Yes? put them in the prefer remote work class

Left we have -> 0, 0 (A, B) - perfect Gini of 0.

Right we have -> 1, 0, 1, 1, 0, 1, 1, 1 (C, D, E, F, G, H, I, J)
we have 2 that don't prefer remote work and 6 that do.

calculate the Gini of this side: 

$$p(1)= \frac{8}6​=0.75$$$$p(0)=\frac{2}8=0.25$$
now we do the rest of the steps as:

$$I_G(p) = 1 - ((0.75)^2 + (0.25)^2)$$$$I_G(p) = 1 - 0.625$$$$I_G(p) = 0.375$$
After we split we usually look at the weighted Gini, this will tell us how good our split was:

$$\text{Weighted Gini} = \left(\frac{n_{\text{left}}}{N} \times G_{\text{left}}\right) + \left(\frac{n_{\text{right}}}{N} \times G_{\text{right}}\right)$$

The n_left is the left child - means: all the numbers on the left side (We have 2)
The n_right is the right child - means: all the numbers on the right side (We have 8 numbers)

The N stands for the total amount of numbers. Which equals to 10 = (8 + 2).

$$\text{Weighted Gini} = \left(\frac{2}{10} \times 0\right) + \left(\frac{8}{10} \times 0.375\right)$$

$$\text{Weighted Gini} = 0 + (0.8 \times 0.375)$$

$$\text{Weighted Gini} = 0.30$$

So if we put the threshold of 41.5 we get a weighted of Gini of 30%, which is pretty good actually.

Now let's check for the other thresholds too:

- Threshold 41.5: Weighted Gini = 0.30
- Threshold 49.0: Weighted Gini = 0.42
- Threshold 56.5: Weighted Gini = 0.32
- Threshold 74.5: Weighted Gini = 0.45
- Threshold 81.5: Weighted Gini = 0.34

So we can clearly see that the best option is leting it on threshold 41.5, It is not right to continue when we have so many midpoints, because we will split the data too many times and we will do a big mistake (Overfitting)

What is Overfitting? let us give you an example:
Imagine you are preparing for a difficult university entrance exam. Instead of studying the core mathematical concepts, you decide to memorize the exact numbers and answers of the 20 questions on the practice quiz.
When you retake the practice quiz, you score a perfect 100%. You look like a genius. But when you sit down for the actual exam and the questions use completely different numbers, you fail miserably. You didn't learn math; you just memorized a specific history.

So that it, now we will compute it on python step by step, because otherwise we will make a 100 line code that will be too hard to explain. Firstly we will compute the table and make the function for the gini:

Table:

| **Individual** | **Introversion Score (1-100)** | **Prefers Remote Work (0 = No, 1 = Yes)** |
| -------------- | ------------------------------ | ----------------------------------------- |
| A              | 22                             | 0                                         |
| B              | 35                             | 0                                         |
| C              | 48                             | 1                                         |
| D              | 50                             | 0                                         |
| E              | 63                             | 1                                         |
| F              | 71                             | 1                                         |
| G              | 78                             | 0                                         |
| H              | 85                             | 1                                         |
| I              | 92                             | 1                                         |
| J              | 97                             | 1                                         |

```python
import numpy as np

X = np.array([[22], [35], [48], [50], [63], [71], [78], [85], [92], [97]])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
# This is how the table looks on python

def calculate_gini(labels):
	n_samples = len(labels)
	if n_samples == 0:
		return 0
	_, counts = np.unique(labels, return_counts=True) 
	return 1.0 -np.sum((counts / n_samples) ** 2)
```

- `n_samples = len(labels)` - the `labels` will be our y, so we look at the total amount of numbers in our label (In our case it is 10, since we have in our y `[0, 0, 1, 0, 1, 1, 0, 1, 1, 1])`. We will need this n_sample, since we will use it in the probability.
- `if n_samples == 0: return 0` - this is a safety check, because imagine if y was empty... no nothing... then we will return 0, so it doesn't explode.

Now we need to count the classes... we have four 0s... and six 1s... but we don't actually have to do it by hand. Because this line will do it for us:

- `_, counts = np.unique(labels, return_counts=True)` - it will count the classes instead of us, but we want to break it in other small piece:
     -  `np.unique(labels)` - This will check our label (y), and return the unique classes `[0,1]` in our case.
     -  `np.unique(labels, return_counts=True)` - this will return two arrays:
     1) `array([0,1]),` 
     2) `array([4,6]))`
     the first array is which unique classes we have? (In our y we have just `[0,1]` as unique classes)
     the second array give us the count of each unique class, since it returned `[4,6]` it means that we have four 0s and six 1s
     
     But since we don't need the class array and need just the count array, we will write:
     `_, count = np.unique(labels, return_counts=True)` since the lowercase means that we will not use this value, so we get back the array with the count `[4,6]`
 - `return 1.0 -np.sum((counts / n_samples) ** 2)` - this is the Gini impurity (Or also Ginni cruddiness) formula, just computed on python. 
     - `np.sum(counts / n_samples) ** 2)` - this part will convert our counts into probabilities and then square them, as we did before:
        $$((\frac{4}{10})^2 + (\frac{6}{10})^2)) $$
    - `1.0 - np.sum((counts / n_samples) ** 2)`
         $$1 - (0.16 + 0.36)$$
         $$I_G(p) = 0.48$$

We can continue, since we pretty much understand that part:

```python

def build_tree(X, y, depth=0, max_depth=2):
    # Base Case: Stop if pure or depth limit reached. Return majority vote.
    if calculate_gini(y) == 0 or depth >= max_depth:
        return int(np.round(np.mean(y)))

    best_gini, best_thresh = 1.0, None
    
    # Generate midpoints sequentially from unique sorted values
    sorted_vals = np.sort(np.unique(X[:, 0]))
    midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

    # Run the tournament loop
    for thresh in midpoints:
        left_y = y[X[:, 0] < thresh]
        right_y = y[X[:, 0] >= thresh]
        
        weighted_gini = (len(left_y) * calculate_gini(left_y) + len(right_y) * calculate_gini(right_y)) / len(y)
        
        if weighted_gini < best_gini:
            best_gini, best_thresh = weighted_gini, thresh

    # Split the dataset using the winner and spawn the child rooms recursively
    left_mask = X[:, 0] < best_thresh
    return {
        f"X < {best_thresh}": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
        "else": build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth)
    }

# Run it
tree = build_tree(X, y, max_depth=2)

import pprint
pprint.pprint(tree)

"""
Output:

{'X < 41.5': 0, 'else': {'X < 81.5': 1, 'else': 1}}
"""
# We can see that the second part is useless, because both branches return 1. So we can just make our decision tree as: X < 41.5
```

- `def build_tree(X, y, depth=0, max_depth=2)` - we add our feature (X) and label (y). Then we write `depth = 0` - it means that we start from the depth of 0 (Because we just started), and we placed the max depth of 2, because:
     - Depth 0 (The Root): The tree asks its very first question (`X < 41.5`).
     - Depth 1 (The Children): Inside the messy right room, it asks a second question (`X < 81.5`).
     - Depth 2 (The Limit): The code checks its current depth. Because `depth == max_depth`, the loops stops and don't create further checks. To don't Overfit the model.

-   `if calculate_gini(y) == 0 or depth >= max_depth: return int(np.round(np.mean(y)))` -
     - the `if calculated_gini(y) == 0` - asks every time "Should I stop building"?
        - `calculate_gini(y) == 0` - imagine that the room contains `[1, 1, 1, 1, 1]` in that case the `gini = 0`, so everything is already identical. No more questions are needed.
    - the `depth >= max_depth` - even if the room contains `[0, 1, 0, 1]`, this is not pure, but if we reached the max_depth, so it stops anyway.

- `return int(np.round(np.mean(y)))` - this is the majority vote, suppose we have this in the room `[0, 1, 1, 1]`, the np.mean will do (1 + 1 + 1 + 0)/4 = 0.75, in this case the `np.round(0.75)` will round it to the closest whole - in our case the leaf predicts 1.

- `best_gini = 1.0`
  `best_thresh = None` - this says that our current best Gini is of 1 (Worst nowhere, so every new guess will beat this one)

- `sorted_vals = np.sort(np.unique(X[:,0]))` - 
     -  `X[:,0]` - it literally means: "Give me every row from column 0." ; output could be `[22, 35, 48, 50, 63, 71, 78, 85, 92, 97]` 
     -  `np.unique` - it removes all the duplicates of the exact column 
     - `np.sort` - this will sort the values (From smallest to biggest)

- `midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2` - this concept is mega important, because it will find all the midpoints automatically:
  22 ----28.5----35
  35 ----41.5----48
  48 ----49----50
  and so on...

- `left_y = y[X[:,0] < thresh]` - this literally means: "If the value is smaller than the threshold, the value will go to the left". For example: If the threshold will be 49:
     since we have `[22, 35, 48]`, it will go to the left:
	     `left_y = [0, 0, 1]`, everything else goes: `right_y = [0,1,1,0,1,1,1]`

- `weighted_gini = ...` - this is the weighted Gini, just computed on python

- `if weighted_gini < best_gini:`
     `best_gini, best_thresh = weighted_gini, thresh` - it will simply change the variables depending by the best value.

- `left_mask = X[:,0] < best_thresh` - this literally forgets about all the others bad threshold, and holds the best threshold.

- `build_tree(...)` - this is the start to everything:
     - Each child repeats the entire process:

    1. Check if it should stop.
    2. Generate candidate thresholds.
    3. Find the best split.
    4. Create its own children.

     This repetition continues until every branch reaches a stopping condition.
-  `pprint.pprint(tree)` - this will make the output look more pretty and readable.


but production ready is different, sadly there is this demon of `scikit-learn` will eat the knowledge, because it is much more secure and better than this, but we at least understand what happens under the hood (Our was just a simplified version).

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier  

X = np.array([[22], [35], [48], [50], [63], [71], [78], [85], [92], [97]])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
# 1 = Prefer remote work
# 0 = Don't prefer remote work  

model = DecisionTreeClassifier(
criterion="gini",
max_depth=2,
min_samples_split=2,
min_samples_leaf=1,
random_state=9,
)

model.fit(X, y)
prediction = model.predict([[100]])

print(f"{prediction} -> prefer remote work" if prediction == 1 else f"{prediction} -> don't prefer remote work")

"""
Output:

[1] -> prefer remote work
"""
```

that how a decision tree work, now we will use something that overpowers it in way too many cases. But at least you are a certified 'Decision Tree maker', so be proud.

Now after making a tree by hand, we will do a forest!

## 4. Random Forest

In poor words, the Random Forest is just many decision trees working together - surprising, right?

But we have to understand, why we would stick to the random forest but not to the decision tree?
It always depends by the situation, but to be blunt - a decision tree can't handle well information change (I can't handle changes too). Why do I say so?

Because imagine this... after making a decision tree, we want to add somebody there.... The problem is one. The midpoints will change straight away, and they may change the result of the tree totally.
For example:

We have this table:

|Introversion|Remote Work|
|--:|--:|
|22|0|
|35|0|
|48|1|
|50|0|
|63|1|
|71|1|
|78|0|
|85|1|
|92|1|
|97|1|

we have a best split:  'X < 41.5', but... what if we add a new person?

|Introversion|Remote Work|
|--:|--:|
|74|0|

Now out best split changes and the is not 0.3000 but: 0.364

That why we can say that the Decision Tree is highly sensitive to the exact boundaries of the training data. This extreme sensitivity is the primary reason why we use Random Forest many times.

Because why we would build a highly sensitive tree when we can build 100 of them?
It is like asking 100 doctors instead of one, because each have their own idea about it, as:

one will say Remote, the other will say Office, other 3 will say Remote, and that much more reliable than one answer.
But does the 100 trees have the same threshold? No. Because why would we need 100 trees with 41.5 < X? That would make 100 prerecorded answers and 100 tree copies. That why the Random Forest - as the name suggests is totally random, the choice of people for example if we have a list of: `[A, B, C, D, E, F, G, H, I, J]`, each random tree will choose casually as:
`[A, C, A, F, J, J, B, H, F, D]`. See? they repeat each other, because they use a strategy as 'with replacement' - that means that when we choose a person out of the list, it doesn't disappear, it stays in the list so when we choose the second time we can choose it again, a list could be even: `[A, A, A, H, F, H, E....]` and so on.

Anyway, there is a really interesting concept, the OOB (Out-Of-Bag).
What is it? Imagine this:
You are in a room with 100 people, you write down each individuals name on a small piece of paper, and then you place the 100 pieces of paper in a hat, put your hand in the hat, and extract a piece of paper, look at the name, write it on a list, and place the piece of paper back in the hat, shake the hat, and keep doing the same stuff 100 times. 

What happens?

Because you put the names back every single time, some lucky people are going to get their names drawn twice, three times, or even more. But because those lucky people are taking up multiple slots on your new team list, it means other people are getting completely left out. The hat didn't choose them even once in all 100 tries.
The Magic - 63.2%
The weird idea of probability is that no matter how many people are in the room, whether there is 100 people, 1,000 people, or a million people - as long as you draw the same number of times as there are people, the breakdown splits the exact same way every single time:

- 63% of the people in the room will have their name wrote down on the list once or more times
- The 37% will be never drawn not even a single time, they are the Out-of-Bag data.

That how the random forest works.

In our case let's say that we have a set of (like before):

```
A B C D E F G H I J
```

After drawing 10 times with replacement, your bootstrap sampling training set might be:

```
A C A F J J B H F D
```

What does that mean? That means that we may get such an outcome:

```
Used for training:
A B C D F H J

Never selected:
E G I
```

But remember! This is a estimated proportion, so this is really likely to happen! But it could not always happen, because the probabilities are never the same.

Anyway, the bootstrap sampling is the training data we get for a tree that got created, as: 
`[A C A F J J B H F D]` this is a bootstrap sampling, because it was created by the original data; `A B C D E F G H I J`

A Random Forest is made up of many Decision Trees. Every tree follows the exact same Decision Tree algorithm, but each tree is trained on a different bootstrap sample of the data. Because of this, every tree learns a slightly different view of the dataset.

This means that each individual tree is an imperfect expert. It is not imperfect because it uses a worse algorithm, but because it does not see all the same training data or all the same features at every split. As a result, some trees will make mistakes on certain samples... 

Let's say a new sample is predicted:

- Tree 1 → Like IT
- Tree 2 → Like IT
- Tree 3 → Dislike IT
- ...
- Tree 100 → Like IT

So let's say we ended up with :

- 21 trees -> Dislike IT
- 79 trees -> Like IT

So we go with the majority:
Answer -> Like IT

Why we get such an outcome?
Because even if 21 of the trees misclassify the answer, other trees get  the answer right. So we go with the majority.


But even after so much idea we didn't ask ourselves... why is it called 'Random Forest'?

Now I'll show you an example:

| Age | Introversion | Salary | Coffee/day | Likes IT |
| --: | -----------: | -----: | ---------: | -------: |
|  20 |           25 |   2500 |          3 |        0 |
|  22 |           40 |   2800 |          2 |        0 |
|  27 |           60 |   3400 |          4 |        1 |
|  31 |           70 |   3900 |          5 |        1 |
1 is for 'likes IT' and 0 is for 'don't like IT'

There are 4 features:
Age, Introversion, Salary, Coffee.

A normal decision tree looks at the features and asks itself... "Which feature gives me the lowest Gini cruddiness?".
Suppose it thinks that introversion is the best feature, so it chooses it. After some calculations, it chooses the best threshold:

Introversion < 50. 

Then it checks the left child, and after this it checks the features again, suppose that introversion is the best again, so it takes it again.

And in the end the tree is fully based on a strong feature. Strong? Yes. Safe? Not really.

But what does the Random forest does?

Random Forest changes the rules completely. It takes... some features, not all of them.
For example:

Tree #1 may get at the root (from the start) only the features of `[Age, Coffee\day]`, he can't even imagine what salary and introversion look like.
Tree #2 may get at the root only the features of `[Age, Introversion]`

so we can clearly understand Every tree sees different features at different splits. 

This is really useful, because now the trees have a different strongest feature, not always the same, and that really helpful, because let us say that we look at a list of patients that have different features:

- Cough Severity
- Fever
- Age
- Blood Pressure

The decision tree will choose the best feature... suppose it is fever.
After splitting them, it checks the child nodes, and then checks the best feature again... maybe fever win again, so we get again the child split by the fever, even if there are patients of 80 years old with high blood pressure.

But here the Random Forest saves us, because it will sparse different features and check the Gini cruddiness for each of the trees, so a tree that don't have fever as feature will be forced to rank by another feature, and it will be much more safe then a Decision Tree.


Let's remember a concept we already learned - OOB (Out of Bag).
But why do we even care about it?

Because imagine this...

``` Storytime
You sit in your room... staring at the celling, there is nothing that really matters in that moment. You just remembered that you teached a RandomForest model, but do you care? Hardly. Yet the company requires it to be already ready, so now you think to yourself "Is my model good enough?", some distress started to creep in. The only thing that you wanted in that moment was to make sure that model will get a good score. That why you thought that giving it the data we already used for training would be a good idea... yet you checked online firstly, sadly....
```

The answer will be no. Why? Because imagine a teacher giving the exam question to the students before the exam. That what happens, the model already memorized the answers, and it will just get a 100% score. You may be happy, but does that mean that it understood something? 
No.
Because he just memorized the answers.

Normally, we solve this by splitting the dataset into two parts:

```
Training Set
```

Used to train the model.

```
Test Set
```

so we firstly train him on the training set and then try to see how good he is with the test set. Because he has never saw the exam questions. So now you try him using the test set and see how it went.

But think even about it:

Imagine we only have 100 patients.
If we reserve 20 of them for testing, we can train using only the remaining 80 patients. Wouldn't it be nice if we could somehow use all 100 patients for training while still having a fair way to test the model?

This is exactly what the Out-of-Bag (OOB) idea gives us.
Remember our bootstrap sample:

```
A C A F J J B H F D
```

Notice that

```
E G I
```

were never selected.

Tree #17 has never seen these patients. So if we ask Tree #17 to predict patient E, it cannot possibly have memorized the answer. For Tree #17, patient E behaves exactly like a brand-new patient arriving at the hospital. That makes E a fair test. The same goes for our sweet G and I.

In other words, bootstrap sampling accidentally creates thousands of tiny "mini-exams" for the trees, allowing us to evaluate the model while still training on almost the entire dataset.

Now after so much theory, we will use the beautiful industry standart:

| Patient | Age | Fever | Cough Severity | Blood Pressure | Oxygen | Sick |
| :------ | --: | ----: | -------------: | -------------: | -----: | ---: |
| A       |  22 |     0 |              1 |            118 |     99 |    0 |
| B       |  35 |     1 |              6 |            126 |     95 |    1 |
| C       |  48 |     1 |              8 |            138 |     91 |    1 |
| D       |  50 |     0 |              2 |            124 |     98 |    0 |
| E       |  63 |     1 |              7 |            145 |     90 |    1 |
| F       |  71 |     1 |              9 |            152 |     87 |    1 |
| G       |  78 |     0 |              3 |            142 |     96 |    0 |
| H       |  85 |     1 |             10 |            158 |     84 |    1 |
| I       |  92 |     1 |              8 |            162 |     82 |    1 |
| J       |  97 |     1 |              9 |            166 |     80 |    1 |

```python
from sklearn.ensemble import RandomForestClassifier 
import numpy as np

# Features:
# [Age, Fever, Cough Severity, Blood Pressure, Oxygen]
X = np.array([
    [22, 0, 1, 118, 99],   # Patient A
    [35, 1, 6, 126, 95],   # Patient B
    [48, 1, 8, 138, 91],   # Patient C
    [50, 0, 2, 124, 98],   # Patient D
    [63, 1, 7, 145, 90],   # Patient E
    [71, 1, 9, 152, 87],   # Patient F
    [78, 0, 3, 142, 96],   # Patient G
    [85, 1,10, 158, 84],   # Patient H
    [92, 1, 8, 162, 82],   # Patient I
    [97, 1, 9, 166, 80],   # Patient J
])

  

X_casual_test = np.array([[25, 0, 1, 115, 99], [60, 1, 8, 140, 89]])

# 0 = Healthy
# 1 = Sick
y = np.array([
    0,  # A
    1,  # B
    1,  # C
    0,  # D
    1,  # E
    1,  # F
    0,  # G
    1,  # H
    1,  # I
    1   # J
])


model = RandomForestClassifier()

model.fit(X, y)

prediction = model.predict(X_casual_test)

model.predict_proba(X_casual_test)

print(f"Predictions for new patients: {prediction}")

print(f"Confidence (Spread between Healthy vs Sick):\n{confidence}")

print(f"Feature Importances (Age, Fever, Cough, BP, O2) : \n{model.feature_importances_}")

"""
Output:

Predictions for new patients: [0 1]
Confidence (Spread between Healthy vs Sick):
[[0.99 0.01]
 [0.   1.  ]]
Feature Importances (Age, Fever, Cough, BP, O2):
[0.02679574 0.2312009  0.31890332 0.14718615 0.2759139 ]
"""
```

Now we will still keep learning the same idea of decision trees... but now that we are not happy with a forest we learn a XGBoost. 

## 5. XGBoost

What is XGBoost?
XGBoost is another idea that come from the decision tree. But now you will think - like all the normal people do - WHY DO WE NEED ANOTHER TREE?? 

And you are right, but all of the previous had a problem:

Decision Tree -> it is way too sensitive to a new data added to the sample. 
Random Forest -> It doesn't learn from the previous mistakes at all.

What do I mean by "It doesn't learn from the previous mistakes at all."

The Random Forest logic:

Airi (Just a name, no new terminology) goes to 100 doctors, just to make sure she is healthy or no.

#1 Doctor said Pneumonia
#2 Doctor said Cancer
#3 Doctor said  Flu
#4 Doctor said Healthy
$5 Doctor said Paraphilic Disorders

After 100 visits they vote on a poll who thinks what and the one with the biggest vote get chose.
But did you notice something? The trees are all independent from the previous tree one.

They dont learn from the mistake the other made, they don't care. That why the XGBoost exists.

The idea of the XGBoost is:

```
The first Doctor looks at Airi and says "I think your health score is 50." But Airi's real health score is 80. So Doctor #1 made a mistake of -30.

The second Doctor doesn't look at Airi from scratch. He looks _only_ at that -30 mistake. He says "Okay, Doctor #1 missed 30 points. Based on this specific error, I am going to add +15 to the score."

Now the collective guess is 65. The remaining mistake is -15.

The third Doctor looks _only_ at that remaining -15 mistake. He says "Alright, the team is still short by 15 points. I'm going to add +10."

By the end of the line, you don't just take the opinion of Doctor #100. You add all of their tiny notes together: Final Answer = Doctor 1 + Doctor 2 + Doctor 3 + Doctor 4... 
So then they say the result as... mild flue, and some mental disease... 

```


So none of the tree start from nothing, they start from the mistake the other tree did.

But how the math work? (Scary part incoming...):

Let say that Airi Sezaki found a rhythmic game online, and lately she plays it a lot, so her score became:

| **Game** | **Scroll Speed (X)** | **PERFECT Hits (y)** | **Baseline Prediction (y^​)** |
| -------- | -------------------- | -------------------- | ----------------------------- |
| #1       | 4.0                  | 40                   | 70                            |
| #2       | 6.0                  | 80                   | 70                            |
| #3       | 8.0                  | 90                   | 70                            |

This are two features. So we calculate the Baseline prediction, which is just:

$$F_0(x) = \frac{1}{n} \sum_{i=1}^{n} y_i$$

or in poor words - the mean; the average. Since we have 3 rows (With the score of the features), we will put n = 3. Then we will just do $\frac{(40 + 80 + 90)}3$ = 70.

Now we can find the gradient, which is just $g_i = \hat{y}_i - y_i$ , this asks us "By how far was the prediction from the reality?" and the Hessian ($h$) - which is just "Is the error changing gently, or is it changing very sharply?"

If the error changed very sharply, Airi would make only a small adjustment because a tiny change in her prediction could have a huge effect.

If the error changed gently, she could adjust more aggressively.:

Right now we used the Mean Square Error, so our Hessian (second derivative) is always 1

|**Game**|**Scroll Speed (X)**|**Target (y)**|**Prediction (y^​)**|**Gradient (g=y^​−y)**|**Hessian (h)**|
|---|---|---|---|---|---|
|#1|4.0|40|70|70 - 40 = **30**|1|
|#2|6.0|80|70|70 - 80 = **-10**|1|
|#3|8.0|90|70|70 - 90 = **-20**|1|

Now we find the threshold by finding the midpoints (Just like Decision tree)

Game Threshold A:
Between 4.0 and 6.0 -> 5.0

Game Threshold B:
Between 6.0 and 8.0 -> 7.0

Before running the tournament (Splitting by thresholds, we need to understand how XGBoost judges whether a split is good or bad.

In a Decision Tree, we asked: "Which split gives me the lowest Gini cruddiness?"

In XGBoost, we ask a different question: "If I create this leaf, how much will it help correct the mistakes made by the previous trees?"

To answer that question, XGBoost computes a score for every leaf.
The formula is:

$$Score = \frac{(\sum g_i)^2}{\sum h_i + \lambda}$$
But what does it means?

- $g_i$ = the gradient (how wrong the prediction was)
- $h_i$ = the Hessian (how sensitive the loss is)
- $\lambda$ = a regularization penalty that discourages overly complex trees


$$
\sum g_i = 30 + (-10) + (-20)
$$

because our numerator is 0, even the answer will be 0, so there is no point in it, Ill just tell that we have 3 hessians, so we will do $h_1$$+ h_2$$+ h_3$ = 3 and lambda.. will be 1.$$
\text{Score} = \frac{0}{3 + 1}
$$
$$
\text{Score} = \frac{0}{4}
$$

$$
\boxed{\text{Score} = 0}
$$

what does the scores usually mean?

- A low score means -> A giant messy and filthy mix of problems. Half the predictions were too high, half were too low, and they completely canceled each other out. It gives the model zero clear clues on how to fix the error.

- A high score means -> A massive, crystal-clear problem where everyone is wrong in the exact same way. Everyone was under-predicted by a mile. This is a goldmine for the model because one simple adjustment fixes the error for the entire group.

- A medium score means -> A decent hint, but still a bit blurry. Most predictions were too low, but a couple were too high. The model can make a halfway-decent correction, but it won't be perfect.

So in our case we got a punch in the gut - cute, but not useful at all.
We purely continue with the mess of the tournament (to split them):

Just so we remember what we did:
Game Threshold A:
Between 4.0 and 6.0 -> 5.0

Game Threshold B:
Between 6.0 and 8.0 -> 7.0

firstly we will put the threshold of 5

is X < 5:
	Yes? Go to the left
	No? Go to the right

Now we have
`left_child = [30]`
`right_child = [-10, -20]`

So we check again the leaf score.

Left_child -
$$\frac{30^2}{1 + 1} = 450$$
supposing that lambda is 1 (So we can tell the model to don't be too excited, because the group is too small to be trusted)

Right_child -
$$\frac{(-10 -20)^2}{2 + 1} = 300$$
We use the Hessian as 1 (and since we had two numbers 1+1 = 2).

Let's see the gain formula:
$$Gain=Score_{left}​+Score_{right}​−Score_{parent}​$$
since our $score_{parent}$ = 0 

We will simply do $450 + 300 = 750$ 



Now we will put the threshold of 7

is X < 7:

- Yes? → Go to the left
- No? → Go to the right

Now we have

```
left_child = [30, -10]
right_child = [-20]
```

Now we calculate the score (again):

Left_child:
$$(30 - 10) = 20$$
$$\frac{20^2}{2 + 1} = 133.33$$

Right_child:
$$\frac{(-20)^2}{1 + 1} = 200$$

What is the gain?

$$Gain=Score_{left}​+Score_{right}​−Score_{parent}​$$
Since our $score_{parent}$  is equal to 0

We will do: $133.33 + 200 = 333.33$ 

So which is a best threshold?

Clearly the Game Threshold A:
Between 4.0 and 6.0 -> 5.0.

Because it has a better gain:
$750>333.33$

and after choosing the threshold, we let them split and use another formula, to get the correction value (w - AKA leaf weight):

$$w = \frac{-\sum g_i}{\sum h_i + \lambda}$$

That the score formula, just without square
Let's calculate the real correction note for our two winning leaves:

- Left Leaf (Game #1, where $g = 30$):$$w_{\text{left}} = \frac{-30}{1 + 1} = -15$$
(The model says: "Our baseline guess was too high for this game, so we need to subtract 15 points.")

- Right Leaf (Games #2 & #3, where $\sum g = -30$):    $$w_{\text{right}} = \frac{-(-30)}{2 + 1} = \frac{30}{3} = +10$$
(The model says: "Our baseline guess was too low for these games, so we need to add 10 points.")

to just don't add the numbers together and make our model choke and die, we use the learning rate ($\eta = 0.1$) 

so left leaf: $-15 * 0.1 = -1.5$ 
right leaf: $10 * 0.1 = +1$

- **Game #1 New Prediction:** $70 - 1.5 = \mathbf{68.5}$ (Down from 70, creeping closer to her true score of 40)
    
- **Game #2 New Prediction:** $70 + 1.0 = \mathbf{71.0}$ (Up from 70, creeping closer to her true score of 80)
    
- **Game #3 New Prediction:** $70 + 1.0 = \mathbf{71.0}$ (Up from 70, creeping closer to her true score of 90)

| **Game** | **Scroll Speed (X)** | **Target (y)** | **Prediction (y^​)** | **Gradient (g=y^​−y)** | **Hessian (h)** |
| -------- | -------------------- | -------------- | -------------------- | ---------------------- | --------------- |
| #1       | 4.0                  | 40             | 70 - 1.5             | 68.5 - 40 = **28.5**   | 1               |
| #2       | 6.0                  | 80             | 70 + 1.0             | 71.0 - 80 = **-9.0**   | 1               |
| #3       | 8.0                  | 90             | 70 + 1.0             | 71.0 - 90 = **-19.0**  | 1               |

and so we will do in production ready:
```python
import xgboost as xgb
import numpy as np  

X_train = np.array([[4.0], [6.0], [8.0]])
y_train = np.array([40, 80, 90])

X_test = np.array([9.0])
y_test = np.array([95])

# Define the model with hyperparameters
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1, base_score=70)

# Train it
model.fit(X_train, y_train)

prediction = model.predict(X_test)

for i in range(len(y_test)):
	print(f"Speed {X_test[i]}: Predicted = {prediction[i]:.2f} | Real Target = {y_test[i]}")

"""
Output:

Speed 9.0: Predicted = 89.88 | Real Target = 95
"""
```

what does the `n_estimators = 100` means? It means all the loops it passes through (The 100 Doctors)
what about `base_score` = that our gradient (average score for the gradient)

But why we shall use this computed version from the library, because it does much more stuff than we did. 
We tried to do it just once and we look just at one feature: Scroll Speed. In a real dataset, you might also have features like Device Audio Latency, Screen Size, or Hours of Sleep.

During the tournament phase, XGBoost doesn't just look at the midpoints for Scroll Speed. It runs the exact same Gain formula for every single midpoint of every single feature you give it.

Note: Historically saying, the Gradient Boosting usually comes after XGBoost, but guess what? The author is too much of a silly goose, and he actually did the opposite. So guess what, kids? No more food. Because already by reading this you can understand 99% of what Gradient Boosting is.
Main differences:

- it uses residuals ($r$), not gradients ($g$). 
  $r = y - \hat{y}$  instead of  $g = \hat{y} - y$ 

- It never uses Hessian, Lambda, Gain Formula. And it has a simpler tree construction.


Now the suffering ended with this! But life is not sweet, and wants you to suffer when you learn Linear algebra. Soooo, we will start with PCA!...
Jokingggg, almost believed it! We will make a whole learning progress under the name of PCA (Principal Component Analysis) - we will start from a semi-beginner stuff and reach PCA slowly. This will be a long topic, so be ready and prepare. Because I already wrote all the Random forest and XGBoost in one day - 16:51 PM. I took just some breaks to read "Kitanai Kimi ga Ichiban Kawaii", soooo, I'll dive half way today and tomorrow finish this whole chapter, so It will exactly take 2 days and some hours - Or maybe I'll go really deep with PCA and end it in 3 days+-, because this topic is really pivotal for Spectral graph and future projects I will make... and the one you will too, because I'll give you some ideas too.

And anyway, since we were not happy about a tree, a forest, and a XGBoost, we wen't straight into linear algebra.
Cute? Absolutely not. The ideas are brain... tiring.


## 6. PCA

That how our fake PCA starts, because now we will do a 12 hours of topic in 2 hours or 3. (Or as I said... more.)
`1. Variance -> 2. Covariance -> 3. Covariance Matrix -> 4. What is a direction?    -> 5. Eigenvectors ->  6. Eigenvalues ->  7. SVD ->  8. PCA`

Okay, so let us start with Variance!

### 1. Variance

Imagine Airi... She has been practicing to that rhythm game for a week, and that the results of her so intense tries (look at the PERFECT Hits):

Week 1:

| Day | PERFECT Hits |
| --: | -----------: |
|   1 |           70 |
|   2 |           71 |
|   3 |           69 |
|   4 |           70 |
|   5 |           70 |

Now I have a question... Is Airi consistent?
Absolutely, she is constantly around 70

Now imagine she practiced another week, but in the first day she trip out of the bed and hurt her hand. In the third day Hianko gave her a visit, so she was really distracted.

Week 2:

| Day | PERFECT Hits |
| --: | -----------: |
|   1 |           15 |
|   2 |           96 |
|   3 |           42 |
|   4 |           83 |
|   5 |           74 |

Is she consistent?
Absolutely not! Because some days she plays like a beginner, and the other days she plays like she sold her soul for some rhythmic game skills.

So already our brain notices, the first week the score is tight and in the other week it is all over the place
Even if the average is almost identical, the difference is that the variance of the Week 1 is small, while the variance of the Week 2 is high!

But now we have another question... Why do we even care about it? like why would we even learn this?  Good question, and now i'll tell you!

Because every Data Engineer checks the variance immediately. Because look at the table:

| Day | PERFECT Hits |
| --: | -----------: |
|   1 |           70 |
|   2 |           71 |
|   3 |           69 |
|   4 |           70 |
|   5 |           70 |

And now ask yourself; "If I would use this for the feature, will that help me?" No. Because the variance is basically 0. 

Think about 20 patients, if you see that their age is around 34-35, will you use it as a valid feature? No. Because the data is useless to feed our model with.
What if the patients had a high age variance? imagine the age is from 25-81 and you noticed that the older is the person the bigger is the chance of it getting sick, then we notice that the feature is really good and helpful for predicting.

Another example (more mathematical):

Think about Airi's score, and now we think "Compared to what should we measure the spread?"
Obviously to the average, dummy.
That why we find the mean (average), which is:

$$\mu=\frac{70+71+69+70+70​=70}5 = 70$$

So now we just do $original - average$:

| Score | Mean | Distance |
| ----: | ---: | -------: |
|    70 |   70 |        0 |
|    71 |   70 |        1 |
|    69 |   70 |       -1 |
|    70 |   70 |        0 |
|    70 |   70 |        0 |

positive outcomes mean: 
above average

negative outcomes mean:
bellow average

so now we do: 0 + 1 + (-1) + 0 + 0 = 0...
Not on the plan... but no worries, we change the idea to square the numbers!
Why to don't use the absolute value?

Because squaring the number means:
- negatives become positive
- bigger deviations are punished much more

so our result is now:

| Score | Distance | Squared Distance |
| ----: | -------: | ---------------: |
|    70 |        0 |                0 |
|    71 |        1 |                1 |
|    69 |       -1 |                1 |
|    70 |        0 |                0 |
|    70 |        0 |                0 |

So we do the same again; 0 + 1 + 1 + 0 + 0 = 2

now we find the average spread!
$$\frac{2}5 = 0.4$$

which is 0.4 -> Extremlly tight!

So what formula we just 'created' ->
$$\sigma^2 = \frac{1}n \sum_{i=1}^n (x_i - \mu)^2$$
That what we just did
Soo, that how the variance work!

In python it looks this way:

```python

import numpy as np

x = np.array([70, 71, 69, 70, 70])

def variance(feature):
    n = len(feature)
    mean = np.mean(feature)

    return np.sum((feature - mean) ** 2) / n

var = variance(x)
print(var)

"""
Output:

0.4
"""
```

Next step is the Covariance.

### 2. Covariance

What is Covariance?...
Let's think about something
Imagine that Hinako Hanamura was practicing to the piano, her progress look like this:

| Day | Practice Hours | Songs Learned |
| --: | -------------: | ------------: |
|   1 |              1 |             2 |
|   2 |              2 |             4 |
|   3 |              3 |             7 |
|   4 |              4 |             9 |
|   5 |              5 |            11 |

What do we understand?  We understand that:
As she practices more...
she learns more songs.
These two features move together.

Practice+ = Song Learned+    ---> Positive covariance

What about something a little sadder:

|Day|Hours Awake|Energy Level|
|--:|--:|--:|
|1|6|95|
|2|10|82|
|3|14|61|
|4|18|37|
|5|22|14|

The longer she's awake...
the more exhausted she becomes.
One feature increases.
The other decreases.

Hours Awake+ = Energy Level-    ---> Negative covariance


And now we have some nonsense

Hinako is reading outside:

|Day|Pages Read|Number of Clouds Outside|
|--:|--:|--:|
|1|12|8|
|2|40|1|
|3|18|10|
|4|53|4|
|5|25|7|

Sometimes pages go up.
Sometimes clouds go down.
Sometimes both increase.

Even if the feature grows or goes down, the second feature is casual (may grow or fell each time)
Pages read+- = Number of Clouds Outside+-    ---> Zero correlation


Now we will 'create the formula'

| Day | Practice Hours | Piano Pieces |
| :-: | :------------: | :----------: |
|  1  |       1        |      2       |
|  2  |       2        |      4       |
|  3  |       3        |      7       |
|  4  |       4        |      9       |
|  5  |       5        |      11      |

First we find the mean (average), just like from the variance:

$$\frac{2 + 4 + 7+9+11}5 = 6.6$$
We will use it for the deviation of piano pieces

$$\frac{1+2+3+4+5}5 = 3$$
We will use it for the deviation of practice hours

that our average... but now we have a one million dollar question... do we square them? No. Because we have two features, that why we will just multiply them

piano pieces:

|Day|Piano Pieces ($y$)|Mean ($\mu_y = 6.6$)|Deviation ($y - \mu_y$)|
|:-:|:-:|:-:|:-:|
|1|2|6.6|$2-6.6=-4.6$|
|2|4|6.6|$4-6.6=-2.6$|
|3|7|6.6|$7-6.6=0.4$|
|4|9|6.6|$9-6.6=2.4$|
|5|11|6.6|$11-6.6=4.4$|

practice hours:

| Day | practice hours($x$) | Mean ($\mu_x = 3$) | Deviation ($x-\mu_x$) |
| :-: | :-----------------: | :----------------: | :-------------------: |
|  1  |          1          |         3          |          -2           |
|  2  |          2          |         3          |          -1           |
|  3  |          3          |         3          |           0           |
|  4  |          4          |         3          |           1           |
|  5  |          5          |         3          |           2           |

Now we just find their product. But why to find their product? Because the product will reveal us if they move togheter them:

| Day | $x-\mu_x$ | $y-\mu_y$ | Product |
| :-: | :-------: | :-------: | :-----: |
|  1  |    -2     |   -4.6    |   9.2   |
|  2  |    -1     |   -2.6    |   2.6   |
|  3  |     0     |    0.4    |    0    |
|  4  |     1     |    2.4    |   2.4   |
|  5  |     2     |    4.4    |   8.8   |

let's look closely:

- Day 1

$$(−2)×(−4.6)=9.2$$
Positive.
Good.
They moved together.

- Day 2

$$(−1)×(−2.6)=2.6$$

Positive again.

- Day 3

$$0 × 0.4 = 0$$

Basically no contribution.

- Day 4

$$1×2.4=2.4$$

Positive.

Again they moved together.

- Day 5

$$2×4.4=8.8$$

Positive.
Once again.

Noticed something? All the answers are positive, not because we forced them, but because both features were moving in the same direction all along.

Now we just add the together:
$$9.2 + 2.6 + 0 + 2.4 + 8.8 = 23$$

But wait, Remember the variance! We didn't stop at adding, in the end we divided them by their total number ($n$). 
$$\frac{23}5 = 4.6$$

Ta-da~ we just did this formula without you ever knowing it:
$$Cov(X,Y) = \frac{1}n \sum_{i=1}^n(x_i - \mu_x)(y_i - \mu_y)$$
The difference is small, in variance we multiply a feature's deviation by itself, meanwhile in covariance we multiply it by the deviation of another feature.

On python it looks as something like this.
```python
import numpy as np

def Cov(X, y):
	n = len(X)
	mean_x = (X - np.mean(X))
	mean_y = (y - np.mean(y))
	return np.sum((X - mean_x) * (y - mean_y)) / n
```

But now we may have a question... what if I have 20 features? Will I have to compute all of them manually?
No.
That why Covariance Matrix exists. Ah, I didn't mean to spoiler it at all, but that it, now we will start the Covariance Matrix... (For you it is maybe today, for me it is tomorrow. Cuz I literally started writing from the Random Forest at 6 AM, so I wrote pretty much in one day - Not good enough, yet almost decent-ish.)

### 3. Covariance Matrix

Good morning to everyone.

This morning I woke up with a question... why do we use Covariance Matrix? 

And the answer is actually simpler than it first looks.
Imagine you having 5 features, or 40, or 100, 4000+
What do we do? Start crying - Good option, yet that wouldn't solve any problem. That what Karl Pearson, Sir Ronald Fisher, and Arthur Cayley thought! (I thought it before them, but didn't feel like writing about it). So with their ideas and help, the Covariance Matrix was born.

But how do we use it beside talking about it?
Let's look how Hinako passed her week (Weekends excluded):

| Day | Practice Hours | Piano Pieces | Sleep Hours | Coffee Cups |
| :-: | :------------: | :----------: | :---------: | :---------: |
|  1  |       1        |      2       |      8      |      1      |
|  2  |       2        |      4       |      7      |      2      |
|  3  |       3        |      7       |      6      |      3      |
|  4  |       4        |      9       |      5      |      4      |
|  5  |       5        |      11      |      4      |      5      |
Now we have 4 features, so to don't confuse them we will give them a name:

X₁ = Practice Hours
X₂ = Piano Pieces
X₃ = Sleep Hours
X₄ = Coffee Cups

So now we will not ask ourselves some thing as:
"What is the covariance between this feature and that one"

We will ask ourselves:
"What is the covariance between every pair of feature we have"

Something as:

- Practice <--> Piano
- Practice <--> Sleep
- Practice <--> Coffee
- Piano <--> Sleep
- Piano <--> Coffee
- Sleep <--> Coffee

That's much more work than computing a single covariance or a single variance. So what will we do in that case?

That we will write a different table:

| | Practice | Piano | Sleep | Coffee |
| :--- | :---: | :---: | :---: | :---: |
| **Practice** | ? | ? | ? | ? |
| **Piano** | ? | ? | ? | ? |
| **Sleep** | ? | ? | ? | ? |
| **Coffee** | ? | ? | ? | ? |
But... what exactly is this table?

Lets write the formula:
$$Cov(X, X) = \frac{1}n \sum(x_i - \mu_x)(x_i - \mu_x)$$

But wait!

$$(x_i - \mu_x)(x_i - \mu_x)$$

This part multiply itself by itself... this remind us of a rule learned in high school:

$$a \times a = a^2$$

We do the same:

$$(x_i - \mu_x)(x_i - \mu_x) = (x_i - \mu_x)^2$$

but we saw this once... That the variance formula!

$$Cov(X,X) = Var(X)$$

This means that each diagonal must be the variance!

|              | Practice Hours | Piano Pieces | Sleep Hours | Coffee Cups |
| :----------- | :------------: | :----------: | :---------: | :---------: |
| **Practice** | Var(Practice)  |      ?       |      ?      |      ?      |
| **Piano**    |       ?        |  Var(Piano)  |      ?      |      ?      |
| **Sleep**    |       ?        |      ?       | Var(Sleep)  |      ?      |
| **Coffee**   |       ?        |      ?       |      ?      | Var(Coffee) |
Because $Cov(X,X)=Var(X)$. Whenever both variables are the same feature, covariance becomes variance (Or in simple words: Because $Cov(X, X) = Var(X)$ - when X repeats itself 2 times it is the variance.)

Remember the table:

| Day | Practice Hours | Piano Pieces | Sleep Hours | Coffee Cups |
| :-: | :------------: | :----------: | :---------: | :---------: |
|  1  |       1        |      2       |      8      |      1      |
|  2  |       2        |      4       |      7      |      2      |
|  3  |       3        |      7       |      6      |      3      |
|  4  |       4        |      9       |      5      |      4      |
|  5  |       5        |      11      |      4      |      5      |
So now we solve the whole problem By using the formula:
$$Var(X) = Cov(X, X) = \frac{1}n \sum_{i=1}^n(x_i - \mu_x)^2$$


We take the mean of each feature:

Mean of Practice Hours:

$$\mu_{\text{Practice}} = \frac{1+2+3+4+5}{5} = \frac{15}{5} = 3$$

Mean of Piano Pieces:

$$\mu_{\text{Piano}} = \frac{2+4+7+9+11}{5} = \frac{33}{5} = 6.6$$

Mean of Sleep Hours:

$$\mu_{\text{Sleep}} = \frac{8+7+6+5+4}{5} = \frac{30}{5} = 6$$

Mean of Coffee Cups:

$$\mu_{\text{Coffee}} = \frac{1+2+3+4+5}{5} = \frac{15}{5} = 3$$


Now we will do $x_i - \mu_x$: 

| **Day** | **Practice Hours** | **Piano Pieces** | **Sleep Hours** | **Coffee Cups** |
| :-----: | :----------------: | :--------------: | :-------------: | :-------------: |
|    1    |         -2         |       -4.6       |        2        |       -2        |
|    2    |         -1         |       -2.6       |        1        |       -1        |
|    3    |         0          |       0.4        |        0        |        0        |
|    4    |         1          |       2.4        |       -1        |        1        |
|    5    |         2          |       4.4        |       -2        |        2        |

- Variance of Practice:    
    $$\sigma^2_{\text{Practice}} = \frac{(-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2}{5} = \frac{4 + 1 + 0 + 1 + 4}{5} = \frac{10}{5} = 2$$
    
- Variance of Piano:
    $$\sigma^2_{\text{Piano}} = \frac{(-4.6)^2 + (-2.6)^2 + 0.4^2 + 2.4^2 + 4.4^2}{5} = \frac{21.16 + 6.76 + 0.16 + 5.76 + 19.36}{5} = \frac{53.2}{5} = 10.64$$
    
- Variance of Sleep:
    $$\sigma^2_{\text{Sleep}} = \frac{2^2 + 1^2 + 0^2 + (-1)^2 + (-2)^2}{5} = \frac{4 + 1 + 0 + 1 + 4}{5} = \frac{10}{5} = 2$$
    
- Variance of Coffee:
    $$\sigma^2_{\text{Coffee}} = \frac{(-2)^2 + (-1)^2 + 0^2 + 1^2 + 2^2}{5} = \frac{4 + 1 + 0 + 1 + 4}{5} = \frac{10}{5} = 2$$
Now the table looks like:

|              | **Practice Hours** | **Piano Pieces** | **Sleep Hours** | Coffee Cups |
| :----------: | :----------------: | :--------------: | :-------------: | :---------: |
| **Practice** |       **2**        |        ?         |        ?        |      ?      |
|  **Piano**   |         ?          |    **10.64**     |        ?        |      ?      |
|  **Sleep**   |         ?          |        ?         |      **2**      |      ?      |
|  **Coffee**  |         ?          |        ?         |        ?        |    **2**    |

That the variance of each!

Imagine doing all this covariance calculation for every pair of features one by one.
...
...
NO. 

Because we will use a linear algebra trick:
$$Σ = \frac{1}n(B^T \times B)$$
Spoilers... The sign we start with is the covariance matrix, not $\sum$ < --- sum.

What does it mean? Let's take the table after we did  ($x_i - \mu_x$):

$$B = \begin{bmatrix} -2 & -4.6 & 2 & -2 \\ -1 & -2.6 & 1 & -1 \\ 0 & 0.4 & 0 & 0 \\ 1 & 2.4 & -1 & 1 \\ 2 & 4.4 & -2 & 2 \end{bmatrix}$$

I'll show just some small step:

- Extract the vectors:
    
    - $\text{Column 1 (Practice)} = \begin{bmatrix} -2, & -1, & 0, & 1, & 2 \end{bmatrix}^T$
        
    - $\text{Column 2 (Piano)} = \begin{bmatrix} -4.6, & -2.6, & 0.4, & 2.4, & 4.4 \end{bmatrix}^T$
        
- Multiply each by the other elements:
    
    - $(-2) \times (-4.6) = 9.2$
        
    - $(-1) \times (-2.6) = 2.6$
        
    - $0 \times 0.4 = 0$
        
    - $1 \times 2.4 = 2.4$
        
    - $2 \times 4.4 = 8.8$
        
- Sum the results:
    
    - $9.2 + 2.6 + 0 + 2.4 + 8.8 = 23$
        
- The covariance:
    
    Last step is always:  $\frac{1}{n} \times \text{Sum}$.
    
    - $\text{Covariance} = \frac{23}{5} = 4.6$

and we do the same for all the pairs till we get:

|              | **Practice Hours** | **Piano Pieces** | **Sleep Hours** | **Coffee Cups** |
| ------------ | ------------------ | ---------------- | --------------- | --------------- |
| **Practice** | 2                  | 4.6              | -2              | 2               |
| **Piano**    | 4.6                | 10.64            | -4.6            | 4.6             |
| **Sleep**    | -2                 | -4.6             | 2               | -2              |
| **Coffee**   | 2                  | 4.6              | -2              | 2               |

That our majestic output... but what now? What we understand?


-------

The variance:

|              | **Practice Hours** | **Piano Pieces** | **Sleep Hours** | Coffee Cups |
| :----------: | :----------------: | :--------------: | :-------------: | :---------: |
| **Practice** |       **2**        |        ?         |        ?        |      ?      |
|  **Piano**   |         ?          |    **10.64**     |        ?        |      ?      |
|  **Sleep**   |         ?          |        ?         |      **2**      |      ?      |
|  **Coffee**  |         ?          |        ?         |        ?        |    **2**    |

the piano pieces have a much larger spread than the other features. Hinako sometimes learns only a few pieces and other days learns more. This feature varies much more across the week.

The covariance:

|              | **Practice Hours** | **Piano Pieces** | **Sleep Hours** | **Coffee Cups** |
| ------------ | ------------------ | ---------------- | --------------- | --------------- |
| **Practice** | ?                  | 4.6              | -2              | 2               |
| **Piano**    | 4.6                | ?                | -4.6            | 4.6             |
| **Sleep**    | -2                 | -4.6             | ?               | -2              |
| **Coffee**   | 2                  | 4.6              | -2              | ?               |

We can notice immediately!
That the matrix is symmetric (It always happens)
Because when we do:

Practice Hours -> Piano = 4.6
Piano Pieces -> Practice = 4.6

That work for all of them

The covariance of Practice Hours -> Piano Pieces = 4.6 (Positive Covariance)
That means that the more hours Hinako practices the more Piano Pieces she learn.

But that not the case for sleep hours, because:
Practice Hours -> Sleep hours = -2 (Negative Covariance)
That means that the more Hinako practices the less she sleeps.

In python we will write:

```python
import numpy as np

X = np.array([
    [1, 2, 8, 1],
    [2, 4, 7, 2],
    [3, 7, 6, 3],
    [4, 9, 5, 4],
    [5,11, 4, 5]
])

def covariance_matrix(X):
    means = np.mean(X, axis=0)
    B = X - means
    covariance = (B.T @ B) / len(B)
    return covariance
    
# Or just:
# covariance = np.cov(X, rowvar=False)

print(covariance_matrix(X))



"""
Output:

[[ 2.    4.6  -2.    2.  ]
 [ 4.6  10.64 -4.6   4.6 ]
 [-2.   -4.6   2.   -2.  ]
 [ 2.    4.6  -2.    2.  ]]
```

That the whole history with them...


So now we will learn... what is a direction?

cute question... that you never had. But we will explore it anyway

### 4. What is a direction

As expected, we all can understand that a direction can be just a line, for example:

Hinako's table about practice and pieces:

|Day|Practice Hours|Piano Pieces|
|:-:|:-:|:-:|
|1|1|2|
|2|2|4|
|3|3|7|
|4|4|9|
|5|5|11|
We can understand that this will be a straight line, for example: 

![[Figure_1001 1.png]]

Are the points randomly scattered? nope, the points are going somewhere for sure, they have a direction. Now imagine drawing an arrow - that the direction.
It simply tells us: "If I move this way, I roughly follow the data."

So what is a direction mathematically?

A direction it's simply a vector
For example:
$$A =
\begin{bmatrix}
1 \\ 0

\end{bmatrix}$$
This vector is just a line to the right.
Meanwhile this vector:
$$B =\begin{bmatrix} 0\\1 \end{bmatrix}$$
This vector will move once up
And in the end:
$$C =\begin{bmatrix} 1\\1 \end{bmatrix}$$

this vector will move diagonally (once right and once up)

same for:
$$D =\begin{bmatrix} 2\\5 \end{bmatrix}$$

this direction shows up that the vector moves more upward than right = Diagonal.
2 times right and 5 times up.

Let's say we take the first part of our result:

$$Σ = \begin{bmatrix} 2 & 4.6 \\ 4.6 &10.64 \end{bmatrix}$$
we can understand:
- Practice Hours and Piano Pieces grow together.
- There is a relationship.

But we don't know which direction follows better this relationship (purely platonic... maybe).

Imagine a ruler, you can rotate it at $90^o$, at $270^o$, at $130^o$... But now we have a question:  
"Which rotation captures the data best?"
That the same question that PCA thinks about... and the answer will be our first eigenvector!

But now we may think... WHY WE WOULD BREAK OUR HEAD SO MUCH JUST FOR SOME DATA?? We do it so we reduce noise, get better outputs by the model, and much more.

But even so...
Now the scary, painful, agonizing, terrible, frightening, abysmal, and terrorizing part starts. 

Eigenvalue...

### 5. Eigenvectors and Eigenvalues

Let's go back to our table:

| Practice Hours | Piano Pieces |
| :------------: | :----------: |
|       1        |      2       |
|       2        |      4       |
|       3        |      7       |
|       4        |      9       |
|       5        |      11      |
We can clearly see the points on the graph (`X = Practice Hours`, while `y = Piano Pieces`)

we will see:

![[Figure_1002.png|483]]

We can say where are the points and how to draw a line, that the direction is upper-right. But machines (computers) can't say so.
They are left in the void, they can just see just:
`(1,2)`
`(2,4)`
`(3,7)`
`(4,9)`
`(5,11)`

They need mathematics to get something in their brain, soooo, who will even help a machine with this ahahah- We will. So we start right on the spot.

Now we have to find a tool... Suppose we say "easy"; take a ruler and try to put it horizontally... then we try to put it vertically... then diagonally... and try many times. So what we are trying to do? We are trying to find the best direction out infinite many direction, that will make us get the more data as possible. Pretty lame.. already after 2 minutes we tried many directions... but we still have billions of possible directions, that why we will use a tool made for this.

Remember the covariance matrix? let's take it with us
$$Σ = \begin{bmatrix} 2 & 4.6 \\ 4.6 &10.64 \end{bmatrix}$$
It may look like a normal table
But it already knows many stuff, for example, it knows:

- how much Practice changes,
- how much Piano changes,
- how much they change together.
(The covariance matrix - think about it as a report of the old data we had, it tells the relationship between the features.)

So instead of rotating the ruler billions of times, imagine staying awake for a day and the covariance matrix looks at you and say:

"You don't have to test every direction."
"I already know the best one."

Suppose you give to the matrix an arrow... something as:
$$\vec{v}= \begin{bmatrix} 1\\0 \end{bmatrix}$$
we say to it: 
"move only to the right"

what does the covariance matrix does?
it transforms the original possition.... sometimes moving it much more up, sometimes more to the left, sometimes to the oposite side... But here is a magical arrow...
That arrow doesn't move our arrow to any side, it just makes it longer... Which?? That the eigenvector. 
But in what way it may help us get the perfect position for our ruler?

But before we get really confused! DISCLAIMER: 

We don't put the covariance matrix on the graph in no way! 
We place the old data that we had, as:

That the old data of which we get the covariance matrix:

| Practice Hours | Piano Pieces |
| :------------: | :----------: |
|       1        |      2       |
|       2        |      4       |
|       3        |      7       |
|       4        |      9       |
|       5        |      11      |
|                |              |
on the graph, meanwhile the covariance matrix is just a spreadsheet that shows us the relation between them, nothing more! That why we scatter the old data on the graph and then we use the covariance matrix to understand the relationship of Practice Hours and Piano Pieces. And in the eigenvector we will use the covariance matrix, because the eigenvalue requires a transformation. That why we use the covariance matrix, because it stores actual information as the relationship of both features and will help us draw a perfect line. (and if we will talk about clouds - we are talking about the dots we scattered on the graph, because when we will work with 1000 dots, it will look like a cloud.) (That what we call a 'Late realization', guys.). Another late realization is that the data (The scattered points on the graph) never moves! The data stays exactly the same... the thing that changes is our coordinate system - just like a map, we turned the map, not the data.

Now we give a real example!

Remember Airi? She kept training till now, and her score improved. 

| Day | Scroll Speed | PERFECT Hits |
| :-: | :----------: | :----------: |
|  1  |      3       |      52      |
|  2  |      5       |      67      |
|  3  |      6       |      74      |
|  4  |      8       |      91      |
|  5  |      10      |     106      |

Firstly we scatter the points on the graph, but that not enough...

![[Figure_1000 1.png|536]]

Because now we want to know the relationship of her features, so we can easily find out how dependent is the data on each other.

so we use the covariance matrix (Wait! But we can use the covariance, no? Yes, but think about the Covariance matrix as the cooler covariance, that why we spam the covariance matrix much more, and beside it the normal covariance can't do any cool SVD or PCA, so it can rest for an eternity, cuz it's filthy).

For sure we remember how to do it:

Find the mean of both features

The Scroll Speed mean:
$$\mu_x = \frac{3 + 5 + 6 + 8 + 10}{5} = \frac{32}{5} = 6.4$$

The PERFECT Hits mean:
$$\mu_y = \frac{52 + 67 + 74 + 91 + 106}{5} = \frac{390}{5} = 78$$

that the mean of our clouds (On the graph):

![[Figure_10001.png|536]]

Now we subtract the original value by the mean

| Day |        $x-\mu_x$ |       $y-\mu_y$ |
| :-: | ---------------: | --------------: |
|  1  | 3−6.4 = **−3.4** | 52−78 = **−26** |
|  2  | 5−6.4 = **−1.4** | 67−78 = **−11** |
|  3  | 6−6.4 = **−0.4** |  74−78 = **−4** |
|  4  |  8−6.4 = **1.6** |  91−78 = **13** |
|  5  | 10−6.4 = **3.6** | 106−78 = **28** |

Now we make our centered matrix (The B we used to get all the covariances)

$$B = \begin{bmatrix} -3.4 & -26 \\ -1.4 & -11 \\ -0.4 & -4 \\ 1.6 & 13 \\ 3.6 & 28 \end{bmatrix}$$
and now comes the old trick:

$$\Sigma = \frac{1}{n}(B^T B)$$
which will result into:

$$\Sigma = \begin{bmatrix} 5.84 & 44.8 \\ 44.8 & 344.4 \end{bmatrix}$$

And now we no longer have Perfect Hits or Scroll Speed, we compressed everything, and now we can see the relationship between the features.
We can understand that the covariance is positive and it stores all the data as:
- how much Scroll Speed varies
- how much PERFECT Hits vary
- how strongly they move together

and now we have a really important question...
"If I had to draw ONE arrow through the middle of the cloud, which direction would explain the most information?"

That where the eigenvector becomes the 'Star'

Now we will use a formula to get that eigenvector, the formula is relatively easy:
$$Av = \lambda{}v$$
Where A becomes our $\Sigma$ 
Anyway, the eigenvector never changes the direction for an already existing line, so don't be scared
Our equation become:
$$\begin{bmatrix} 5.84 & 44.8 \\ 44.8 & 344.4 \end{bmatrix} v = \lambda v$$

From this all we can understand:

The eigenvector answers to such a question as:
"Which direction best follows the cloud?" 

But we have another guest...
The eigenvalue.

The eigenvalue ($\lambda$) answers to such a question as:
"How important is that direction?"

Because think about this two scenario:

$\lambda = 350$ 
From this we can understand that the information is mega stretched almost in a line. It looks like it is about to form a highway 
What does it mean? 
It means that the data is not just "there", it also has a strong and predictable structure!

$\lambda = 0.6$ 
From this we can understand that the data is just spread randomly or is just flat. The points doesn't form any highway; they are just drifting slightly around the center. 

![[Pasted image 20260710161515.png]]

If the eigenvalues were, say, $\lambda_1 = 5$ and $\lambda_2 = 4$, the cloud would be a circle. There is no "special" direction because the data is equally spread in every direction. In this case, PCA doesn't help much because there is no "highway" to find


Now we can continue with the idea, we return to the original formula:

$$Av = \lambda{}v$$

Now we will just move from the right side of the equation the values to the left:
$$Av - \lambda{v} = 0$$
It looks like a highschool formula now:
Just like:
$$3x−5x=(3−5)x$$

we can factor out the vector:
$$(A-\lambda I)v=0$$

$I$ = $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ - It always equals to this, when we have a 3x3 matrix, it just expand to a 3x3 matrix too, it is just that it will still have a row of 1 going diagonally.

Normally - and as usual - a matrix equation has only one solution:
$$v = 0$$
which is completely useless, that why we will... make the determinant be zero!

$$det(A-\lambda I)v=0$$

This equation is called the characteristic equation - It reveals every eigenvalue of the matrix.
Once we know the eigenvalues, we simply plug each one back into

$$(A-\lambda I)v=0$$

So to find everything, now we do:

$$\left( \begin{bmatrix} 5.84 & 44.8 \\ 44.8 & 344.4 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) v = 0$$
so now we will do:

$$-\lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} -\lambda & 0 \\ 0 & -\lambda \end{bmatrix}$$
and it equals to:

$$\begin{bmatrix} 5.84 & 44.8 \\ 44.8 & 344.4 \end{bmatrix} + \begin{bmatrix} -\lambda & 0 \\ 0 & -\lambda \end{bmatrix} = \begin{bmatrix} 5.84 - \lambda & 44.8 \\ 44.8 & 344.4 - \lambda \end{bmatrix}$$

And now we have some unknown vectors... what to do?

We don't have to do nothing special, just use as we said before:
$$det(A-\lambda I)v=0$$

Now we will just solve all in one shot:
$$(5.84 - \lambda)(344.4 - \lambda) - 44.8^2 = 0$$$$2011.296 - 5.84\lambda - 344.4\lambda + \lambda^2 - 2007.04 = 0$$
$$\lambda^2 - 350.24\lambda + 4.256 = 0$$
This equation is called the characteristic equation.

Now we will do the most loved equation in the world...

$$\lambda = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$
$$\lambda_1 \approx 350.23$$

$$\lambda_2 \approx 0.012$$

So that our eigenvalues... what do we understand from them

So now we have this values... what do they mean?

Dominant direction ($\lambda_1 \approx 350.23$) - this is a big value, that why it means that the information is extremely stretched. That means that most than the majority of the information lays on the direction of the first eigenvector

Submissive direction/minimal spread ($\lambda_2 \approx 0.012$) - this is a really small value, meaning that there is almost no variance or spread in the direction of the second eigenvector

Now we substitute the $\lambda$

$$\begin{bmatrix} 5.84 - 350.23 & 44.8 \\ 44.8 & 344.4 - 350.23 \end{bmatrix} v = 0$$
then:
$$\begin{bmatrix} -344.39 & 44.8 \\ 44.8 & -5.83 \end{bmatrix} v = 0$$

Now we make an equation (Take the first row...):
$$-344.39x + 44.8y = 0$$


Rearranging to solve for $y$:

$$44.8y = 344.39x$$

$$y = \frac{344.39}{44.8}x$$

$$y \approx 7.69x$$

Now we just choose whatever number we want for x:
let's always choose 1 at the first:
$$x=1$$
Then:
$$v= \begin{bmatrix} 1\\ 7.69 \end{bmatrix}$$
Now we just normalize it:

$$L = \sqrt{1^2 + 7.69^2} = \sqrt{1 + 59.1361} = \sqrt{60.1361} \approx 7.75$$

Divide both components by $L$**:

$$v_{normalized} = \begin{bmatrix} 1 / 7.75 \\ 7.69 / 7.75 \end{bmatrix} \approx \begin{bmatrix} 0.129 \\ 0.992 \end{bmatrix}$$
$$v_{normalized} = \begin{bmatrix} 0.129 \\ 0.992 \end{bmatrix}$$

So this is our first principal component.
What do we understand, we understand that every time it goes 0.129 times to the right, it will go 0.992 times up.  

Now we find the second:
$$\lambda_2$$
which is approximately:
$$v_2 = \begin{bmatrix} 0.992 \\ -0.129 \end{bmatrix}$$
So $\lambda_1$ and $\lambda_2$ are perpendicular.

We did all of this for understanding PCA in a easier way, this will answer:
"Why PCA keeps only PC1 (First Principal Component)"

Look again at the eigenvalues.

$$\lambda_1 = 350.2$$
$$\lambda_2 = 0.01$$

The first direction contains almost all the variation.
The second contains almost nothing.

So our graph suppose may look like:

![[eigenvector11.png|481]]

Ta-da~ So much pain for this? yup.
Hateful, but you can do nothing about it.
Anyway, we PC (Principal component) = Eigenvector. It is just a cute title.

Now we will start with SVD... Which is brain kicking in many wayssss, so let's try to don't drown in math. Ah, and most probably you understood 10% of what I spoke... so no worries, I'll stop at PCA for much more and solve real problems, and then you will understand it much better.

### 6. SVD (Singular value decomposition)

Imagine Airi Sezaki keeps a diary.
Every day she writes down how much affection happened between her and Hinako.

Since life is scary, we counterattack, so we make it even more scary:

| Day | Hugs | Smile Count | Blush Count | Head Pats | Gifts | Coffee Together | Walk |
| :-: | :--: | :---------: | :---------: | :-------: | :---: | :-------------: | :--: |
|  1  |  5   |     12      |      8      |     2     |   0   |        1        |  1   |
|  2  |  7   |     15      |     11      |     3     |   1   |        1        |  1   |
|  3  |  2   |      5      |      2      |     0     |   0   |        0        |  0   |
|  4  |  9   |     18      |     14      |     4     |   2   |        1        |  1   |
|  5  |  6   |     13      |      9      |     3     |   1   |        1        |  0   |
What do we notice now? The headache? Sensory overload? Both? - Exactly
But imagine if tomorrow we Airi will write another 10 features, and after an year she will have over 3 thousands of them...
Soooo, what with what we feed our model... since there is a feature that matter the most, right? Good luck finding it and I am sure you wouldn't use MAX Fabled 5.0 for such a question, right??
That why we will use SVD.

What SVD will do for us?

Imagine you have a complex audio track (your raw data) that has vocals, drums, bass, and guitar all mashed together into one sound file. It’s a mess of noise, and you can’t easily tell what's what.
SVD is like taking that "messy" track and running it through a specialized digital filter that perfectly isolates every instrument.

What do we understand from our table with data (The features)? 
We can imagine 2 dimensions... but there are 7 dimensions, lovely? Absolutely not!
There is no way!
That why we will use svd to find the most important features. SVD will look at the many dots and think:
"If I had to describe these five stars, what is the biggest direction they stretch?"

Suppose SVD Discovers that:
```
#1 Pattern

Hugs, Smile Counts, Blush Counts, Head pats grow togheter
```

and then presume SVD discovers that:
```
#2 Pattern

Gifts, Coffee Togheter, walk grow togheter too
```

And already those 7 features become:
```
Affection Pattern
Dating Pattern
```

without us writing those names. SVD found them.

So we may get instead of the axes `hugs, smiles`

The axes changes.... and we get  `Affection Pattern, Dating Pattern` as new axes

Because he may think that  Hugs, Smile Counts, Blush Counts, Head pats
are basically describing the SAME thing, so if we had:
```
Hugs = 7
Smile = 15
Blush = 11
Head Pats = 3
```

he may change to:
```
Affection = 9.8
```

Which is just one number, but it means a lot

But let's say that it check the second axes, where we have:

```
Coffee = 1 
Walk = 1 
Gift = 2
```

SVD will just say:
```
Date Activity = 1.7
```

So what we have now? Now we don't have 7 dimensions, we have 2 dimensions.

That why we will use this formula:
$$A = U\Sigma{V^T}$$

- $V^T$ (The Rotation - The "Directions"): This tells you the directions of the axes in your original data space. These are the "patterns" (like the wall, the door, the window).
    
- $\Sigma$ (The Scaling - The "Importance"): This is a diagonal matrix containing the singular values. Just like the eigenvalues we used earlier, these are the "volume faders." They tell you which patterns actually matter (large values) and which are just noise (tiny values).
    
- $U$ (The Projection - The "Coordinates"): This tells you how your original data points sit on those new "pure" axes.

So now we take again our table... and give an example:

| Day | Hugs | Smile Count | Blush Count | Head Pats | Gifts | Coffee Together | Walk |
| --: | ---: | ----------: | ----------: | --------: | ----: | --------------: | ---: |
|   1 |    5 |          12 |           8 |         2 |     0 |               1 |    1 |
|   2 |    7 |          15 |          11 |         3 |     1 |               1 |    1 |
|   3 |    2 |           5 |           2 |         0 |     0 |               0 |    0 |
|   4 |    9 |          18 |          14 |         4 |     2 |               1 |    1 |
|   5 |    6 |          13 |           9 |         3 |     1 |               1 |    0 |
Now in a matrix it will look like this:
$$A = \begin{bmatrix} 5 & 12 & 8 & 2 & 0 & 1 & 1 \\ 7 & 15 & 11 & 3 & 1 & 1 & 1 \\ 2 & 5 & 2 & 0 & 0 & 0 & 0 \\ 9 & 18 & 14 & 4 & 2 & 1 & 1 \\ 6 & 13 & 9 & 3 & 1 & 1 & 0 \end{bmatrix}$$
So we remember:

Rows = Days
Columns = Features.

Now we will start to look at the features and see if they have a toxic relationship (Actually, we will never count 10k features, but I show you an example, so I show how the SVD acts with the collumns):

| Hugs | Smile |
| ---: | ----: |
|    5 |    12 |
|    7 |    15 |
|    2 |     5 |
|    9 |    18 |
|    6 |    13 |
When hugs increase...
smiles also increase.

| Hugs | Blush |
| ---: | ----: |
|    5 |     8 |
|    7 |    11 |
|    2 |     2 |
|    9 |    14 |
|    6 |     9 |
Again...
almost identical behavior.

| Hugs | Head Pats |
| ---: | --------: |
|    5 |         2 |
|    7 |         3 |
|    2 |         0 |
|    9 |         4 |
|    6 |         3 |
Again...
same tendency.

So of course we think:
Hugs , Smile, Blush, Head Pats -> Way too toxic as relationship - meaning = They are really related.

So SVD thinks: "Can one hidden variable explain most of these?"

So imagine we invent: 
$$z$$
And what will it be equal too:
$$z=0.30(Hugs)+0.62(Smile)+0.55(Blush)+0.45(HeadPats)+⋯$$

No worries about the numbers, SVD finds the axes of maximum variance all by himself

So instead of remembering all the values of day one, we will compute one number
For Day 1
We will do:
$$z_1 = (w_{1,1} \times 5) + (w_{1,2} \times 12) + (w_{1,3} \times 8) + (w_{1,4} \times 2) + (w_{1,5} \times 0) + (w_{1,6} \times 1) + (w_{1,7} \times 1)$$
That actually is:
$$z=0.30(5)+0.62(12)+0.55(8)+0.45(2)+⋯$$
Suppose
$$z=15.8$$
So now day 1 becomes:
```
Day 1

Affection Score

15.8
```

So that a step ahead, because we have 1 number instead of 7
So that what a linear layer is! and it will be done to the rest of the days 

Now we look at the formula once again... and explain what it will do:

$$A = U\Sigma{V^T}$$

- $V^T$ - This will ask itself: "How should I mix the original columns?". For example, let's say it finds out this:
  $$V^T = \begin{bmatrix} 0.30 & 0.62 & 0.55 & 0.45 & 0.12 & 0.08 & 0.04 \\ \vdots & & & & & & \end{bmatrix}$$

When we can see: "hugs by 0.30," it means that hugs contributes by 0.30, while there are stronger features as 0.62 ....

Because 0.62 and 0.55 are much larger than 0.04, the math is literally screaming at the SVD: "Hey, pay way more attention to features 2 and 3 than to feature 7 when calculating this score."

We can understand that it means:
"Multiply Hugs by 0.30"
"Smile by 0.62"
"Blush by 0.55"
And so on...
as:

- **Day 1**:
    
    $(5 \times 0.30) + (12 \times 0.62) + (8 \times 0.55) + (2 \times 0.45) + (0 \times 0.12) + (1 \times 0.08) + (1 \times 0.04)$
    
    $= 1.5 + 7.44 + 4.4 + 0.9 + 0 + 0.08 + 0.04 = \mathbf{14.36}$
    
- **Day 2**:
    
    $(7 \times 0.30) + (15 \times 0.62) + (11 \times 0.55) + (3 \times 0.45) + (1 \times 0.12) + (1 \times 0.08) + (1 \times 0.04)$
    
    $= 2.1 + 9.3 + 6.05 + 1.35 + 0.12 + 0.08 + 0.04 = \mathbf{19.04}$
$...$


so now we have just one feature instead of 7.

$$\text{Projection} = \begin{bmatrix} 14.36 \\ 19.04 \\ 4.8 \\ 23.72 \\ 16.36 \end{bmatrix}$$

What do we even get?? We understand that:
On the new hidden axis discovered by SVD, Day 1 lies at coordinate 14.36.
SVD rotates the coordinate space - It looks at your data from an important angle, so the most important pattern line up with the data:

$$\begin{array}{|c|ccccccc|} \hline \text{Day} & \text{PC1} & \text{PC2} & \text{PC3} & \text{PC4} & \text{PC5} & \text{PC6} & \text{PC7} \\ \hline 1 & 14.36 & -0.91 & 0.52 & \dots & \dots & \dots & \dots \\ 2 & 19.04 & -0.30 & -0.44 & \dots & \dots & \dots & \dots \\ 3 & 4.80 & 1.72 & -0.10 & \dots & \dots & \dots & \dots \\ 4 & 23.72 & -1.15 & 0.81 & \dots & \dots & \dots & \dots \\ 5 & 16.36 & 0.64 & -0.79 & \dots & \dots & \dots & \dots \\ \hline \end{array}$$

We wrote only the first one, because PC1 explains already 92% Of the information, PC2 explains 6%, PC3 explains 1% and so on, so they are useless next to the 92%  and maybe the 6% information.

|**Day**|**PC1**|**PC2**|
|---|---|---|
|**1**|14.36|-0.91|
|**2**|19.04|-0.30|
|**3**|4.80|1.72|
|**4**|23.72|-1.15|
|**5**|16.36|0.64|

So the idea is:
```
Original Data

5 rows × 7 features

↓

SVD

↓

5 rows × 7 principal components

↓

Keep only the important ones

↓

5 rows × 2 principal components

or

5 rows × 1 principal component
```




After this we will check the error matrix, to see how well how much truth did we lose when throwing away the PC3, PC4....
We will use this idea:

$$A_{approx} = U_{reduced} \cdot \Sigma_{reduced} \cdot V^T_{reduced}$$

The Subtraction
The Error Matrix ($E$) is simply the original table minus the "sketch":

$$E = A - A_{approx}$$

If you look at the calculation for one cell (Day 1, Hugs):

$$\begin{bmatrix} 5 \end{bmatrix} - \begin{bmatrix} 4.8 \end{bmatrix} = \begin{bmatrix} 0.2 \end{bmatrix}$$

The Error Matrix Visualization
When you do this for the whole table, you get the "Residue" matrix:

$$E = \begin{bmatrix} 0.2 & 0.1 & -0.3 & \dots & \dots \\ -0.1 & 0.05 & 0.4 & \dots & \dots \\ \dots & \dots & \dots & \dots & \dots \\ \end{bmatrix}$$
The error matrix will be 5x7 like the original, this is just an illustration.
And the matrix shall have small numbers in it , because large numbers are an alert. Because it means that you throw something important away! (One of the PC was important, and you just discarded it like trash.)

In python it will look as simple as:

```python
import numpy as np

# That our data
A = np.array([
    [5, 12, 8, 2, 0, 1, 1],
    [7, 15, 11, 3, 1, 1, 1],
    [2, 5, 2, 0, 0, 0, 0],
    [9, 18, 14, 4, 2, 1, 1],
    [6, 13, 9, 3, 1, 1, 0]
])

# 1. Mean Centering (Crucial Step)
# SVD works as PCA only if data is centered at the origin.
# You will understand that later
A_centered = A - np.mean(A, axis=0)

# 2. Compute SVD
# U: Left singular vectors, S: Singular values, Vt: Right singular vectors (transposed)
U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

# 3. Project to K dimensions (let's keep 2)
k = 2
V_reduced = Vt[:k].T  # Take top 2 rows, transpose to get components
A_reduced = A_centered @ V_reduced

print("Compressed Data (First 2 Components):\n", A_reduced)

"""
Output:

Compressed Data (First 2 Components):
 [[ 1.30342012 -0.96193574]
 [-3.54185665 -0.28281232]
 [11.2085746   0.34521945]
 [-8.38713342  0.3963111 ]
 [-0.58300465  0.50321752]]

"""
# We got such a different outcome, because the computed version does additional steps that I didn't show, but I did them for you to understand how svd actually works, and why we use it
```

After this pain, we will move to PCA, but no worries, after PCA, I'll solve many problems, showing you how it works and why we use it.

### 7. PCA (Principal Component Analysis)

What is even PCA?? Why did we pass such a brain tiring idea before it?
We did that, just to understand what the PCA actually does, because without the PCA the spectral graph will look like Chinese! (Or if you are Chinese it may look like Russian! if you are both, idk then.)

The PCA asks:
"If I had to keep only the most important information, which directions should I keep?"

In the end I'll make a list of what we learned and explain what to use when and stuff.

Let's take Airi's perfect score again:

| Day | Scroll Speed | PERFECT Hits |
| :-: | :----------: | :----------: |
|  1  |      3       |      52      |
|  2  |      5       |      67      |
|  3  |      6       |      74      |
|  4  |      8       |      91      |
|  5  |      10      |     106      |

we do the same old steps, finding the mean of both:

- $\mu_{SS}$ $\approx 6.4$
- $\mu_{PH}$ $\approx 78$

Now we subtract everything by it:

| **Day** | **Scroll Speed (x)** | **Hits (y)** | **Avg Speed ($SS - \mu_{SS}$)** | **Avg Hits ($PH - \mu_{PH}$)** |
| ------- | -------------------- | ------------ | ------------------------------- | ------------------------------ |
| 1       | 3                    | 52           | **-3.4**                        | **-26**                        |
| 2       | 5                    | 67           | **-1.4**                        | **-11**                        |
| 3       | 6                    | 74           | **-0.4**                        | **-4**                         |
| 4       | 8                    | 91           | **1.6**                         | **13**                         |
| 5       | 10                   | 106          | **3.6**                         | **28**                         |

Now we find their toxic relationship, which will be:
$$\Sigma = \begin{bmatrix} 5.84 & 44.8 \\ 44.8 & 344.4 \end{bmatrix}$$
And now we will find the directions that explain our data the best, together with how important each direction is.:

Anyway, the eigenvalues are the one from before, since the example is the same:

$$\lambda_1 = 350.228$$
$$\lambda_2 = 0.012$$

What do we understand?

Total variance = $\lambda_1 + \lambda_2 = 350.24$
$$\frac{\lambda_1}{Total variance} = 99.9\%$$
$$\frac{\lambda_2}{Total variance} = 0.01\%$$


The first eigenvector explains about 99.9% of the variance in the data, while the second explains almost nothing

The eigenvectors:
$$v_1 = \begin{bmatrix} 0.129 \\ 0.992 \end{bmatrix}$$
$$v_2 = \begin{bmatrix} -0.992 \\ 0.129 \end{bmatrix}$$
We do a quick check now:
($0.129 \times -0.992 + 0.992 \times 0.129 \approx 0$)
Their dot product is zero. That means  that they are perpendicular.
That means that PC1 and PC2 are indeed independent and PC2 captures almost no information.

Again, so we remember, In PCA the eigenvectors of the covariance matrix are called PC (Principal components)

After knowing this... the magic comes:
We will change the axes. We will not have our same x (Which measures the 'Scroll speed') or y (Which measures the 'PERFECT Hits') we will have our new axes: PC1 -> That the main axes (The signal), let's say it will just go above our Data, and then PC2 -> We already said that this perpendicular and useless (The noise), so we can see it as noise. If we delete PC2, we lose a tiny amount of information. In our example that loss is only about 0.01%, so for practical purposes we can pretend we lost almost nothing. We just stop measuring the "width" of the cloud (The cloud of data we have), because the width is just noise. We keep the "length" of the cloud, which is where all the actual information is. 
In other examples the second PC2 can be really important and etc.


Soooo, the SVD was clearly the most brain traumatizing. That why we will solve problems with it. So now we will make a recap:

___

Variance → Use it when you want to know how much one feature spreads around its mean (around the average). High variance = values are scattered. Low variance = values stay close together.

Covariance → Use it when you want to know if two features move together. Positive = both increase together. Negative = one increases while the other decreases. Near zero = little linear relationship.

Covariance Matrix → Use it when you have many features and want every variance and every covariance in one object. This becomes the starting point for PCA.

Direction (Vector) → Use it whenever you need a path or orientation in space. Every arrow points somewhere and has a length.

Eigenvector → Use it when you want to find the most important directions hidden inside a matrix. For a covariance matrix, these are the directions where the data naturally stretches.

Eigenvalue → Use it to measure how important an eigenvector is. Large eigenvalue = lots of information (variance). Small eigenvalue = mostly noise.

Principal Component (PC) → This is simply an eigenvector of the covariance matrix, but with a fancier name because we're doing PCA. PC1 explains the most variance, PC2 explains the second most, and so on.

Projection → Use it when you want to know where each sample lies on a principal component. Instead of describing Airi with 7 original features, you describe her by a coordinate on PC1, PC2, etc.

SVD (Singular Value Decomposition) → Use it when you want to decompose a data matrix itself into hidden patterns. It works even when computing a covariance matrix is inconvenient or expensive. It is the workhorse behind many ML algorithms.

PCA (Principal Component Analysis) → Use it when you want to reduce dimensionality while keeping as much information as possible. It rotates the coordinate system so the important directions become the new axes, then throws away the unimportant ones.

___

Okay, so imagine this, Airi is working under a medical company, and the boss hands her the data... but she doesn't has just 1 feature or 2... she has 8k of them... so what to do?

1. Firstly she cheeks the variance, so if a feature is like:

```
Hospital Country

Italy
Italy
Italy
Italy
Italy
Italy
Italy
Italy
```

it sucks, variance = 0, so delete

2. Now check the covariance

Let's say she found out that:

```
Average Lung Density

and

Average Tissue Density
```

Are moving together, so there is no sense to keep both.

She still has 7800 features instead of 8000 - still awfull

3. She computes the covariance matrix.

Not because she likes matrices.
Because now she wants to ask: "What are the hidden directions inside all these measurements?"

4. Then she will find the eigenvector and eigenvalue

Here the magic happens, because the first eigenvector may discover:

```
overall tumor score
```

Nobody wrote that feature, yet math did.

The second eigenvector may find:

```
inflamation patterns
```

Math found that too, and so on...

Now instead of the 8000 features she has:

```
PC1 = Tumor Severity

PC2 = Inflammation

PC3 = Body Size

PC4 = Scanner Effect

...
```

Much better already

Now let's say the eigenvalues are like:

```
PC1
65%

PC2
20%

PC3
8%

PC4
4%

Everything else

3%
```

Then we discard the other and get what is relevant.

And get:

```
Patient A

PC1 = 3.2

PC2 = -1.4

PC3 = 0.7

PC4 = 2.1
```

4 numbers instead of of 8000 features

much more convenient and almost same data. 

That was a small example. But now you will proceed and cry less, cuz this was the hardest part of all actually.

Now I will go and recreate the next month (Yeah, I did it all backwards, firstly the Pyspark, then SQL, then graph, and in the end this one)

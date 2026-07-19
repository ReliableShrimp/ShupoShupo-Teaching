# Month 1 — ML Basics & Just Enough Math

Hello, explorer.

I don't know where you started — maybe this is your first topic? MAYBE?? I never numbered them, so honestly I don't even know where you'd begin, but I'll start from here anyway. I do have some lovely guesses about how you ended up here, though:

**The most probable one:**

- You're an interviewer, checking this out to see if it's really worth hiring people in the big 2026/2027 era.
- You like the idea of automating half of everything and getting hated like a tax collector because you automated somebody's job. No excuses like _"I'm just making jobs easier!"_
- You saw a video on TikTok (I don't even post videos — maybe I'll get desperate enough to start eventually).

**Also highly possible:**

- You love IT and math, so you figured there's no better field to connect the two. (And yeah, that's exactly why I'll teach you to write hundreds of lines of code using pure math, only to later trade all that effort for the sad reality of `scikit-learn` and `pyspark.ml`. But hey — at least you'll actually understand the bug instead of just hitting a `ValueError: Found input variables with inconsistent numbers of samples` and burning 2.5 million tokens on Claude hoping it fixes itself.)
- You found a secure, satisfying career path, because the further we go into the future, the more [people will "boo" us when we talk about AI](https://www.youtube.com/watch?v=tNH43a1EI7s&t=2s). But remember: they hate us 'cause they ain't us. So keep studying.

**Too rare to be likely:**

- You just like learning.

Sadly, no vibecoding here — we're going to mentally wrestle some demons, so let's start.

This first month is all about ML basics and a bit of math. Even if you already know chain rules, partial derivatives, and gradient descent, I'll still explain them — maybe this time they'll tattoo themselves into your brain.

We'll start the easy way... Linear Regression, because life is easy:

## Table of Contents

# Table of Contents
1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. XGBoost
6. PCA — Variance → Covariance → Covariance Matrix → Direction → Eigenvectors/values → PCA

---

## 1. Linear Regression

Why do we use linear regression? Why do we even need it?

Luckily for you, I won't say something like _"linear regression is a fundamental mathematical tool to establish a direct, quantitative..."_ — because who understands that besides a senior engineer and an AI?

So first, I'll give you an example, explain why we use it, and then compute it in Python.

But before that — we'll use numpy for a lot of this, so let's start with the basic idea of numpy. You'll see `np.array` a lot — think of it as a Python list, but built only for math operations. You'll see things like:

`np.array([1, 2, 3, 4])` ← this is a 1D vector, nothing more:

$$\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}$$

`np.array([[1, 2], [3, 4]])` ← this is already 2D:

$$ \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}$$

and the more you add, the more columns or rows you get:

`np.array([[1, 2, 3], [4, 5, 6]])` ← still 2D:

$$ \begin{bmatrix} 1 & 2 & 3 \\\\ 4 & 5 & 6 \end{bmatrix} $$

and the last one:

`np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])` ← still 2D, because to get 3D we'd need a tensor, and we don't work with those (yet):

$$ \begin{bmatrix} 1 & 2 & 3 \\\\ 4 & 5 & 6 \\\\ 7 & 8 & 9 \end{bmatrix} $$

### Example 1: Red lights

Imagine this: you're driving home from work, and you keep hitting red lights. You want to predict how long they'll cost you, so you write down what you've observed so far:

1 red light → 6 minutes 2 red lights → 11 minutes 3 red lights → 12 minutes 4 red lights → 17 minutes 5 red lights → 20 minutes 6 red lights → 23 minutes

You stare at the screen and think: "that's not even close to how a real red light works" — maybe, but we care less about that right now. The only thing we care about is predicting how long we'll wait at red lights we _don't_ have data for — say, 7 red lights? 10? 25?

That's why we use linear regression: to predict the next red lights (this is roughly how Google Maps works too, among many other factors).

So how do we predict the red lights? First, we write our `x` and `y` in Python:

```python
import numpy as np

# A numpy array with 6 rows and 1 column
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])
```

The matrix `X` will always be written with more rows; later I'll explain what to do once there are more features.

<img width="640" height="480" alt="Time to wait at red light1" src="https://github.com/user-attachments/assets/c2a7f099-929e-4898-a153-49cf0985f7db" />

Now that we have `X` and `y`, we need a first guess — the line that best passes through the dots. `w` and `b` will help with this, even though we start them at 0. Same for the learning rate (`lr`), which we'll usually set to 0.01 (when the data is small, you can raise it):

```python
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])

w = 0
b = 0
lr = 0.01
# w starts at 0 because this X only has one feature (don't worry about more features yet)
```

Now we write the learning algorithm (gradient descent). It figures out the exact values `w` and `b` should settle on.

```python
import numpy as np

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([[6], [11], [12], [17], [20], [23]])

w = 0.0  # weight
b = 0.0  # bias
lr = 0.01  # learning rate

for epoch in range(1500):
    # 1500 loops so gradient descent can nudge w and b into place.
    # "epoch" is just a fancy placeholder name — you could write "i" instead.
    prediction = X * w + b    # our beautiful linear expression
    error = prediction - y    # by how far the model is missing the mark
    loss = np.mean(error**2)  # the loss of the model — explained below

    nabla_w = (1/len(X)) * np.sum(error * X)
    nabla_b = (1/len(X)) * np.sum(error)

    w = w - nabla_w * lr
    b = b - nabla_b * lr

    # Every 300 loops, show the result
    if epoch % 300 == 0:
        print(f"Step {epoch}: Loss = {loss} | w = {w} | b = {b}")
        print("")

print("----FINAL RESULT----")
print("")
print(f"The loss of the model is: {loss}")
print(f"The weight of the model is: {w}")
print(f"The bias of the model is: {b}")
```

Before showing the result, let's unpack the jargon.

First, `np.mean()` — you'll see it a couple of times here, and if you don't get it, the rest will look like Chinese to you. `np.mean()` just takes a bunch of numbers and gives you the average, nothing more. So `np.mean([2, 4, 6])` adds them up (2+4+6=12) and divides by how many numbers there were (3), giving you `4`. That's the whole trick.

`for epoch in range(1500):` — we loop 1500 times so the model can make its loss and guess better each round (don't push it to 10000+ for a toy dataset like this — the model starts tweaking the millionth decimal for no real benefit).

`prediction = X * w + b` — this is the line equation we're building, remember `y = mx + b` from school? Same thing, different letters. At the start, `w = 0` and `b = 0`, so on loop 1, the prediction for _every_ `X` value is literally 0 (anything times 0 plus 0 is still 0). Bad prediction, I know, but it's the starting point — we have to start somewhere. If you're bothered by starting at 0, we'll cover Xavier (Glorot) and Kaiming initialization much later (boo hoo to the poor soul who can't endure an easy start).

`error = prediction - y` — checks "how far off was my guess from the real answer." If `prediction` is 0 and the real `y` is 6, the error is `0 - 6 = -6`, meaning we way under-guessed. This happens for all 6 rows at once, because numpy is smart like that — it doesn't make you loop manually, it operates on the whole array in one shot.

`loss = np.mean(error**2)` — we square every error first (so negatives become positive, and big errors get punished harder than small ones), then average all of them with `np.mean`. This single number, `loss`, tells you "how bad is my model doing right now" — the smaller, the better. If you see something like `loss = 500`, it might be time to double-check your data or predictions, because I could probably guess better blind. This kind of loss is called **Mean Squared Error (MSE)**.

Now the scary part — `nabla_w` and `nabla_b`. Don't panic: "nabla" is just a fancy Greek symbol people use to mean "gradient," which just means "which direction, and by how much, should I nudge `w` and `b` to make the loss smaller." Think of standing on a hill in the fog — you can't see the bottom, but you can feel which way the ground slopes under your feet, so you take a small step in that direction. That's exactly what these two lines do.

`nabla_w = (1/len(X)) * np.sum(error * X)` — this multiplies the error by the matching `X` value for each row (so bigger `X` values get "blamed" more when there's error), sums all of that up, then divides once by `len(X)` (how many rows we have — 6 here). That gives us "how much, and in which direction, to adjust `w`."

`nabla_b = (1/len(X)) * np.sum(error)` — same idea, but simpler: since the bias `b` isn't multiplied by `X`, we just sum the error directly and divide once by `len(X)`.

`w = w - nabla_w * lr` and `b = b - nabla_b * lr` — this is where the actual "learning" happens. We nudge `w` a little bit in the direction that reduces the loss, and how big that nudge is depends on `lr` (the learning rate, 0.01 here). If `lr` were bigger, like 0.5, the steps would be huge and reckless — like running (or jumping) straight down the hill instead of walking. If `lr` were tiny, like 0.00001, it's an eternal wait — like the wind nudging you down the hill so slowly you never arrive. 0.01 is usually a decent middle ground for small data like this.

This block then repeats 1500 times, each round getting a slightly better `w` and `b`, and a slightly smaller loss — the model slowly "waking up" and realizing "oh wait, the pattern is more like +3ish minutes per red light, plus some starting offset."

Running it for real gives us:

```
Step 0: Loss = 253.166667 | w = 0.616667 | b = 0.148333
Step 300: Loss = 0.866124  | w = 3.643565 | b = 1.845942
Step 600: Loss = 0.653026  | w = 3.516713 | b = 2.389024
Step 900: Loss = 0.581796  | w = 3.443372 | b = 2.703008
Step 1200: Loss = 0.557986 | w = 3.400970 | b = 2.884539

----FINAL RESULT----
The loss of the model is: 0.550042
The weight of the model is: 3.376517
The bias of the model is: 2.989229
```

The loss isn't huge, but it's not settled yet either — so we run the same training loop again, continuing from these weights instead of starting over:

```
Step 0: Loss = 0.550028 | w = 3.376455 | b = 2.989492
Step 300: Loss = 0.547367 | w = 3.362282 | b = 3.050171
Step 600: Loss = 0.546478 | w = 3.354088 | b = 3.085253
Step 900: Loss = 0.546181 | w = 3.349350 | b = 3.105535
Step 1200: Loss = 0.546082 | w = 3.346611 | b = 3.117262

----FINAL RESULT----
The loss of the model is: 0.546048
The weight of the model is: 3.345031
The bias of the model is: 3.124025
```
<img width="640" height="480" alt="End of Time to wait at red light1" src="https://github.com/user-attachments/assets/e9f4754c-a150-4178-bf22-97ee193201f4" />

The loss fell by a small amount. Let's see how the model actually guesses:

- 4 red lights → **16.50 minutes** (actual value in the dataset: 17)
- 5 red lights → **19.85 minutes** (actual value in the dataset: 20)
- 6 red lights → **23.19 minutes** (actual value in the dataset: 23)
- 10 red lights → **36.57 minutes** (brand-new, unseen value! probably off, but a fair try)

That's how linear regression works. Another example: the taxi driver.


### Example 2: Taxi driver (two features)

We know just this:

2 km, 1 red light → 9 minutes 3 km, 2 red lights → 14 minutes 4 km, 1 red light → 15 minutes 5 km, 3 red lights → 22 minutes 6 km, 2 red lights → 24 minutes 8 km, 4 red lights → 34 minutes

Here we can see distance and red-light count together — the first point in the corner is 2 km / 1 red light, the last one is 8 km / 4 red lights.


And here's the total travel time for each point.

<img width="800" height="600" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/3a082658-db31-447c-a52f-bbe54e2720f4" />

And as usual, we stare blankly at the screen, wondering if we'll ever get home, since we still have 9 km and 5 red lights to go... we grab our beloved "I Love Your Cruddy" manga and start reading instead. But then we remember the model can help us predict it — so what do we do? Keep reading it, obviously, and once we're done ugly-crying, we finally write the algorithm:

```python
import numpy as np

X = np.array([[2, 1], [3, 2], [4, 1], [5, 3], [6, 2], [8, 4]])
y = np.array([[9], [14], [15], [22], [24], [34]])

w = np.zeros((2, 1))
# Same idea as w = np.array([0, 0]), just shaped as a column vector (2, 1)
# instead of a flat array (2,) — that column shape is what makes X @ w line up.
# You can always write w = np.zeros((number_of_features, 1)).
b = 0.0
lr = 0.01

for epoch in range(1500):
    prediction = X @ w + b  # matrix multiplication, since we now have 2+ features (works for 10k+ too)
    error = prediction - y

    loss = np.mean(error**2)

    nabla_w = (1/len(X)) * (X.T @ error)  # matrix form of the same gradient, shaped to match w
    nabla_b = (1/len(X)) * np.sum(error)

    w = w - nabla_w * lr
    b = b - nabla_b * lr

    if epoch % 500 == 0:
        print(f"Loop {epoch}, loss: {loss}, w: {w}, b: {b}")
        print("")

print("----FINAL RESULT----")
print(f"The loss of the model is: {loss}")
print(f"The weight of the model is: {w}")
print(f"The bias of the model is: {b}")

"""
Output:

Loop 0, loss: 453.0, w: [[1.07666667]
 [0.50333333]], b: 0.19666666666666666

Loop 500, loss: 0.057043825490346554, w: [[3.28721594]
 [1.76879512]], b: 0.5077036270753326

Loop 1000, loss: 0.0551605963821002, w: [[3.27619854]
 [1.81110907]], b: 0.45998788483373343

----FINAL RESULT----
The loss of the model is: 0.054914888850511386
The weight of the model is: [[3.27602155]
 [1.82016599]]
The bias of the model is: 0.4378494071196286
"""
```

The results are good, so we run it again with the new weight and bias plugged in as the starting point:

```
Loop 0, loss: 0.054915,   w: [3.27602341 1.82017369], b: 0.437820
Loop 500, loss: 0.054866, w: [3.27701465 1.82216066], b: 0.427331
Loop 1000, loss: 0.054856, w: [3.27771669 1.82261747], b: 0.422327

----FINAL RESULT----
The loss of the model is: 0.054853
The weight of the model is: [3.2781008  1.82273217]
The bias of the model is: 0.419935
```

Our model predicted:

<img width="1000" height="800" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/50fd40b4-4fc1-4ac9-8915-d1483f3daa76" />


Let's guess some minutes:

- 2 km, 1 red light → **8.80 minutes** (actual value: 9)
- 5 km, 3 red lights → **22.28 minutes** (actual value: 22)
- 6 km, 2 red lights → **23.73 minutes** (actual value: 24)
- 8 km, 4 red lights → **33.94 minutes** (actual value: 34)
- 9 km, 5 red lights → **39.04 minutes** (brand-new, unseen trip!)

That's how linear regression works. But sadly, in a real job, we usually reach for libraries like `scikit-learn`, which already contain highly optimized implementations of Linear Regression. Instead of writing dozens of lines for gradient descent, gradients, weight updates, and loss tracking ourselves, we just create the model, fit it to our data, and let the library handle the math.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Same idea as the taxi driver example
X = np.array([[2, 1], [3, 2], [4, 1], [5, 3], [6, 2], [8, 4]])

# Labels
y = np.array([9, 14, 15, 22, 24, 34])  # 1D now, since scikit-learn expects that shape here

model = LinearRegression()
model.fit(X, y)

print(f"Weights: {model.coef_}")       # coefficients = weights
print(f"Bias: {model.intercept_}")     # intercept = bias

prediction = model.predict([[9, 5]])
print(f"\nPrediction for 9 km and 5 red lights: {prediction[0]:.2f} minutes")

"""
Weights: [3.27848101 1.82278481]
Bias: 0.4177215189873458

Prediction for 9 km and 5 red lights: 39.04 minutes
"""
```

Since — I hope — you got something out of this, let's move to logistic regression. Because life is long, depending on how much you expect to live.

---

## 2. Logistic Regression

So far we've used Linear Regression. Its job was simple: predict any real number.

For example: 4 red lights → 17 minutes 8 km taxi trip → 34 minutes House size → \$250,000

But now imagine a different kind of question: will this email be Spam / Not Spam? Will this patient have cancer — Yes / No? We only need two answers (binary logic) and, ideally, a probability of it being one or the other. That's where **Logistic Regression** comes in.

Why not just use Linear Regression for this? Because Linear Regression can output negative numbers and numbers above 1 — and you can't have negative odds, or 101%+ of something. We want a real probability, and probabilities are locked into a strict range by reality itself:

- 0.00: 0% chance (impossible)
- 0.50: 50% chance (a perfect coin flip)
- 1.00: 100% chance (guaranteed)

Anything above 1 or below 0 breaks the math. That's why we need something that takes the raw output of our linear equation and squashes it into a probability. Who can do that? **Blaise Pascal** and **Pierre de Fermat**? Maybe — but today we'll use the **Sigmoid function**.

Think of the sigmoid as a visual "S-curve" acting like a trash compactor for numbers. No matter how massive or microscopic your input `z` is, the output can't escape the boundaries of 0 and 1:

<img width="548" height="364" alt="image" src="https://github.com/user-attachments/assets/a92c5cff-2952-43e6-9d6d-bc4f96aef653" />

The formula:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **Massive positive scores:** if the model is extremely confident ($z = 5000$), $e^{-5000}$ is effectively 0. The formula becomes $1 / (1 + 0) = 1.00$ (guaranteed).
- **Massive negative scores:** if the model is fully convinced against it ($z = -500$), $e^{500}$ becomes a huge number. The formula collapses toward $1/\text{infinity} \approx 0.00$ (impossible).
- **The dead center:** if the model is completely unsure and outputs $z = 0$, $e^0 = 1$, so the formula becomes $1/(1+1) = 0.50$.

In Python:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

That's the main new idea — the only other thing that changes is the loss function, which is no longer MSE but **BCE (Binary Cross Entropy)**.

$$J(w, b) = - \frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

Don't get too scared — you'll basically never solve this by hand, unless every machine on Earth breaks and you're stuck with a pen and a notebook. I'd put the odds of that at roughly 0. What matters is that when you take the derivative of this messy `J(w, b)` with respect to `w` (a bit of partial-derivative work), the calculus matches up with the sigmoid perfectly — all the log terms and painful fractions cancel out, leaving just:

$$\frac{\partial J}{\partial w} = \frac{1}{n} X^T (A - Y)$$

In words: take our predictions (usually called `A`), subtract the true labels (`y`), and multiply by the features (`X`). In code:

```python
import numpy as np

def safe_log(y_true, p_pred):
    epsilon = 1e-15
    p_pred = np.clip(p_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
```

Why write it like this? To keep our code from crashing into `NaN` (Not a Number) or `Inf` (Infinity). `epsilon = 0.000000000000001`, so if something like that happens, we notice immediately that something's wrong.

Why `np.clip`? Because `p_pred = np.clip(p_pred, epsilon, 1 - epsilon)` pushes any value that's too small up to the minimum allowed, and any value that's too big down to the maximum allowed (0.9999... i.e. 99.999...%). The rest is just the formula, written in Python.

Our graph looks like this:

<img width="900" height="600" alt="image" src="https://github.com/user-attachments/assets/455db790-08af-439c-9567-b77ba5a7ac1a" />

```python
import numpy as np

# Features: [Body Temperature, Cough Severity]
X = np.array([
    [36.5, 2],  # Patient 1  (Healthy)
    [38.2, 7],  # Patient 2  (Sick)
    [36.8, 1],  # Patient 3  (Healthy)
    [39.1, 9],  # Patient 4  (Sick)
    [37.0, 3],  # Patient 5  (Healthy)
    [38.5, 6],  # Patient 6  (Sick)
    [36.2, 2],  # Patient 7  (Healthy)
    [37.8, 5],  # Patient 8  (Sick)
    [39.5, 8],  # Patient 9  (Sick)
    [36.7, 4],  # Patient 10 (Healthy)
])

y_true = np.array([[0], [1], [0], [1], [0], [1], [0], [1], [1], [0]])
# 1 = sick, 0 = healthy

w = np.zeros((2, 1))
b = 0.0
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

Optimized bias: -0.04131919500891592
"""
```

Now we update the weight and bias and train the same loop again:

```
--- Training Complete ---

The current loss: 0.0624768838569563

Optimized weights:
[[-0.30891162]
 [ 2.57792074]]

Optimized bias: -0.05786375733604678
```

Model prediction:

<img width="1000" height="700" alt="image" src="https://github.com/user-attachments/assets/dd1cb549-e7a8-414e-a367-d626a27de41f" />

What do we understand? Every dot above the line is "sick," every dot below is "healthy."

Some tries, using the final weights and bias:

- Temp = 36.5°C, Cough = 2 → **Healthy** (99.79% confidence)
- Temp = 38.4°C, Cough = 7 → **Sick** (99.78% confidence)
- Temp = 37.2°C, Cough = 4 → **Healthy** (77.52% confidence)
- Temp = 39.0°C, Cough = 9 → **Sick** (100.00% confidence)
- Temp = 36.4°C, Cough = 1 → **Healthy** (99.98% confidence)

Now the industry-level version:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [36.5, 2], [38.2, 7], [36.8, 1], [39.1, 9], [37.0, 3],
    [38.5, 6], [36.2, 2], [37.8, 5], [39.5, 8], [36.7, 4],
])

# Labels: 1 = Sick, 0 = Healthy
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])

model = LogisticRegression()
model.fit(X, y_true)

print(f"Model weight is: {model.coef_}")
print(f"Model bias is: {model.intercept_}")

prediction = model.predict([[38.1, 6]])
print(f"Prediction for a patient with 38.1 fever and cough severity 6: {prediction[0]:.2f}")

"""
Output:

Model weight is: [[0.62351466 1.04001144]]
Model bias is: [-28.05834858]
Prediction for a patient with 38.1 fever and cough severity 6: 1.00
"""
```

We'll continue with decision trees now that logistic regression clicked. But as usual, the creator of this repo will complain about something like: why do I even have to learn?? I'm sooo tired, every day, x/7 I study, I don't even play games — but hey, at least I keep re-reading "I Love Your Cruddy" over and over.

---

## 3. Decision Tree

Here the idea moves much more into something human-readable and less algebraic. A decision tree is a game of 20 questions where we build a tree of decisions humans can actually read.

Imagine your boss asks you to make something that isn't a pile of algorithms, but gives a plain answer to the questions people actually have. A doctor will never be thrilled by a logistic regression model that just says "trust me, plug in the numbers and I'll predict the result."

If we hand our 10-patient dataset to a Decision Tree, it doesn't look for weights (`w`) or a bias (`b`) — instead it follows logic like:

```
Is Body Temperature < 37.5°C?
    Yes: Classify as Healthy (Green).
    No: Go to the next question.

Is Cough Severity < 5?
    Yes: Classify as Healthy (Green).
    No: Classify as Sick (Red).
```

But before computing this, let's understand it with a small example:

|**Patient**|**Body Temperature**|**Status**|
|---|---|---|
|A|36.5|0 (Healthy)|
|B|37.0|0 (Healthy)|
|C|38.0|1 (Sick)|
|D|39.0|1 (Sick)|

We have 2 healthy patients and 2 sick patients. Now we look at how "filthy" (messy) this dataset is, using **Gini impurity**, which ranges from 0 to 0.5.

What is Gini impurity? Imagine you close your eyes, reach into a bag, and grab one animal. The question is: "how difficult is it to correctly guess what I picked?" That's exactly what Gini impurity measures.

The formula:

$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$

Square each probability $p_i^2$, add them together, then subtract that total from 1.

$$p_i = \frac{n_i}{N}$$

where $p_i$ is the probability of class $i$, $n_i$ is the count of class $i$, and $N$ is the total number of items.

In our table: 2 healthy, 2 sick, 4 total.

$$I_G(p) = 1 - \left(\left(\tfrac{1}{2}\right)^2 + \left(\tfrac{1}{2}\right)^2\right) = 1 - (0.25 + 0.25) = 0.5$$

A Gini of 0.5 here is the maximum possible "cruddiness" for this list — the two classes are perfectly mixed, so let's help it out.

Now we search for the best split — for our example, a threshold on body temperature. How do we find it? By looking at the midpoints between consecutive sorted values:

|**Patient**|**Body Temperature**|**Status**|
|---|---|---|
|A|36.5|0 (Healthy)|
|B|37.0|0 (Healthy)|
|C|38.0|1 (Sick)|
|D|39.0|1 (Sick)|

- 36.5 → 37.0: midpoint = 36.75 (same class — skip)
- 37.0 → 38.0: midpoint = 37.5 (different classes — keep!)
- 38.0 → 39.0: midpoint = 38.5 (same class — skip)

So the only real candidate threshold is **37.5**:

```
Is body temperature < 37.5?
    Yes? → Healthy
    No?  → Sick
```

The general structure of a tree:

```
Root
  └── Internal nodes
        └── Leaf nodes
```

<img width="590" height="680" alt="image" src="https://github.com/user-attachments/assets/0e25b11d-2c36-4814-ab86-20407653ce67" />


Now for a bigger example, so the pattern really sinks in:

|**Individual**|**Introversion Score (1–100)**|**Prefers Remote Work (0/1)**|
|---|---|---|
|A|22|0|
|B|35|0|
|C|48|1|
|D|50|0|
|E|63|1|
|F|71|1|
|G|78|0|
|H|85|1|
|I|92|1|
|J|97|1|

4 people prefer remote work... wait, let's actually count: 6 prefer it (C, E, F, H, I, J), 4 don't (A, B, D, G).

Candidate thresholds (midpoints between differently-labeled neighbors):

- 22 → 35: same class → skip
- 35 → 48: different → 41.5
- 48 → 50: different → 49.0
- 50 → 63: different → 56.5
- 63 → 71: same class → skip
- 71 → 78: different → 74.5
- 78 → 85: different → 81.5
- the rest are same-class

Let's test `X < 41.5`:

- Left: `[A, B] = [0, 0]` → Gini = 0 (pure!)
- Right: `[C, D, E, F, G, H, I, J] = [1, 0, 1, 1, 0, 1, 1, 1]` → 6 "yes", 2 "no"

$$p(1) = \frac{6}{8} = 0.75 \qquad p(0) = \frac{2}{8} = 0.25$$

$$I_G(p) = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375$$

Weighted Gini for this split:

$$\text{Weighted Gini} = \left(\frac{n_{\text{left}}}{N} \times G_{\text{left}}\right) + \left(\frac{n_{\text{right}}}{N} \times G_{\text{right}}\right)$$

$$\text{Weighted Gini} = \left(\frac{2}{10} \times 0\right) + \left(\frac{8}{10} \times 0.375\right) = 0.30$$

So threshold 41.5 gives a weighted Gini of 30% — pretty good! Checking all the candidate thresholds:

- Threshold 41.5: Weighted Gini = 0.30
- Threshold 49.0: Weighted Gini = 0.42
- Threshold 56.5: Weighted Gini = 0.32
- Threshold 74.5: Weighted Gini = 0.45
- Threshold 81.5: Weighted Gini = 0.34

41.5 wins. It's not a good idea to keep splitting through every remaining midpoint, though — we'd slice the data too many times and overfit.

**What is overfitting?** Imagine studying for a tough entrance exam by memorizing the exact answers to the 20 practice questions instead of the underlying concepts. You retake the practice quiz and score 100% — genius! But the real exam uses different numbers, and you fail miserably. You didn't learn the math; you memorized a specific history.

Let's compute all of this in Python, step by step (a single 100-line block would be too hard to follow at once). First the table and the Gini function:

```python
import numpy as np

X = np.array([[22], [35], [48], [50], [63], [71], [78], [85], [92], [97]])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

def calculate_gini(labels):
    n_samples = len(labels)
    if n_samples == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    return 1.0 - np.sum((counts / n_samples) ** 2)
```

- `n_samples = len(labels)` — `labels` is our `y`, so this is the total count in that group (10 here). We need it for the probabilities.
- `if n_samples == 0: return 0` — a safety check: if `y` were ever empty, we return 0 instead of blowing up.
- `_, counts = np.unique(labels, return_counts=True)` — this counts each class for us:
    - `np.unique(labels)` returns the unique classes, e.g. `[0, 1]`.
    - With `return_counts=True`, it also returns how many of each: `[4, 6]`.
    - We only need the counts, so the underscore (`_`) throws away the class-labels array and keeps just `[4, 6]`.
- `1.0 - np.sum((counts / n_samples) ** 2)` — the Gini formula, computed in Python: turn counts into probabilities, square them, sum them, subtract from 1: $1 - (0.16 + 0.36) = 0.48$.

Now the tree-builder:

```python
def build_tree(X, y, depth=0, max_depth=2):
    # Base case: stop if pure, or depth limit reached. Return the majority vote.
    if calculate_gini(y) == 0 or depth >= max_depth:
        return int(np.round(np.mean(y)))

    best_gini, best_thresh = 1.0, None

    # Generate midpoints from the sorted unique values
    sorted_vals = np.sort(np.unique(X[:, 0]))
    midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

    # Run the tournament loop
    for thresh in midpoints:
        left_y = y[X[:, 0] < thresh]
        right_y = y[X[:, 0] >= thresh]

        weighted_gini = (len(left_y) * calculate_gini(left_y) + len(right_y) * calculate_gini(right_y)) / len(y)

        if weighted_gini < best_gini:
            best_gini, best_thresh = weighted_gini, thresh

    # Split using the winning threshold and recurse into both children
    left_mask = X[:, 0] < best_thresh
    return {
        f"X < {best_thresh}": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
        "else": build_tree(X[~left_mask], y[~left_mask], depth + 1, max_depth)
    }

tree = build_tree(X, y, max_depth=2)

import pprint
pprint.pprint(tree)

"""
Output:

{'X < 41.5': 0, 'else': {'X < 81.5': 1, 'else': 1}}
"""
# The second split is actually useless — both branches return 1 anyway,
# so the "real" decision boundary is just X < 41.5.
```

- `def build_tree(X, y, depth=0, max_depth=2)` — takes our features (`X`) and labels (`y`). `depth = 0` means we start at the root, `max_depth = 2` caps how deep we go:
    - Depth 0 (the root): first question (`X < 41.5`).
    - Depth 1 (the children): inside the messy right room, a second question (`X < 81.5`).
    - Depth 2 (the limit): we hit `max_depth`, so we stop, to avoid overfitting.
- `if calculate_gini(y) == 0 or depth >= max_depth: return int(np.round(np.mean(y)))`
    - `calculate_gini(y) == 0` — e.g. if the room is `[1, 1, 1, 1, 1]`, Gini is already 0, everything's identical, no more questions needed.
    - `depth >= max_depth` — even if the room is still mixed (like `[0, 1, 0, 1]`), we stop once we've hit the depth limit.
    - `int(np.round(np.mean(y)))` — the majority vote. If the room is `[0, 1, 1, 1]`, `np.mean` gives `0.75`, and `np.round(0.75)` rounds to `1` — the leaf predicts 1.
- `best_gini = 1.0`, `best_thresh = None` — our current best Gini starts at the worst possible value, so any real split will beat it.
- `sorted_vals = np.sort(np.unique(X[:, 0]))` — `X[:, 0]` grabs every row of column 0, `np.unique` drops duplicates, `np.sort` orders them smallest to largest.
- `midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0` — automatically computes every midpoint: 22↔35→28.5, 35↔48→41.5, and so on.
- `left_y = y[X[:, 0] < thresh]` — "if the value is smaller than the threshold, it goes left." E.g. with `thresh = 49`, values `[22, 35, 48]` go left → `left_y = [0, 0, 1]`, and the rest go right.
- `weighted_gini = ...` — the weighted Gini, computed in Python.
- `if weighted_gini < best_gini: best_gini, best_thresh = weighted_gini, thresh` — updates our running best.
- `left_mask = X[:, 0] < best_thresh` — locks in the winning threshold, forgetting all the worse ones.
- `build_tree(...)` — each child repeats the whole process (check stopping condition → generate thresholds → find the best split → spawn its own children) until every branch hits a stopping condition.
- `pprint.pprint(tree)` — just makes the output easier to read.

Production-ready code looks a lot shorter, since `scikit-learn`'s implementation handles everything for us — but at least now you understand what's happening under the hood (ours was a simplified version):

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([[22], [35], [48], [50], [63], [71], [78], [85], [92], [97]])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])
# 1 = prefers remote work, 0 = doesn't

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=9,
)

model.fit(X, y)
prediction = model.predict([[100]])

print(f"{prediction} -> prefers remote work" if prediction == 1 else f"{prediction} -> doesn't prefer remote work")

"""
Output:

[1] -> prefers remote work
"""
```

That's how a decision tree works. Now we'll move to something that outperforms it in a lot of cases — but you're officially a certified "Decision Tree maker," so be proud.

---

## 4. Random Forest

In plain words, a Random Forest is just a bunch of Decision Trees working together — surprising, right?

Why would we bother with a Random Forest instead of sticking with one Decision Tree? It depends on the situation, but bluntly: a single decision tree doesn't handle new information well (relatable — I don't handle change well either).

Imagine building a decision tree, then adding a new person to the dataset. The problem: the midpoints shift immediately, and that can completely change the tree's result. For example, our best split was `X < 41.5` — but add one new person:

|Introversion|Remote Work|
|--:|--:|
|74|0|

...and now the best split isn't 0.300 anymore — it's 0.364.

That's why a Decision Tree is highly sensitive to the exact boundaries of the training data, and it's the main reason we lean on Random Forests instead.

Why build one fragile tree when you can build 100? It's like asking 100 doctors instead of one — each with their own take: one says Remote, another says Office, three more say Remote — and that's far more reliable than a single answer. But do all 100 trees use the same threshold? No — because 100 identical `X < 41.5` trees would just be 100 photocopies. That's why Random Forest, true to its name, brings in randomness: the choice of people (rows) for each tree is random.

For example, from `[A, B, C, D, E, F, G, H, I, J]`, one tree's random sample might be `[A, C, A, F, J, J, B, H, F, D]` — notice some repeat! That's because we sample **with replacement**: once we pick a person, they go right back in the pool, so we might pick them again (and again).

**Out-Of-Bag (OOB):** imagine a room of 100 people. You write each name on a slip of paper, drop them in a hat, draw one, write it on a new list, put it back, shake, and repeat 100 times. Because names go back in every time, some people get drawn multiple times — which means others get left out entirely, not even once.

The weird magic: no matter how many people are in the room — 100, 1,000, a million — as long as you draw the same number of times as there are people, the split works out the same way, on average:

- ~63% of people get drawn onto the list at least once.
- ~37% never get drawn — that's the **Out-of-Bag** data.

So with our set `A B C D E F G H I J`, after drawing 10 times with replacement, a bootstrap sample might look like:

```
A C A F J J B H F D
```

Which could give us:

```
Used for training: A B C D F H J
Never selected:    E G I
```

This is just an _estimated_ proportion — it won't always land exactly this way, since it's random every time.

A Random Forest is many Decision Trees, each trained on its own bootstrap sample, so each tree sees a slightly different slice of the data. That makes every individual tree an imperfect expert — not because its algorithm is worse, but because it never sees the whole picture. Some trees will misjudge certain samples...

Say a new sample gets predicted:

- Tree 1 → Likes IT
- Tree 2 → Likes IT
- Tree 3 → Dislikes IT
- ...
- Tree 100 → Likes IT

Suppose we end up with 21 trees saying "Dislike" and 79 saying "Like" — we go with the majority: **Like IT**. Even though 21 trees got it wrong, enough of the others got it right, so the crowd wins.

But why is it called "Random _Forest_" specifically? Here's the missing piece:

|Age|Introversion|Salary|Coffee/day|Likes IT|
|--:|--:|--:|--:|--:|
|20|25|2500|3|0|
|22|40|2800|2|0|
|27|60|3400|4|1|
|31|70|3900|5|1|

(1 = likes IT, 0 = doesn't)

There are 4 features: Age, Introversion, Salary, Coffee. A normal decision tree asks itself: "which feature gives me the lowest Gini impurity?" Suppose it decides Introversion wins, splitting at `Introversion < 50`. Then it checks the left child, asks the same question again, and — suppose — Introversion wins _again_. Eventually the whole tree leans entirely on one feature. Strong? Yes. Safe? Not really.

A Random Forest changes the rules: each tree only gets to see **some** of the features, not all of them. Tree #1 might only get `[Age, Coffee/day]` at the root — it has no idea Salary or Introversion even exist. Tree #2 might only get `[Age, Introversion]`. So every tree ends up with a different "strongest feature," rather than all of them piling onto the same one.

This matters a lot in practice: imagine patients with Cough Severity, Fever, Age, and Blood Pressure. A single decision tree might always pick Fever as the best feature, over and over, even for 80-year-old patients with high blood pressure who really need that feature considered. A Random Forest forces some trees to rank by a different feature entirely, making the overall model much safer than a single Decision Tree.

Back to OOB — why do we even care about it?

```
Storytime:
You're staring at the ceiling, nothing really matters in that moment.
You just remembered you trained a Random Forest model, but do you care? Hardly.
Yet the company needs it ready, so you start wondering, "is my model good enough?"
Some distress creeps in. All you want is to know the model actually scores well.
So you think: "what if I test it on the data I already used for training?"
You check online first, though... and sadly...
```

The answer is no. Why? Imagine a teacher handing out the exam questions before the exam. The model already memorized the answers — it'll score 100%, sure, but did it actually _understand_ anything? No. It just memorized.

Normally we solve this by splitting the dataset into a **Training Set** (used to train the model) and a **Test Set** (data the model has never seen, used to check how well it generalizes).

But think about it: if we only have 100 patients and reserve 20 for testing, we only get to train on 80. Wouldn't it be nice to use all 100 for training while still having a fair way to test?

That's exactly what OOB gives us. Remember our bootstrap sample `A C A F J J B H F D`? Notice `E G I` were never selected. Tree #17 has never seen those three patients, so asking it to predict patient E is a fair test — for Tree #17, E is exactly like a brand-new patient walking through the door. Same for G and I.

In other words, bootstrap sampling accidentally creates thousands of tiny "mini-exams" for the trees, letting us evaluate the model while still training on almost the whole dataset.

Now, the industry-standard version:

|Patient|Age|Fever|Cough Severity|Blood Pressure|Oxygen|Sick|
|:--|--:|--:|--:|--:|--:|--:|
|A|22|0|1|118|99|0|
|B|35|1|6|126|95|1|
|C|48|1|8|138|91|1|
|D|50|0|2|124|98|0|
|E|63|1|7|145|90|1|
|F|71|1|9|152|87|1|
|G|78|0|3|142|96|0|
|H|85|1|10|158|84|1|
|I|92|1|8|162|82|1|
|J|97|1|9|166|80|1|

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Features: [Age, Fever, Cough Severity, Blood Pressure, Oxygen]
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

# 0 = Healthy, 1 = Sick
y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1])

model = RandomForestClassifier(random_state=42)  # fixed seed, so results are reproducible
model.fit(X, y)

prediction = model.predict(X_casual_test)
confidence = model.predict_proba(X_casual_test)

print(f"Predictions for new patients: {prediction}")
print(f"Confidence (Healthy vs Sick):\n{confidence}")
print(f"Feature Importances (Age, Fever, Cough, BP, O2):\n{model.feature_importances_}")

"""
Output:

Predictions for new patients: [0 1]
Confidence (Healthy vs Sick):
[[0.96 0.04]
 [0.03 0.97]]
Feature Importances (Age, Fever, Cough, BP, O2):
[0.003 0.268 0.273 0.157 0.299]
"""
```

> Note: Random Forest results are inherently a bit random (it's in the name!) — without `random_state` set, you'd get slightly different numbers every run. The pattern — Cough Severity and Oxygen mattering most, Age barely at all — is the part worth remembering.

Now that we've done a tree and a forest, we'll learn something that beats both in a lot of situations: **XGBoost**.

---

## 5. XGBoost

What is XGBoost? Another idea built on decision trees — and like any normal person, you're probably thinking: WHY DO WE NEED ANOTHER TREE?? And you're right to ask, because both previous ideas had a real weakness:

- **Decision Tree** → way too sensitive to new data being added.
- **Random Forest** → doesn't learn from its own past mistakes at all.

What do I mean by "doesn't learn from its own past mistakes"? Here's the Random Forest logic:

Airi (just a name, no new terminology) visits 100 doctors just to make sure she's healthy or not.

```
Doctor #1 said Pneumonia
Doctor #2 said Cancer
Doctor #3 said Flu
Doctor #4 said Healthy
Doctor #5 said a Paraphilic Disorder
```

After 100 visits, they vote, and whichever diagnosis got the most votes wins. But notice: every tree/doctor is completely independent of the ones before it — they don't learn from each other's mistakes, they don't care.

That's exactly why XGBoost exists:

```
The first doctor looks at Airi and says "I think your health score is 50."
But Airi's real score is 80 — so Doctor #1's mistake is -30.

The second doctor doesn't examine Airi from scratch. He looks ONLY at that -30
mistake and says, "Doctor #1 missed by 30 points — based on that specific
error, I'll add +15."

Now the combined guess is 65. The remaining mistake is -15.

The third doctor looks ONLY at that -15 and adds +10.

By the end, you don't just take Doctor #100's opinion — you add up everyone's
tiny corrections: Final Answer = Doctor 1 + Doctor 2 + Doctor 3 + ...
```

None of the trees start from scratch — each one starts from the previous tree's mistake.

Now the math (scary part incoming). Suppose Airi found a rhythm game online and has been playing a lot:

|**Game**|**Scroll Speed (x)**|**PERFECT Hits (y)**|**Baseline Prediction (ŷ)**|
|---|---|---|---|
|#1|4.0|40|70|
|#2|6.0|80|70|
|#3|8.0|90|70|

The baseline prediction is just the mean:

$$F_0(x) = \frac{1}{n} \sum_{i=1}^{n} y_i = \frac{40+80+90}{3} = 70$$

Now we find the **gradient**, $g_i = \hat{y}_i - y_i$ — "by how far was the prediction off?" — and the **Hessian** ($h$) — "is the error changing gently, or sharply?" If the error changes sharply, we should adjust cautiously, because a small change in prediction could swing things wildly. If it changes gently, we can afford to adjust more boldly.

Here we're effectively using Mean Squared Error, so our Hessian (second derivative) is always 1:

|**Game**|**Scroll Speed (x)**|**Target (y)**|**Prediction (ŷ)**|**Gradient (g=ŷ−y)**|**Hessian (h)**|
|---|---|---|---|---|---|
|#1|4.0|40|70|70 − 40 = **30**|1|
|#2|6.0|80|70|70 − 80 = **−10**|1|
|#3|8.0|90|70|70 − 90 = **−20**|1|

Now we find candidate thresholds (midpoints, just like a Decision Tree):

- Threshold A: between 4.0 and 6.0 → **5.0**
- Threshold B: between 6.0 and 8.0 → **7.0**

Before running the tournament (splitting by thresholds), we need to know how XGBoost judges a split. In a Decision Tree we asked, "which split gives the lowest Gini impurity?" In XGBoost we ask instead: "if I create this leaf, how much will it help correct the mistakes made by the previous trees?" XGBoost computes a **score** for every leaf:

$$Score = \frac{(\sum g_i)^2}{\sum h_i + \lambda}$$

where $g_i$ is the gradient, $h_i$ is the Hessian, and $\lambda$ is a regularization penalty discouraging overly complex trees.

Before splitting at all (the "parent" node with all 3 games together):

$$\sum g_i = 30 + (-10) + (-20) = 0$$

Since the numerator is 0, the score is 0 regardless of the denominator:

$$\text{Score} = \frac{0^2}{3 + 1} = 0$$

What do scores mean?

- **Low score** → a messy mix of over- and under-predictions that cancel out. No clear signal for how to fix things.
- **High score** → a big, crystal-clear problem where everyone's wrong in the same direction. A goldmine, since one simple adjustment fixes the whole group.
- **Medium score** → a decent hint, but still a bit blurry.

So our un-split parent scores a big fat 0 — cute, but not useful. Let's split.

**Threshold = 5:** `is X < 5` → left = `[30]`, right = `[-10, -20]`

$$\text{Score}_{\text{left}} = \frac{30^2}{1 + 1} = 450 \qquad \text{Score}_{\text{right}} = \frac{(-10-20)^2}{2 + 1} = 300$$

$$Gain = Score_{left} + Score_{right} - Score_{parent} = 450 + 300 - 0 = 750$$

**Threshold = 7:** `is X < 7` → left = `[30, -10]`, right = `[-20]`

$$\text{Score}_{\text{left}} = \frac{(30-10)^2}{2 + 1} = \frac{400}{3} \approx 133.33 \qquad \text{Score}_{\text{right}} = \frac{(-20)^2}{1 + 1} = 200$$

$$Gain = 133.33 + 200 - 0 = 333.33$$

Threshold **5.0** wins, since $750 > 333.33$.

After choosing the winning threshold, we compute the actual correction value for each leaf — the **leaf weight** ($w$):

$$w = \frac{-\sum g_i}{\sum h_i + \lambda}$$

(same as the score formula, just without the square)

- Left leaf (Game #1, $g = 30$): $w_{\text{left}} = \dfrac{-30}{1+1} = -15$ — "our baseline was too high for this game, subtract 15."
- Right leaf (Games #2 & #3, $\sum g = -30$): $w_{\text{right}} = \dfrac{30}{2+1} = +10$ — "our baseline was too low here, add 10."

To keep the model from lurching too far in one step, we scale by a learning rate ($\eta = 0.1$):

- Left leaf: $-15 \times 0.1 = -1.5$
- Right leaf: $10 \times 0.1 = +1.0$

New predictions:

- **Game #1:** $70 - 1.5 = \mathbf{68.5}$ (down from 70, closer to the true 40)
- **Game #2:** $70 + 1.0 = \mathbf{71.0}$ (up from 70, closer to the true 80)
- **Game #3:** $70 + 1.0 = \mathbf{71.0}$ (up from 70, closer to the true 90)

|**Game**|**x**|**y**|**New Prediction**|**New Gradient**|**h**|
|---|---|---|---|---|---|
|#1|4.0|40|68.5|68.5 − 40 = **28.5**|1|
|#2|6.0|80|71.0|71.0 − 80 = **−9.0**|1|
|#3|8.0|90|71.0|71.0 − 90 = **−19.0**|1|

And here it is, production-ready:

```python
import xgboost as xgb
import numpy as np

X_train = np.array([[4.0], [6.0], [8.0]])
y_train = np.array([40, 80, 90])

X_test = np.array([9.0])
y_test = np.array([95])

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1, base_score=70)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

for i in range(len(y_test)):
    print(f"Speed {X_test[i]}: Predicted = {prediction[i]:.2f} | Real Target = {y_test[i]}")

"""
Output:

Speed 9.0: Predicted = 89.88 | Real Target = 95
"""
```

`n_estimators=100` is all the loops it runs through (the 100 doctors). `base_score` is our starting average (the mean baseline). But why use the library version instead of the hand-rolled one? We only ever looked at one feature (Scroll Speed); a real dataset might have Device Audio Latency, Screen Size, Hours of Sleep, and more. During the tournament, XGBoost doesn't just check midpoints for Scroll Speed — it runs the exact same Gain formula for every midpoint of every feature you give it.

**Note:** historically, plain Gradient Boosting came before XGBoost, but this repo does it backwards, so by now you already understand 99% of what Gradient Boosting is. The main differences:

- It uses residuals ($r = y - \hat{y}$) instead of gradients ($g = \hat{y} - y$).
- It never uses a Hessian, a $\lambda$, or the Gain formula, and its tree construction is simpler.

Now the suffering with trees is over! Next up: PCA. Jokinggg — almost had you! We'll build up to PCA slowly, starting from the fundamentals. It's a long topic, so buckle in.

---

## 6. PCA (Principal Component Analysis)

This is where our "fake PCA" starts, because we're about to squeeze 12 hours of topic into 2–3 (or more, honestly):

`1. Variance → 2. Covariance → 3. Covariance Matrix → 4. What is a direction? → 5. Eigenvectors → 6. Eigenvalues → 7. SVD → 8. PCA`

### 1. Variance

Imagine Airi practicing her rhythm game for a week (PERFECT Hits):

**Week 1**

|Day|PERFECT Hits|
|--:|--:|
|1|70|
|2|71|
|3|69|
|4|70|
|5|70|

Is Airi consistent? Absolutely — she's constantly around 70.

Now imagine another week where, on day 1, she trips out of bed and hurts her hand, and on day 3, Hinako visits and distracts her the whole time.

**Week 2**

|Day|PERFECT Hits|
|--:|--:|
|1|15|
|2|96|
|3|42|
|4|83|
|5|74|

Is she consistent this time? Absolutely not — some days she plays like a beginner, other days like she sold her soul for rhythm-game skill.

Even though the average is almost identical between the two weeks, Week 1's scores are tight while Week 2's are scattered all over. That's the difference between low variance and high variance.

But why do we even care? Because every data engineer checks variance almost immediately. Look at the Week 1 table again — ask yourself, "if I used this as a feature, would it help me?" No — because the variance is basically 0.

Think about 20 patients whose age all sits around 34–35 — would you use age as a feature? No, it's useless for prediction. But if those patients' ages ranged from 25 to 81, and older patients were more likely to be sick, suddenly the feature is genuinely useful.

A more mathematical example: think about Airi's Week 1 scores, and ask "compared to what should we measure the spread?" Obviously the average.

$$\mu=\frac{70+71+69+70+70}{5} = 70$$

Now we subtract the mean from each value:

|Score|Mean|Distance|
|--:|--:|--:|
|70|70|0|
|71|70|1|
|69|70|-1|
|70|70|0|
|70|70|0|

Positive = above average, negative = below average. Adding them up: $0+1+(-1)+0+0=0$ — not what we wanted, since positives and negatives cancel out. So instead, we square each distance (negatives become positive, and bigger deviations get punished harder):

|Score|Distance|Squared Distance|
|--:|--:|--:|
|70|0|0|
|71|1|1|
|69|-1|1|
|70|0|0|
|70|0|0|

Sum: $0+1+1+0+0=2$. Average spread: $\frac{2}{5}=0.4$ — extremely tight!

That's the formula we just "invented":

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$$

In Python:

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

### 2. Covariance

Imagine Hinako Hanamura practicing piano, tracked like this:

|Day|Practice Hours|Songs Learned|
|--:|--:|--:|
|1|1|2|
|2|2|4|
|3|3|7|
|4|4|9|
|5|5|11|

As she practices more, she learns more songs — these two features move together.

Practice ↑ = Songs Learned ↑ → **positive covariance**

Now something a little sadder:

|Day|Hours Awake|Energy Level|
|--:|--:|--:|
|1|6|95|
|2|10|82|
|3|14|61|
|4|18|37|
|5|22|14|

The longer she's awake, the more exhausted she gets — one goes up, the other goes down.

Hours Awake ↑ = Energy Level ↓ → **negative covariance**

And now, some nonsense — Hinako reading outside:

|Day|Pages Read|Clouds Outside|
|--:|--:|--:|
|1|12|8|
|2|40|1|
|3|18|10|
|4|53|4|
|5|25|7|

Sometimes pages go up, sometimes clouds go down, sometimes both increase — there's no consistent relationship.

Pages Read ↕ = Clouds Outside ↕ → **~zero correlation**

Now let's "create the formula," using the practice/piano table:

$$\mu_{\text{pieces}} = \frac{2+4+7+9+11}{5} = 6.6 \qquad \mu_{\text{hours}} = \frac{1+2+3+4+5}{5} = 3$$

We don't square the deviations this time — since we have two _different_ features, we multiply them together instead:

|Day|Piano Pieces ($y$)|Deviation ($y-\mu_y$)|
|:-:|:-:|:-:|
|1|2|$2-6.6=-4.6$|
|2|4|$4-6.6=-2.6$|
|3|7|$7-6.6=0.4$|
|4|9|$9-6.6=2.4$|
|5|11|$11-6.6=4.4$|

|Day|Practice Hours ($x$)|Deviation ($x-\mu_x$)|
|:-:|:-:|:-:|
|1|1|-2|
|2|2|-1|
|3|3|0|
|4|4|1|
|5|5|2|

Now their product — this reveals whether they move together:

|Day|$x-\mu_x$|$y-\mu_y$|Product|
|:-:|:-:|:-:|:-:|
|1|-2|-4.6|9.2|
|2|-1|-2.6|2.6|
|3|0|0.4|0|
|4|1|2.4|2.4|
|5|2|4.4|8.8|

Every product is positive — not because we forced it, but because both features moved in the same direction the whole time. Adding them up: $9.2+2.6+0+2.4+8.8=23$. Dividing by $n$ (same idea as variance): $\frac{23}{5}=4.6$.

Ta-da — we just derived this without you noticing:

$$Cov(X,Y) = \frac{1}{n} \sum_{i=1}^n(x_i - \mu_x)(y_i - \mu_y)$$

The only difference from variance: in variance we multiply a feature's deviation by _itself_, in covariance we multiply it by the deviation of _another_ feature.

```python
import numpy as np

def Cov(X, y):
    n = len(X)
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    return np.sum((X - mean_x) * (y - mean_y)) / n
```

But what if we have 20 features? Do we compute every pair by hand? No — that's exactly why the **Covariance Matrix** exists.

### 3. Covariance Matrix

Why do we use a Covariance Matrix? The answer is simpler than it first sounds. Imagine 5 features, or 40, or 4000+. What do we do — start crying? Tempting, but it wouldn't solve anything. Karl Pearson, Sir Ronald Fisher, and Arthur Cayley had the same thought (I thought of it before them, but never got around to writing it down), and the Covariance Matrix was born.

Let's look at Hinako's week (weekends excluded):

|Day|Practice Hours|Piano Pieces|Sleep Hours|Coffee Cups|
|:-:|:-:|:-:|:-:|:-:|
|1|1|2|8|1|
|2|2|4|7|2|
|3|3|7|6|3|
|4|4|9|5|4|
|5|5|11|4|5|

4 features — let's name them $X_1$ = Practice Hours, $X_2$ = Piano Pieces, $X_3$ = Sleep Hours, $X_4$ = Coffee Cups.

Instead of asking "what's the covariance between this feature and that one" one pair at a time, we ask: "what's the covariance between _every_ pair of features?" That's:

- Practice ↔ Piano
- Practice ↔ Sleep
- Practice ↔ Coffee
- Piano ↔ Sleep
- Piano ↔ Coffee
- Sleep ↔ Coffee

That's a lot more work than a single covariance or variance, so we organize it into a table:

|—|Practice|Piano|Sleep|Coffee|
|:--|:-:|:-:|:-:|:-:|
|**Practice**|?|?|?|?|
|**Piano**|?|?|?|?|
|**Sleep**|?|?|?|?|
|**Coffee**|?|?|?|?|

What exactly is this table? Recall:

$$Cov(X, X) = \frac{1}{n} \sum(x_i - \mu_x)(x_i - \mu_x) = \frac{1}{n}\sum(x_i-\mu_x)^2 = Var(X)$$

So the diagonal is just each feature's variance:

|—|Practice Hours|Piano Pieces|Sleep Hours|Coffee Cups|
|:--|:-:|:-:|:-:|:-:|
|**Practice**|Var(Practice)|?|?|?|
|**Piano**|?|Var(Piano)|?|?|
|**Sleep**|?|?|Var(Sleep)|?|
|**Coffee**|?|?|?|Var(Coffee)|

Means: $\mu_{\text{Practice}}=3$, $\mu_{\text{Piano}}=6.6$, $\mu_{\text{Sleep}}=6$, $\mu_{\text{Coffee}}=3$.

Deviations ($x_i - \mu_x$):

|**Day**|**Practice**|**Piano**|**Sleep**|**Coffee**|
|:-:|:-:|:-:|:-:|:-:|
|1|-2|-4.6|2|-2|
|2|-1|-2.6|1|-1|
|3|0|0.4|0|0|
|4|1|2.4|-1|1|
|5|2|4.4|-2|2|

- Variance(Practice) = $\frac{4+1+0+1+4}{5}=2$
- Variance(Piano) = $\frac{21.16+6.76+0.16+5.76+19.36}{5}=\frac{53.2}{5}=10.64$
- Variance(Sleep) = $\frac{4+1+0+1+4}{5}=2$
- Variance(Coffee) = $\frac{4+1+0+1+4}{5}=2$

Now imagine computing every off-diagonal covariance by hand, one pair at a time... no. We use a linear algebra trick instead:

$$\Sigma = \frac{1}{n}(B^T B)$$

(the covariance matrix is $\Sigma$ here, don't confuse it with $\sum$, the "sum" symbol). $B$ is our table after subtracting the means:

$$B = \begin{bmatrix} -2 & -4.6 & 2 & -2 \\\\ -1 & -2.6 & 1 & -1 \\\\ 0 & 0.4 & 0 & 0 \\\\ 1 & 2.4 & -1 & 1 \\\\ 2 & 4.4 & -2 & 2 \end{bmatrix}$$

Doing this for every pair gives:

|—|**Practice**|**Piano**|**Sleep**|**Coffee**|
|---|---|---|---|---|
|**Practice**|2|4.6|-2|2|
|**Piano**|4.6|10.64|-4.6|4.6|
|**Sleep**|-2|-4.6|2|-2|
|**Coffee**|2|4.6|-2|2|

Notice the matrix is **symmetric** — this always happens, because Practice↔Piano and Piano↔Practice are computed the same way.

Piano Pieces has a much bigger spread (10.64) than the other features — Hinako sometimes learns only a few pieces, other days a lot more. Practice Hours ↔ Piano Pieces = +4.6 (positive): more practice, more pieces learned. Practice Hours ↔ Sleep Hours = -2 (negative): more practice, less sleep.

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

# Or just: covariance = np.cov(X, rowvar=False)

print(covariance_matrix(X))

"""
Output:

[[ 2.    4.6  -2.    2.  ]
 [ 4.6  10.64 -4.6   4.6 ]
 [-2.   -4.6   2.   -2.  ]
 [ 2.    4.6  -2.    2.  ]]
"""
```

### 4. What is a direction?

Cute question, one you've probably never asked yourself — but we'll explore it anyway.

Hinako's practice/piano table forms a fairly straight line when plotted:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/857c3835-1e13-4faa-a41f-9641c2e52de9" />


Are the points randomly scattered? Nope — they're clearly heading somewhere. Draw an arrow along that trend, and that arrow _is_ the direction. It tells us: "if I move this way, I roughly follow the data."

So what is a direction, mathematically? Simply a vector. For example:

$$A = \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}$$

points purely to the right.

$$B =\begin{bmatrix} 0 \\\\ 1 \end{bmatrix}$$

points purely up.

$$C =\begin{bmatrix} 1 \\\\ 1 \end{bmatrix}$$

moves diagonally (once right, once up).

$$D =\begin{bmatrix} 2 \\\\ 5 \end{bmatrix}$$

moves more upward than rightward — 2 times right, 5 times up.

Recall part of our earlier result:

$$\Sigma = \begin{bmatrix} 2 & 4.6 \\\\ 4.6 &10.64 \end{bmatrix}$$

We understand that Practice Hours and Piano Pieces grow together — there's a relationship — but we don't yet know _which exact direction_ best captures it (purely platonic, of course).

Imagine holding a ruler you can rotate to any angle — 90°, 270°, 130°... "Which rotation captures the data best?" That's the exact question PCA is trying to answer, and the answer is our first **eigenvector**.

But why bother breaking our heads over this at all? Because it reduces noise, produces better model inputs, and a lot more. Now the scary, painful, agonizing, terrible, frightening, abysmal, terrorizing part begins: eigenvalues.

### 5. Eigenvectors and Eigenvalues

Back to the practice/piano table. A computer only ever sees raw coordinates:

`(1,2)` `(2,4)` `(3,7)` `(4,9)` `(5,11)`

We can eyeball where the points are and how to draw a line through them — the direction is up-and-to-the-right. Machines can't "eyeball" anything; they need actual math.

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/4bd72808-b7c3-4fd3-b5e9-b1d80560d860" />

Remember the covariance matrix?

$$\Sigma = \begin{bmatrix} 2 & 4.6 \\\\ 4.6 &10.64 \end{bmatrix}$$

It looks like a plain table, but it already knows: how much Practice changes, how much Piano changes, and how much they change together — think of it as a compressed report of the relationship between the features.

So instead of testing billions of possible rotations by hand, imagine the covariance matrix telling you: "you don't have to test every direction — I already know the best one."

Give it an arrow, say $\vec{v}= \begin{bmatrix} 1\\\\0 \end{bmatrix}$ ("move only to the right"). The covariance matrix transforms it — sometimes rotating it, sometimes stretching it. But there's a magical special case: some arrows don't get rotated at all by this transformation, only _stretched or shrunk_. That special arrow is the **eigenvector**.

**Important disclaimer:** we never plot the covariance matrix itself. We plot the _original data_ — the covariance matrix is just a spreadsheet describing the relationship between features, nothing more. The eigenvector is what we compute _from_ the covariance matrix, and it tells us the best direction to draw through the original data. (And when we say "cloud" — we mean the scattered data points, because with 1,000+ dots, they really do start to look like a cloud.) One more late realization: the data itself never moves. What changes is our coordinate system — like rotating a map, not the terrain.

Now a fuller example. Remember Airi? She kept training, and her scores improved:

|Day|Scroll Speed|PERFECT Hits|
|:-:|:-:|:-:|
|1|3|52|
|2|5|67|
|3|6|74|
|4|8|91|
|5|10|106|

First we scatter the points:

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/4aad7c42-5d71-428b-80bd-a68057f88893" />

Now we want the relationship between the features, so we compute the covariance matrix (yes, we _could_ use plain covariance — but think of the covariance matrix as covariance's cooler sibling, the one that can actually do SVD and PCA later).

Means:

$$\mu_x = \frac{3+5+6+8+10}{5} = 6.4 \qquad \mu_y = \frac{52+67+74+91+106}{5} = 78$$

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/8948c978-7f0f-4499-824e-8c890f18183b" />

Deviations:

|Day|$x-\mu_x$|$y-\mu_y$|
|:-:|--:|--:|
|1|3−6.4 = **−3.4**|52−78 = **−26**|
|2|5−6.4 = **−1.4**|67−78 = **−11**|
|3|6−6.4 = **−0.4**|74−78 = **−4**|
|4|8−6.4 = **1.6**|91−78 = **13**|
|5|10−6.4 = **3.6**|106−78 = **28**|

Centered matrix:

$$B = \begin{bmatrix} -3.4 & -26 \\\\ -1.4 & -11 \\\\ -0.4 & -4 \\\\ 1.6 & 13 \\\\ 3.6 & 28 \end{bmatrix}$$

Applying $\Sigma = \frac{1}{n}(B^T B)$ gives:

$$\Sigma = \begin{bmatrix} 5.84 & 45.4 \\\\ 45.4 & 353.2 \end{bmatrix}$$

We no longer have "Perfect Hits" or "Scroll Speed" individually — we've compressed everything into the relationship between them: how much Scroll Speed varies, how much PERFECT Hits vary, and how strongly they move together.

Now the important question: "if I had to draw ONE arrow through the middle of this cloud, which direction explains the most information?" That's where the eigenvector becomes the star.

$$Av = \lambda v$$

where $A$ is our $\Sigma$ — an eigenvector never changes direction under this transformation, so don't be scared:

$$\begin{bmatrix} 5.84 & 45.4 \\\\ 45.4 & 353.2 \end{bmatrix} v = \lambda v$$

The **eigenvector** answers: "which direction best follows the cloud?" The **eigenvalue** ($\lambda$) answers: "how important is that direction?"

- $\lambda = 350$: the data is heavily stretched, almost forming a "highway" — a strong, predictable structure.
- $\lambda = 0.6$: the data is just spread out or flat, no real "highway" to find.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/c68c1916-b553-42a8-b4dc-d804dea10194" />

If the eigenvalues were, say, $\lambda_1 = 5$ and $\lambda_2 = 4$, the cloud would look like a circle — no special direction, because the data is spread roughly equally in every direction. PCA doesn't help much here, since there's no "highway" to find.

Back to the formula. Moving everything to one side:

$$Av - \lambda v = 0$$

Just like the high-school move $3x-5x=(3-5)x$, we factor out the vector:

$$(A-\lambda I)v=0$$

where $I = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix}$ (and for a 3×3 matrix, it just expands the same way — 1s along the diagonal).

Normally an equation like this only has the trivial solution $v = 0$, which is useless — so instead we force the **determinant** to be zero:

$$\det(A-\lambda I)=0$$

This is called the **characteristic equation**, and it reveals every eigenvalue. Once we know the eigenvalues, we plug each back into $(A-\lambda I)v=0$ to find the matching eigenvector.

$$\left( \begin{bmatrix} 5.84 & 45.4 \\\\ 45.4 & 353.2 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \end{bmatrix} \right) v = 0 \quad\Rightarrow\quad \begin{bmatrix} 5.84 - \lambda & 45.4 \\\\ 45.4 & 353.2 - \lambda \end{bmatrix} v = 0$$

Solving the determinant:

$$(5.84 - \lambda)(353.2 - \lambda) - 45.4^2 = 0$$ $$\lambda^2 - 359.04\lambda + 1.528 = 0$$

Using the quadratic formula:

$$\lambda = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \quad\Rightarrow\quad \lambda_1 \approx 359.036 \qquad \lambda_2 \approx 0.00426$$

- **Dominant direction** ($\lambda_1 \approx 359.04$): a big value — most of the information lies along this direction.
- **Minimal-spread direction** ($\lambda_2 \approx 0.0043$): tiny — almost no variance along this direction.

Substituting $\lambda_1$ back in and taking the first row:

$$-347.36x + 45.4y = 0 \quad\Rightarrow\quad y = \frac{347.36}{45.4}x \approx 7.65x$$

Choosing $x=1$: $v = \begin{bmatrix} 1 \\\\ 7.65 \end{bmatrix}$. Normalizing ($L=\sqrt{1^2+7.65^2}\approx7.72$):

$$v_{1,\text{normalized}} \approx \begin{bmatrix} 0.130 \\\\ 0.992 \end{bmatrix}$$

This is our first principal component: for every 0.130 it moves right, it moves 0.992 up.

The second eigenvector, from $\lambda_2$, is perpendicular to the first:

$$v_2 \approx \begin{bmatrix} -0.992 \\\\ 0.130 \end{bmatrix}$$

$v_1$ and $v_2$ are indeed perpendicular — their dot product is ~0.

We already said it, but it's worth repeating: **PC (Principal Component) = Eigenvector**. It's just a fancier title for the same thing.

### 6. SVD (Singular Value Decomposition)

Imagine Airi Sezaki keeps a diary, writing down how much affection happens between her and Hinako each day. Since life loves to be scary, we make it even scarier:

|Day|Hugs|Smile Count|Blush Count|Head Pats|Gifts|Coffee Together|Walk|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|1|5|12|8|2|0|1|1|
|2|7|15|11|3|1|1|1|
|3|2|5|2|0|0|0|0|
|4|9|18|14|4|2|1|1|
|5|6|13|9|3|1|1|0|

Headache? Sensory overload? Both — exactly. Now imagine 10 more features tomorrow, and 3,000+ a year from now. What do we feed our model, given that some features clearly matter more than others? Good luck finding "the one that matters most" by hand — and I sure hope you weren't planning to ask an LLM to eyeball it for you. That's exactly why we use SVD.

Imagine a messy audio track — vocals, drums, bass, and guitar all mashed into one file. SVD is like running that track through a filter that isolates every instrument. Looking at our 7-feature table, SVD asks: "if I had to describe these data points, what's the biggest direction they stretch in?"

Suppose SVD discovers:

```
Pattern #1: Hugs, Smiles, Blushes, Head Pats grow together      → "Affection Pattern"
Pattern #2: Gifts, Coffee Together, Walks grow together too      → "Dating Pattern"
```

Nobody wrote those labels — math found them. So instead of 7 axes (Hugs, Smiles, ...), we get 2 new axes: `Affection Pattern`, `Dating Pattern`. If `Hugs=7, Smile=15, Blush=11, Head Pats=3`, SVD might compress that into `Affection = 9.8` — a single number carrying a lot of the same signal. Same for `Coffee=1, Walk=1, Gift=2` → `Date Activity = 1.7`. Seven dimensions become two.

The formula:

$$A = U\Sigma{V^T}$$

- $V^T$ (rotation / "directions") — the directions of the axes in the original feature space; the "patterns" it found.
- $\Sigma$ (scaling / "importance") — a diagonal matrix of singular values, just like eigenvalues: which patterns matter (large values) and which are noise (tiny values).
- $U$ (projection / "coordinates") — how each original data point sits on the new "pure" axes.

In matrix form:

$$A = \begin{bmatrix} 5 & 12 & 8 & 2 & 0 & 1 & 1 \\\\ 7 & 15 & 11 & 3 & 1 & 1 & 1 \\\\ 2 & 5 & 2 & 0 & 0 & 0 & 0 \\\\ 9 & 18 & 14 & 4 & 2 & 1 & 1 \\\\ 6 & 13 & 9 & 3 & 1 & 1 & 0 \end{bmatrix}$$

(rows = days, columns = features). We can spot "toxic" (i.e. highly correlated) relationships between columns — Hugs & Smiles rise together, Hugs & Blushes rise together, Hugs & Head Pats rise together. So `Hugs, Smiles, Blushes, Head Pats` are basically describing the same underlying thing.

SVD asks: "can one hidden variable explain most of these?" Suppose it invents:

$$z=0.30(\text{Hugs})+0.62(\text{Smile})+0.55(\text{Blush})+0.45(\text{HeadPats})+\cdots$$

_(illustrative weights — SVD finds the true axes of maximum variance entirely on its own.)_ So instead of remembering 7 numbers for Day 1, we compute one:

$$z_1 = (5 \times 0.30) + (12 \times 0.62) + (8 \times 0.55) + (2 \times 0.45) + (0 \times 0.12) + (1 \times 0.08) + (1 \times 0.04)$$ $$= 1.5 + 7.44 + 4.4 + 0.9 + 0 + 0.08 + 0.04 = \mathbf{14.36}$$

And Day 2:

$$z_2 = (7\times0.30)+(15\times0.62)+(11\times0.55)+(3\times0.45)+(1\times0.12)+(1\times0.08)+(1\times0.04) = \mathbf{19.04}$$

Because 0.62 and 0.55 are much larger than 0.04, the math is basically screaming: "pay way more attention to features 2 and 3 than to feature 7." Continuing this for every day gives a single-column "projection":

$$\text{Projection} = \begin{bmatrix} 14.36 \\\\ 19.04 \\ 4.8 \\\\ 23.72 \\\\ 16.36 \end{bmatrix}$$

_(These illustrative weights and shares are for building intuition — see the real, code-verified numbers below, which come out a bit different once SVD is actually run on this data.)_

SVD rotates the coordinate space so the most important pattern lines up with the data:

|Day|PC1|PC2|PC3|...|
|:-:|--:|--:|--:|:-:|
|1|14.36|-0.91|0.52|...|
|2|19.04|-0.30|-0.44|...|
|3|4.80|1.72|-0.10|...|
|4|23.72|-1.15|0.81|...|
|5|16.36|0.64|-0.79|...|

We keep only PC1 (and maybe PC2), because the remaining components carry far less of the total signal:

|**Day**|**PC1**|**PC2**|
|---|---|---|
|**1**|14.36|-0.91|
|**2**|19.04|-0.30|
|**3**|4.80|1.72|
|**4**|23.72|-1.15|
|**5**|16.36|0.64|

So the pipeline is:

```
Original Data (5 rows × 7 features)
        ↓ SVD
5 rows × 7 principal components
        ↓ Keep only the important ones
5 rows × 2 (or even 1) principal components
```

After this, we check the **error matrix**, to see how much information we lost by throwing away PC3, PC4, etc.:

$$A_{approx} = U_{reduced} \cdot \Sigma_{reduced} \cdot V^T_{reduced} \qquad E = A - A_{approx}$$

For one cell (Day 1, Hugs): $[5] - [4.8] = [0.2]$. Doing this for the whole table gives a small "residue" matrix, the same shape as the original (5×7). It should stay small — a big number in the error matrix is a warning sign that you threw away a principal component that actually mattered.

In Python:

```python
import numpy as np

A = np.array([
    [5, 12, 8, 2, 0, 1, 1],
    [7, 15, 11, 3, 1, 1, 1],
    [2, 5, 2, 0, 0, 0, 0],
    [9, 18, 14, 4, 2, 1, 1],
    [6, 13, 9, 3, 1, 1, 0]
])

# 1. Mean centering (crucial step) — SVD only behaves like PCA if data is centered at the origin
A_centered = A - np.mean(A, axis=0)

# 2. Compute SVD
# U: left singular vectors, S: singular values, Vt: right singular vectors (transposed)
U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)

# 3. Project down to k dimensions (let's keep 2)
k = 2
V_reduced = Vt[:k].T
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
# A different-looking outcome than the illustrative one above, because the real SVD does
# extra steps under the hood that the simplified walkthrough skipped — this is what actually
# running it gives you. (For reference: real explained variance here is ≈98.9% / 0.7% / 0.3%
# for PC1/PC2/PC3 — PC1 still dominates completely, just not exactly "92%".)
```

After this pain, we move to PCA — but no worries, once PCA clicks, I'll go through several worked problems so you can see _why_ we use it, not just how.

### 7. PCA (Principal Component Analysis)

What is PCA, actually, and why did we suffer through all that math first? Because without it, spectral graph theory would look like Chinese (or Russian, if you're Chinese — and if you're both, I honestly don't know).

PCA asks: "if I had to keep only the most important information, which directions should I keep?"

Back to Airi's scores:

|Day|Scroll Speed|PERFECT Hits|
|:-:|:-:|:-:|
|1|3|52|
|2|5|67|
|3|6|74|
|4|8|91|
|5|10|106|

Same old steps: $\mu_{SS} \approx 6.4$, $\mu_{PH} \approx 78$, giving

$$\Sigma = \begin{bmatrix} 5.84 & 45.4 \\\\ 45.4 & 353.2 \end{bmatrix}$$

And the same eigen-decomposition as before:

$$\lambda_1 \approx 359.036 \qquad \lambda_2 \approx 0.00426$$

Total variance $= \lambda_1+\lambda_2 \approx 359.04$:

$$\frac{\lambda_1}{\text{Total}} \approx 99.999\% \qquad \frac{\lambda_2}{\text{Total}} \approx 0.001\%$$

$$v_1 \approx \begin{bmatrix} 0.130 \\\\ 0.992 \end{bmatrix} \qquad v_2 \approx \begin{bmatrix} -0.992 \\\\ 0.130 \end{bmatrix}$$

The first eigenvector explains essentially all the variance in the data, and the second explains almost nothing

Quick check: $(0.130)(-0.992) + (0.992)(0.130) \approx 0$ — their dot product is zero, so they're perpendicular. PC1 and PC2 are independent, and PC2 captures almost no information.

After finding this, the magic happens: **we change the axes.** Instead of "Scroll Speed" (x) and "PERFECT Hits" (y), we get new axes — PC1 (the signal, running right along the length of the data) and PC2 (the noise, running across the width of the cloud). If we drop PC2, we lose only about 0.001% of the information — for practical purposes, basically nothing. We stop measuring the "width" of the cloud (pure noise) and keep the "length" (where all the real information lives). In other examples, PC2 can matter a lot more — this one just happens to be an especially clean, nearly one-dimensional relationship.

### Quick recap

- **Variance** → how much one feature spreads around its mean. High variance = scattered values, low variance = tightly clustered values.
- **Covariance** → whether two features move together. Positive = both increase together, negative = one increases while the other decreases, near zero = little linear relationship.
- **Covariance Matrix** → every variance and covariance for many features, in one object. The starting point for PCA.
- **Direction (vector)** → a path or orientation in space; every arrow has a direction and a length.
- **Eigenvector** → the most important directions hidden inside a matrix; for a covariance matrix, the directions the data naturally stretches along.
- **Eigenvalue** → how important an eigenvector is. Large = lots of variance, small = mostly noise.
- **Principal Component (PC)** → an eigenvector of the covariance matrix, with a fancier name because we're doing PCA. PC1 explains the most variance, PC2 the second-most, and so on.
- **Projection** → where each sample lands on a principal component — instead of 7 raw features, a single coordinate on PC1, PC2, etc.
- **SVD** → decomposes a data matrix directly into hidden patterns, even when computing a covariance matrix is inconvenient or expensive. The workhorse behind a lot of ML.
- **PCA** → reduces dimensionality while keeping as much information as possible, by rotating the coordinate system so the important directions become the new axes, then dropping the unimportant ones.

### One more worked example

Imagine Airi working at a medical company. Her boss hands her data with 8,000 features. What now?

1. **Check variance.** A feature like "Hospital Country: Italy, Italy, Italy, Italy, ..." has variance = 0 — delete it.
2. **Check covariance.** If "Average Lung Density" and "Average Tissue Density" move together, there's no point keeping both. She's now down to ~7,800 features — still awful.
3. **Compute the covariance matrix** — not because she loves matrices, but to ask: "what hidden directions live inside all these measurements?"
4. **Find the eigenvectors and eigenvalues.** The first eigenvector might reveal an "overall tumor score" — nobody wrote that feature, math found it. The second might reveal an "inflammation pattern." Now instead of 8,000 features, she has:

```
PC1 = Tumor Severity
PC2 = Inflammation
PC3 = Body Size
PC4 = Scanner Effect
...
```

Much better already. Suppose the eigenvalues break down as:

```
PC1: 65%   PC2: 20%   PC3: 8%   PC4: 4%   everything else: 3%
```

Keep the first four, discard the rest, and get, for one patient:

```
Patient A: PC1 = 3.2, PC2 = -1.4, PC3 = 0.7, PC4 = 2.1
```

Four numbers instead of 8,000 features — much more convenient, and almost all the same signal. That was a small example — but this was the hardest part of the whole month, so it's mostly downhill (cry-wise) from here.

---

Now I'll go recreate the next month (yes, backwards — PySpark first, then SQL, then graph theory, and finally this one).

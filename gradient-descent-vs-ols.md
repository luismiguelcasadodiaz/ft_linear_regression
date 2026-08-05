# Gradient Descent: An Alternative to Ordinary Least Squares

## 1. Why an Alternative Method?

Ordinary Least Squares (OLS) solves for the slope $m$ and intercept $b$ **analytically** — it gives you an exact formula, derived by setting derivatives to zero and solving directly. This works great for simple linear regression, but it has limits:

- For models with many parameters (e.g., multiple regression with hundreds of features), solving the system of normal equations directly requires matrix inversion, which is computationally expensive ($O(p^3)$ for $p$ features) and can be numerically unstable.
- Some models (logistic regression, neural networks, etc.) have no closed-form solution at all.

**Gradient descent** solves the same minimization problem — minimize the error — but does it **iteratively**, by taking small steps downhill on the error surface until it converges to (approximately) the same answer OLS gives analytically.

---

## 2. The Same Starting Point: The Cost Function

Both methods start from the same idea. We define a cost function measuring how wrong our line is. For linear regression, using the **Mean Squared Error (MSE)**:

$$J(m, b) = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - (mx_i + b)\right)^2$$

(Some formulations use $\frac{1}{2n}$ instead of $\frac{1}{n}$ — this is purely a convenience that cancels a factor of 2 later. Either is fine; we'll use $\frac{1}{n}$ here and note where the choice matters.)

- **OLS** finds the minimum by solving $\frac{\partial J}{\partial m} = 0$ and $\frac{\partial J}{\partial b} = 0$ directly, algebraically.
- **Gradient descent** finds the minimum by starting at some initial guess $(m, b)$ and repeatedly moving in the direction that decreases $J$ the fastest.

Think of $J(m,b)$ as a bowl-shaped surface (it's a convex paraboloid in $m$ and $b$ for linear regression). OLS jumps straight to the bottom of the bowl using algebra. Gradient descent rolls a ball down the slope, step by step, until it settles at the bottom.

---

## 3. The Gradient

The **gradient** of $J$ is the vector of partial derivatives — it points in the direction of steepest *ascent* at any given point $(m, b)$:

$$\nabla J(m,b) = \left( \frac{\partial J}{\partial m},\ \frac{\partial J}{\partial b} \right)$$

To *decrease* $J$, we move in the **opposite** direction of the gradient. That's the whole idea of "descent."

### Computing the partial derivatives

Starting from:

$$J(m, b) = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - (mx_i + b)\right)^2$$

**With respect to $m$** (chain rule):

$$\frac{\partial J}{\partial m} = \frac{1}{n}\sum_{i=1}^{n} 2\left(y_i - (mx_i + b)\right)(-x_i) = -\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - (mx_i + b)\right)$$

**With respect to $b$:**

$$\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n} 2\left(y_i - (mx_i + b)\right)(-1) = -\frac{2}{n}\sum_{i=1}^{n} \left(y_i - (mx_i + b)\right)$$

Notice these are **the exact same expressions** that showed up in the OLS derivation — the difference is that OLS sets them to zero and solves algebraically, while gradient descent *uses their value* (as a direction and magnitude) to know how to update $m$ and $b$.

---

## 4. The Update Rule

At each iteration, we nudge $m$ and $b$ a small step in the direction that reduces $J$:

$$m := m - \alpha \frac{\partial J}{\partial m}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Substituting the derivatives:

$$m := m + \frac{2\alpha}{n}\sum_{i=1}^{n} x_i\left(y_i - (mx_i + b)\right)$$

$$b := b + \frac{2\alpha}{n}\sum_{i=1}^{n} \left(y_i - (mx_i + b)\right)$$

Here, $\alpha$ (alpha) is the **learning rate** — a small positive number that controls how big each step is.

### Why subtract the gradient?

- If $\frac{\partial J}{\partial m} > 0$, it means increasing $m$ would increase the error — so we should *decrease* $m$. Subtracting a positive gradient does exactly that.
- If $\frac{\partial J}{\partial m} < 0$, increasing $m$ would decrease the error — so we should *increase* $m$. Subtracting a negative gradient increases $m$.

Either way, subtracting the gradient always moves you "downhill."

---

## 5. The Algorithm, Step by Step

1. **Initialize** $m = 0$, $b = 0$ (or small random values).
2. **Repeat** for a fixed number of iterations, or until $J$ stops changing meaningfully:
   - Compute predictions: $\hat{y}_i = mx_i + b$ for all $i$.
   - Compute the errors: $e_i = y_i - \hat{y}_i$.
   - Compute the gradients:
     $$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_i x_i e_i, \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum_i e_i$$
   - Update simultaneously:
     $$m \leftarrow m - \alpha \frac{\partial J}{\partial m}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
3. **Stop** when the gradients are close to zero (the slope of the bowl is flat — you're near the bottom) or after a set number of iterations.

> **Important:** update $m$ and $b$ *simultaneously* — compute both new values using the *old* $m$ and $b$, then assign both at once. Updating $m$ first and then using the new $m$ to compute the update for $b$ introduces a subtle bug.

---

## 6. The Learning Rate $\alpha$

This is the trickiest part to get right in practice:

- **Too small**: convergence is very slow — you take tiny steps and need thousands of iterations.
- **Too large**: you overshoot the minimum, and the algorithm can oscillate or even diverge (the error gets *worse* each iteration instead of better).
- **Just right**: steady, efficient convergence toward the minimum.

A common practical technique is to **normalize your data first** (e.g., scale $x$ and $y$ to a similar range, like $[0,1]$ or zero mean/unit variance). Raw mileage values (tens of thousands) and raw price values (thousands) have very different scales, which distorts the shape of the cost surface and makes it hard to pick a single $\alpha$ that works well for both $m$ and $b$.

---

## 7. Feature Scaling: Normalization and Denormalization

Section 6 mentioned that unscaled data (like raw mileage in the tens of thousands vs. price in the thousands) can make gradient descent unstable or cause it to **diverge** — the errors and gradients blow up exponentially instead of shrinking, and $\theta_0, \theta_1$ end up as absurd values (e.g., `1e+296`). Here's the math for fixing that with normalization, and for converting your results back afterward.

### 7.1 Why unscaled data breaks gradient descent

The update rule for the slope is:

$$\theta_1 := \theta_1 - \alpha \cdot \frac{1}{m}\sum_i \left(\hat{y}_i - y_i\right) \cdot x_i$$

If $x_i$ (mileage) is on the order of $10^5$, and the initial error $(\hat y_i - y_i)$ is also large (since $\theta_0=\theta_1=0$ is a poor starting guess), then each term in the sum is huge. Multiplying by $\alpha$ and updating $\theta_1$ can overshoot the minimum by a wide margin. The next iteration's error is then even bigger, and the process snowballs — this is divergence.

### 7.2 Min-max normalization

Rescale both $x$ (mileage) and $y$ (price) into the range $[0, 1]$:

$$x_n = \frac{x - x_{min}}{x_{max} - x_{min}}, \qquad y_n = \frac{y - y_{min}}{y_{max} - y_{min}}$$

Run gradient descent entirely on $x_n, y_n$. This keeps every term in the update rule small and comparable in magnitude, so a single, moderate learning rate (e.g., $\alpha \in [0.01, 0.5]$) works reliably for both $\theta_0$ and $\theta_1$.

Gradient descent then converges to normalized parameters $\theta_{0,n}, \theta_{1,n}$ satisfying:

$$y_n = \theta_{1,n} \cdot x_n + \theta_{0,n}$$

These are **not** directly usable for predicting real prices from real mileage — they only work on the normalized scale. You need to convert them back.

### 7.3 Deriving the denormalization formulas

Start by substituting the normalization definitions into $y_n = \theta_{1,n} x_n + \theta_{0,n}$:

$$\frac{y - y_{min}}{y_{max} - y_{min}} = \theta_{1,n} \cdot \frac{x - x_{min}}{x_{max} - x_{min}} + \theta_{0,n}$$

Multiply both sides by $(y_{max} - y_{min})$ to isolate $y$:

$$y - y_{min} = \theta_{1,n} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}} \cdot (x - x_{min}) + \theta_{0,n}(y_{max} - y_{min})$$

$$y = \underbrace{\theta_{1,n} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}}}_{\text{this is } \theta_1} \cdot (x - x_{min}) + \theta_{0,n}(y_{max} - y_{min}) + y_{min}$$

Expanding the $(x - x_{min})$ term and collecting everything that doesn't multiply $x$ gives the true intercept:

$$y = \theta_1 \cdot x \;-\; \theta_1 \cdot x_{min} + \theta_{0,n}(y_{max} - y_{min}) + y_{min}$$

### 7.4 The denormalization formulas

$$\theta_1 = \theta_{1,n} \cdot \frac{y_{max} - y_{min}}{x_{max} - x_{min}}$$

$$\theta_0 = \theta_{0,n}(y_{max} - y_{min}) + y_{min} - \theta_1 \cdot x_{min}$$

**Order matters:** compute $\theta_1$ first, then plug the already-denormalized $\theta_1$ into the $\theta_0$ formula, since $\theta_0$'s formula depends on it.

A common mistake is denormalizing $\theta_0$ as just `theta0_n * (ymax - ymin) + ymin`, dropping the $-\theta_1 \cdot x_{min}$ term. That term only disappears when $x_{min} = 0$ — which is essentially never true for real mileage data — so skipping it gives a wrong intercept even though $\theta_1$ comes out correct.

---

## 8. Batch vs. Stochastic vs. Mini-Batch Gradient Descent

The algorithm described in Section 5 iterates a **fixed number of times**, and each iteration uses **all $n$ points at once** to compute one gradient before taking a step. This is called **batch gradient descent**. It's not the only way to do it — there are two other common variants.

### Batch Gradient Descent (what's described above)

Each **iteration** uses **all $n$ points** to compute one gradient, then takes a single step:

$$\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - (mx_i + b)\right)$$

- One iteration = one full pass over the dataset = one update of $m, b$.
- The gradient is exact at every step — the true direction of steepest descent for the full cost function.
- The path toward the minimum is smooth and deterministic.

### Stochastic Gradient Descent (SGD)

Instead of summing over all points before updating, SGD updates $m$ and $b$ after **each individual point**:

$$\frac{\partial J_i}{\partial m} = -2x_i\left(y_i - (mx_i + b)\right), \qquad \frac{\partial J_i}{\partial b} = -2\left(y_i - (mx_i + b)\right)$$

(no sum, no $\frac{1}{n}$ — just the error contributed by point $i$)

- One **epoch** = one full pass through all $n$ points = $n$ separate updates of $m, b$ (one per point).
- Each individual update is a noisy, cheap approximation of the true gradient — it points roughly downhill, but not exactly.
- The path to the minimum is jittery/zig-zaggy rather than smooth, but each step is far cheaper to compute, which matters on very large datasets.
- The point order is typically shuffled every epoch so the noise doesn't introduce systematic bias.

### Mini-Batch Gradient Descent

A middle ground: split the data into small batches (e.g., 32 points at a time), compute the gradient over each batch, and update after each batch rather than after each point or the whole dataset. This is the standard approach in modern deep learning — it balances the stability of batch GD against the speed and scalability of SGD.

### Comparison

| | Batch GD | Stochastic GD | Mini-Batch GD |
|---|---|---|---|
| Update frequency | Once per full dataset pass | Once per point | Once per small batch |
| Gradient accuracy | Exact | Noisy | Somewhat noisy |
| Convergence path | Smooth | Zig-zag | Moderate |
| Good for small datasets | ✅ Simplest & most stable | Overkill, unnecessary noise | Overkill |
| Good for huge datasets / deep learning | Too slow (one giant gradient per step) | ✅ Common, but noisy | ✅ Most common in practice |

For a small, two-parameter problem like fitting mileage vs. price (as in the 42 `ft_linear_regression` project), **batch gradient descent** is the right and expected choice — simpler to implement, easier to debug, and it converges reliably since $J(m,b)$ is convex.

---

## 9. Why Does This Converge to the Same Answer as OLS?

For simple linear regression, $J(m,b)$ is a **convex** function — it has exactly one minimum, no false valleys to get stuck in. Since the gradient always points toward increasing error, moving opposite to it always moves toward that single minimum. With a small enough learning rate and enough iterations, gradient descent converges to the *same* $(m, b)$ that the OLS closed-form solution gives you directly.

| | OLS (closed-form) | Gradient Descent |
|---|---|---|
| **How it finds the minimum** | Solves $\nabla J = 0$ algebraically | Iteratively steps opposite the gradient |
| **Exactness** | Exact (up to floating-point precision) | Approximate, converges over iterations |
| **Speed for simple regression** | Fast — one-shot computation | Slower — needs many iterations |
| **Scales to many features?** | Expensive (matrix inversion) | Efficient, widely used for large-scale ML |
| **Requires tuning?** | No | Yes — learning rate, iteration count |
| **Works for non-linear models?** | Generally no closed form | Yes — this is why it's the backbone of deep learning |

---

## 10. The Coefficient of Determination ($R^2$)

Once you have $\theta_0, \theta_1$ (denormalized, on real mileage/price), it's useful to measure **how well the line actually fits the data** — that's what $R^2$ is for.

$R^2$ tells you how much of the variation in $y$ is explained by your regression line, as opposed to being left over as unexplained noise. It's a single number between 0 and 1 (or 0%–100%):

- $R^2 = 1$: the line explains all the variation — every point sits exactly on the line.
- $R^2 = 0$: the line explains none of the variation — your model does no better than just predicting $\bar{y}$ (the mean) for every point.
- Values in between give the fraction of variability your model accounts for. E.g. $R^2 = 0.85$ means 85% of the variation in $y$ is explained by the line, and 15% is unexplained.

### 10.1 Splitting total variation

For any point $y_i$, its deviation from the mean $\bar{y}$ splits into a part your model explains and a part it doesn't:

$$\underbrace{(y_i - \bar{y})}_{\text{total deviation}} = \underbrace{(\hat{y}_i - \bar{y})}_{\text{explained by model}} + \underbrace{(y_i - \hat{y}_i)}_{\text{unexplained (residual)}}$$

where $\hat{y}_i = mx_i + b$ is the model's prediction for point $i$. Summing the squares of these across all points gives three quantities:

**Total Sum of Squares** (total variation in $y$, ignoring the model):

$$SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**Residual Sum of Squares** (the error the model still has — what OLS/gradient descent minimizes):

$$SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Explained Sum of Squares** (variation the model successfully accounts for):

$$SS_{reg} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2$$

For least-squares regression, it's a mathematical fact that:

$$SS_{tot} = SS_{reg} + SS_{res}$$

### 10.2 The formula

$$R^2 = \frac{SS_{reg}}{SS_{tot}} = 1 - \frac{SS_{res}}{SS_{tot}}$$

In practice, the second form is what's actually computed, since $SS_{res}$ is usually easier to get directly from the fitted model:

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### 10.3 Intuition

- $SS_{tot}$ is fixed — it depends only on the data ($y$ values), not on the model. It represents how spread out $y$ naturally is.
- $SS_{res}$ is what's left over after the model does its best — the sum of squared errors between actual and predicted values.
- A very good model has $SS_{res}$ tiny compared to $SS_{tot}$, so $\frac{SS_{res}}{SS_{tot}} \approx 0$ and $R^2 \approx 1$.
- A model no better than guessing the mean has $SS_{res} \approx SS_{tot}$, so $R^2 \approx 0$.

### 10.4 Shortcut for simple linear regression

For simple linear regression (one predictor $x$), $R^2$ equals the **square of the Pearson correlation coefficient** between $x$ and $y$:

$$R^2 = r^2, \quad \text{where } r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

This is a handy sanity check: the numerator of $r$ is the same expression that appears in the mean-centered slope formula, so if $m$ has already been computed that way, most of the work toward $r$ is already done.

### 10.5 Applying it to `ft_linear_regression`

Once $\theta_0, \theta_1$ are denormalized (real mileage/price scale), $R^2$ is a good way to report how good the fit actually is — it's a common bonus request for this project:

1. Compute $\hat{y}_i = \theta_1 \cdot mileage_i + \theta_0$ for every point.
2. Compute $\bar{y}$, the mean of the actual prices.
3. Compute $SS_{res} = \sum (price_i - \hat{y}_i)^2$.
4. Compute $SS_{tot} = \sum (price_i - \bar{y})^2$.
5. Compute $R^2 = 1 - \dfrac{SS_{res}}{SS_{tot}}$.

---

## 11. Connecting Back to Your `ft_linear_regression` Warning

The `RuntimeWarning: invalid value encountered in scalar divide` you hit with OLS happens because the closed-form formula has a denominator that can become exactly zero (no variance in $x$). Gradient descent has no such division — it never breaks down that way, since it just follows the slope of $J$ at each step. That's part of why gradient descent is often the more robust and general-purpose choice, even though for simple 2-parameter linear regression, OLS is normally the faster and simpler option when your data is well-behaved.

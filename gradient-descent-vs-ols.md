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

A common practical technique (especially relevant for the 42 `ft_linear_regression` project) is to **normalize your data first** (e.g., scale $x$ and $y$ to a similar range, like $[0,1]$ or zero mean/unit variance). Raw mileage values (tens of thousands) and raw price values (thousands) have very different scales, which distorts the shape of the cost surface and makes it hard to pick a single $\alpha$ that works well for both $m$ and $b$.

---

## 7. Why Does This Converge to the Same Answer as OLS?

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

## 8. Connecting Back to Your `ft_linear_regression` Warning

The `RuntimeWarning: invalid value encountered in scalar divide` you hit with OLS happens because the closed-form formula has a denominator that can become exactly zero (no variance in $x$). Gradient descent has no such division — it never breaks down that way, since it just follows the slope of $J$ at each step. That's part of why gradient descent is often the more robust and general-purpose choice, even though for simple 2-parameter linear regression, OLS is normally the faster and simpler option when your data is well-behaved.

# Linear Regression: Formulas and Derivation

## 1. The Formulas

Given a set of points $(x_i, y_i)$ for $i = 1, \dots, n$, the least-squares regression line $y = mx + b$ has:

**Slope:**

$$m = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - \left(\sum x_i\right)^2}$$

Equivalent form using means:

$$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

**Intercept:**

$$b = \bar{y} - m\bar{x}$$

where $\bar{x}$ and $\bar{y}$ are the means of the x and y values.

---

## 2. Derivation

The goal is to find the line $y = mx + b$ that minimizes the sum of squared errors between the actual values $y_i$ and the predicted values $\hat{y}_i = mx_i + b$.

### Step 1 — Define the error function

$$E(m, b) = \sum_{i=1}^{n} (y_i - (mx_i + b))^2$$

This depends on two unknowns, $m$ and $b$. To find the minimum, take the partial derivative with respect to each and set it to zero.

### Step 2 — Differentiate with respect to $b$

$$\frac{\partial E}{\partial b} = \sum_{i=1}^{n} 2(y_i - mx_i - b)(-1) = 0$$

Simplifying:

$$\sum y_i - m\sum x_i - nb = 0$$

Solving for $b$:

$$b = \frac{\sum y_i - m\sum x_i}{n} = \bar{y} - m\bar{x}$$

This is the first "normal equation" — the intercept formula, expressed in terms of $m$.

### Step 3 — Differentiate with respect to $m$

$$\frac{\partial E}{\partial m} = \sum_{i=1}^{n} 2(y_i - mx_i - b)(-x_i) = 0$$

Simplifying:

$$\sum x_i y_i - m\sum x_i^2 - b\sum x_i = 0$$

### Step 4 — Substitute $b$ into this equation

Replace $b = \bar{y} - m\bar{x}$:

$$\sum x_i y_i - m\sum x_i^2 - (\bar{y} - m\bar{x})\sum x_i = 0$$

Expand:

$$\sum x_i y_i - m\sum x_i^2 - \bar{y}\sum x_i + m\bar{x}\sum x_i = 0$$

Since $\sum x_i = n\bar{x}$ and $\sum y_i = n\bar{y}$:

$$\sum x_i y_i - m\sum x_i^2 - n\bar{x}\bar{y} + mn\bar{x}^2 = 0$$

### Step 5 — Group and solve for $m$

$$\sum x_i y_i - n\bar{x}\bar{y} = m\left(\sum x_i^2 - n\bar{x}^2\right)$$

$$m = \frac{\sum x_i y_i - n\bar{x}\bar{y}}{\sum x_i^2 - n\bar{x}^2}$$

Multiplying numerator and denominator by $n$ (using $n\bar{x} = \sum x_i$, $n\bar{y} = \sum y_i$) gives the standard form:

$$m = \frac{n\sum x_i y_i - \sum x_i \sum y_i}{n\sum x_i^2 - \left(\sum x_i\right)^2}$$

### Step 6 — The mean-centered equivalent form

It can be shown by expanding the products that:

$$\sum x_i y_i - n\bar{x}\bar{y} = \sum (x_i - \bar{x})(y_i - \bar{y})$$

$$\sum x_i^2 - n\bar{x}^2 = \sum (x_i - \bar{x})^2$$

which gives the alternative, often more intuitive, form:

$$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

---

## 3. Summary of the Logic

1. Define the total error as the sum of squared residuals.
2. Take the partial derivative with respect to each unknown ($m$ and $b$) and set both to zero (minimum condition).
3. This produces two "normal equations" in two unknowns.
4. Solve the system: first express $b$ in terms of $m$, then substitute into the other equation to isolate $m$ purely in terms of the data.
5. Once $m$ is known, compute $b$.

---

## 4. Worked Example

Points: (1,2), (2,3), (3,5), (4,4), (5,6)

- $\bar{x} = 3$, $\bar{y} = 4$
- Numerator: $(1-3)(2-4)+(2-3)(3-4)+(3-3)(5-4)+(4-3)(4-4)+(5-3)(6-4) = 4+1+0+0+4 = 9$
- Denominator: $(1-3)^2+(2-3)^2+(3-3)^2+(4-3)^2+(5-3)^2 = 4+1+0+1+4 = 10$
- $m = 9/10 = 0.9$
- $b = 4 - 0.9 \times 3 = 1.3$

**Fitted line:** $y = 0.9x + 1.3$


[return](https://github.com/luismiguelcasadodiaz/ft_linear_regression/blob/main/README.md)

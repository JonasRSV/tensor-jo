# tensor-jo

---

A tiny tensor library written in python

## Installation

```bash
> git clone https://github.com/JonasRSV/tensor-jo.git
> cd tensor-jo
> python3 setup.py install
```

## Examples

### Linear Regression

---

```python
import tensorjo as tj
import numpy as np
x = np.arange(0, 10)
y = x + 5

# Variables
a = tj.var(np.random.rand())
b = tj.var(np.random.rand())

err = tj.mse(y, a * x + b)

print("before training: coefficient %s -- bias: %s -- mse: %s" %
      (a, b, err.output()))

# Updating manually
for i in range(1200):
    g = tj.gradients(err, [a, b])

    a.update(a.v - np.mean(g[0]) * 1e-2)
    b.update(b.v - np.mean(g[1]) * 1e-2)

print("after training: coefficient %s -- bias: %s -- mse: %s" % (a, b,
                                                                 err.output()))

## Or use a builtin optimiser

# Variables
a = tj.var(np.random.rand())
b = tj.var(np.random.rand())

print("before training: coefficient %s -- bias: %s -- mse: %s" %
      (a, b, err.output()))

err = tj.mse(y, a * x + b)

opt = tj.opt.gd(err)
opt.rounds = 1200

opt.minimise([a, b])

print("after training: coefficient %s -- bias: %s -- mse: %s" % (a, b,
                                                                 err.output()))
```

### Logistic Regression

---

```python
import tensorjo as tj
import numpy as np


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


x = np.random.rand(10) - 0.5
y = sigmoid(4 * x - 1)

# Variables
a = tj.var(np.random.rand())
b = tj.var(np.random.rand())

o = tj.sigmoid(a * x + b)
err = tj.mse(y, o)

print("before training: coefficient %s -- bias: %s -- mse: %s" %
      (a, b, err.output()))

# Updating manually
for i in range(5000):
    g = tj.gradients(err, [a, b])

    a.update(a.v - np.mean(g[0]) * 1e-0)
    b.update(b.v - np.mean(g[1]) * 1e-0)

print("after training: coefficient %s -- bias: %s -- mse: %s" % (a, b,
                                                                 err.output()))

## Or use a builtin optimiser

# Variables
a = tj.var(np.random.rand())
b = tj.var(np.random.rand())

print("before training: coefficient %s -- bias: %s -- mse: %s" %
      (a, b, err.output()))

o = tj.sigmoid(a * x + b)
err = tj.mse(y, o)

opt = tj.opt.gd(err)
opt.rounds = 5000
opt.dt = 1e-0

opt.minimise([a, b])

print("after training: coefficient %s -- bias: %s -- mse: %s" % (a, b,
                                                                 err.output()))
```

# Inheritance in Python (From Simple Classes to PyTorch `nn.Module`)

> **Goal:** Understand what inheritance is, see it clearly in a tiny example, and recognize (and inspect!) it inside a large framework like PyTorch.

---

## 1. What Is Inheritance?

**Inheritance** lets a class (the *child* or *subclass*) reuse and extend code from another class (the *parent* or *superclass*).

* Child “**is-a**” parent: `Dog` **is an** `Animal`.
* The child automatically gets the parent’s attributes/methods unless it overrides them.

---

## 2. A Tiny, Obvious Example

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

class Dog(Animal):  # Dog inherits from Animal
    def speak(self):
        return f"{self.name} says woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says meow!"

pets = [Dog("Rex"), Cat("Mimi")]
for p in pets:
    print(p.speak())

Dog("Woof").speak()
```

### What’s happening?

* `Dog(Animal)` and `Cat(Animal)` **inherit** `__init__` and `speak` (until overridden).
* `Dog` and `Cat` **override** `speak()` to specialize behavior.
* Calling `Dog("Woof").speak()`:

  * Creates a `Dog` instance.
  * Calls the method; Python passes the instance as `self` automatically.

> **Why you can “see” inheritance easily here:**
> The parent is tiny, and the child overrides a visible method (`speak`). There’s no mystery—everything is in one screen.

---

## 3. A Real-World Example: PyTorch’s `nn.Module`

```python
import torch
from torch import nn
import inspect

class SimpleNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNet(10, 32, 1)

# (A) Inherited utility: count parameters
print(sum(p.numel() for p in model.parameters()))

# (B) Inherited mode switching
model.eval()
print("training flag:", model.training)

# (C) Peek at parent source
print(inspect.getsource(nn.Module.eval))
print(SimpleNet.__mro__)                     # Show inheritance chain
print(inspect.getsource(nn.Module.__call__)) # How forward() is invoked
```

### What’s being inherited?

`SimpleNet` gets a **huge API** from `nn.Module`, for example:

* **Call flow:** `__call__` (actually `_call_impl`) wraps `forward()` (hooks, autocast, etc.).
* **Parameter & buffer management:** `parameters()`, `named_parameters()`, `register_parameter()`, `buffers()`, `register_buffer()`.
* **Device/dtype moves:** `to()`, `cpu()`, `cuda()`, `half()`.
* **Mode switches:** `train()`, `eval()`.
* **Serialization:** `state_dict()`, `load_state_dict()`.
* **Traversal:** `children()`, `modules()`, `named_modules()`.

You only implement `forward()`. The rest is **already implemented** in `nn.Module`.

> **Why it’s harder to “see” inheritance here:**
> `nn.Module` is large, and you rarely open its source. Much of the magic happens indirectly (e.g., `__call__` calls your `forward`). You rely on documentation or introspection to notice what you got “for free.”

---

## 4. Making Hidden Behavior Visible

Use Python’s introspection tools:

```python
import inspect
from torch import nn

# Show the method resolution order (who Python looks at, and in what order)
print(SimpleNet.__mro__)

# See how nn.Module implements __call__ and eval()
print(inspect.getsource(nn.Module.__call__))
print(inspect.getsource(nn.Module.eval))
```

Common internal call chains:

* `model(x)` → `nn.Module.__call__(...)` → `self.forward(x)`
* `model.eval()` → `nn.Module.train(False)` → loops over children and sets flags
* `model.to(device)` → `nn.Module._apply()` → moves every parameter/buffer

---

## 5. Key Takeaways

* **Small demo (Animal/Dog/Cat):**
  Inheritance is explicit and easy to track: small parent, clear overrides.

* **Large framework (PyTorch):**
  Inheritance gives you tons of functionality, but it’s behind the scenes. You need to:

  1. Read docs or source.
  2. Use `inspect.getsource`, `dir()`, `__mro__`.
  3. Observe behavior by calling inherited methods (e.g., `model.parameters()`).

* **Rule of thumb:**
  In your own code, prefer composition unless a shared interface/contract is clearly beneficial. In libraries, inheritance often standardizes that contract.


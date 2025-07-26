# Confusion Matrix

*ATTENTION: this first commit is just a pre-written code by chatgpt based on my ideas in order to have a first commit. I still need to transfer my annotations to the computer to start writing my own ideas here.*

> **Goal:** Understand what is the confusion matrix, see all the possible metrics about it, clarify the relation between this metrics and their usefullness to day to day needs.

---

## 1  What *is* a confusion matrix and why should I care?

A confusion matrix is a 2 × 2 table that compares **model predictions** with **ground‑truth labels** for a binary classification task. It counts four outcomes:

|              | **Predicted +**     | **Predicted –**     |
| ------------ | ------------------- | ------------------- |
| **Actual +** | **TP** (True Pos.)  | **FN** (False Neg.) |
| **Actual –** | **FP** (False Pos.) | **TN** (True Neg.)  |

### Example 

Suppose we test 1 000 e‑mails (150 spam, 850 ham). The spam‑filter flags 120 as spam; 90 of those are real spam.

|          | Pred Spam | Pred Ham | Total |
| -------- | --------- | -------- | ----- |
| Spam (+) |  90 TP    |  60 FN   |  150  |
| Ham (–)  |  30 FP    |  820 TN  |  850  |

---

## 2  Which metrics can I read off the matrix?

Below each metric is phrased as the **question** it answers, followed by *formula → value → pros ∣ cons* using the spam example (TP = 90, FP = 30, FN = 60, TN = 820).

### 2.1 When the filter shouts "spam", how often is it right? — **Precision (PPV)**

*Formula*   $\displaystyle \text{TP}/(\text{TP}+\text{FP}) = 90/(90+30) = 0.75$
*Pros*  • Trustworthiness of alerts. • Key for costly false alarms.
*Cons*  • Ignores misses (FN). • Drops if prevalence is low.

### 2.2 Of all real spam, how much did we catch? — **Recall (Sensitivity ∣ TPR)**

*Formula*   $90/(90+60)=0.60$
*Pros*  • Measures coverage of positives. • Crucial in medical screening.
*Cons*  • Says nothing about false alarms.

### 2.3 How do we balance Precision ↔ Recall with one score? — **F₁‑Score**

*Formula*   $2PR/(P+R)=2·0.75·0.60/(0.75+0.60)=0.67$
*Pros*  • Single knob to compare models. • Equal weight to P & R.
*Cons*  • Hides trade‑offs; not interpretable probabilistically.

### 2.4 When the model misses, how bad is it? — **False Negative Rate (FNR)**

*Formula*   $\text{FN}/(\text{FN}+\text{TP}) = 60/150 = 0.40$
*Pros*  • Directly complements Recall (FNR = 1–TPR).
*Cons*  • Harder to reason about than Recall.

### 2.5 Among all legitimate mails, how often are they left alone? — **True Negative Rate (Specificity)**

*Formula*   $820/(820+30)=0.964$
*Pros*  • Measures safety for negatives. • Needed for ROC curves.
*Cons*  • Can be misleading if negatives dominate dataset.

### 2.6 How noisy are the spam alerts? — **False Positive Rate (FPR)**

*Formula*   $30/(820+30)=0.036$   (FPR = 1–Specificity)
*Pros*  • Axis of ROC; ties to alert budget.
*Cons*  • Doesn’t show what was missed.

### 2.7 How are these metrics related?

*Precision* ≈ TPR · Prevalence / \[TPR · Prevalence + FPR · (1–Prevalence)]
…and so on. Trade‑offs visualised via ROC or PR curves.

---

## 3  How does a confusion matrix connect to Bayes’ rule?

**Recall** is the likelihood $P(\text{Pred+}\mid\text{Actual+})$.
**Precision** is the posterior $P(\text{Actual+}\mid\text{Pred+})$, scaled by the prior (prevalence).
Equation: $\text{Precision}=\dfrac{\text{Recall}·\text{Prevalence}}{\text{Recall}·\text{Prevalence}+\text{FPR}·(1-\text{Prevalence})}$.

### Example 

Dataset prevalence = 15 %. Using spam filter’s Recall = 0.60, FPR = 0.036 → Precision ≈ 0.75 (matches earlier).

*Pros*  • Gives intuition about base‑rate fallacy.
*Cons*  • Needs prevalence estimate.

## 4 Why precision is so magical?

- Five ways of seeing precision

---

## 4  What are **five lenses** for viewing a confusion matrix?

1. **Raw counts** — operational impact (# of mistakes).
2. **Rate view** — proportions (TPR, FPR).
3. **Bayesian view** — likelihood → posterior.
4. **Cost‑weighted** — multiply cells by monetary or risk cost.
5. **Curve view** — sweep a threshold to plot ROC / PR.

Each lens highlights different decisions (alert limits, cost trade‑offs, risk appetite).

---

## 5  When should I use a confusion matrix and how do I build one?

**When?** Whenever ground truth is available for classification: medical tests, fraud flags, industrial quality control.
**How?** In Python:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # TP,FN,FP,TN order
```

**Example pitfalls**

* Class imbalance → look at rates not counts.
* Multi‑class → use one‑vs‑rest matrices or an $n×n$ matrix.

**Pros**  • Immediate, visual error summary.
**Cons**  • Needs labelled data; limited to classification.

---

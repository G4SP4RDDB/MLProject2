import numpy as np
import matplotlib.pyplot as plt

# === Paste terminal output here ===
raw_output = r"""
 K | avg train acc | avg train f1 | avg test acc | avg test f1
--------------------------------------------------------------------
 1 |       68.9375 |       0.2720 |      70.0000 |       0.2745
 2 |       69.0344 |       0.2754 |      70.0083 |       0.2748
 3 |       69.9906 |       0.3039 |      69.5417 |       0.2781
 4 |       70.0646 |       0.3026 |      70.0000 |       0.2762
 5 |       71.6396 |       0.3354 |      69.6917 |       0.2781
 6 |       72.1417 |       0.3428 |      69.9708 |       0.2784
 7 |       72.3052 |       0.3469 |      69.6458 |       0.2783
 8 |       72.8510 |       0.3576 |      70.0208 |       0.2752
 9 |       72.9510 |       0.3619 |      69.9917 |       0.2749
10 |       72.5844 |       0.3535 |      69.9792 |       0.2755
11 |       72.5177 |       0.3531 |      69.9625 |       0.2754
12 |       72.3354 |       0.3592 |      69.9917 |       0.2747
13 |       72.2073 |       0.3764 |      69.9750 |       0.2745
14 |       72.1375 |       0.4389 |      69.9667 |       0.2759
15 |       72.1469 |       0.4483 |      69.7667 |       0.2792
16 |       72.1417 |       0.4896 |      69.9417 |       0.2769
17 |       72.3000 |       0.5109 |      69.7958 |       0.2786
18 |       72.3240 |       0.5118 |      69.5000 |       0.2817
19 |       72.5323 |       0.5280 |      69.6417 |       0.2812
20 |       72.7542 |       0.5178 |      69.4625 |       0.2834
21 |       72.6635 |       0.5356 |      69.4542 |       0.2820
22 |       72.8750 |       0.5472 |      69.1250 |       0.2875
23 |       73.0052 |       0.5356 |      69.6167 |       0.2822
24 |       73.0667 |       0.5489 |      69.4458 |       0.2873
25 |       72.9781 |       0.5521 |      69.1333 |       0.2848
26 |       73.2260 |       0.5573 |      69.2250 |       0.2896
27 |       73.0646 |       0.5611 |      69.2625 |       0.2873
28 |       73.2073 |       0.5593 |      69.2542 |       0.2892
29 |       73.4792 |       0.5655 |      69.3875 |       0.2856
30 |       73.4521 |       0.5672 |      69.3750 |       0.2884
"""


# === Parse ===
ks = []
train_acc = []
train_f1 = []
test_acc = []
test_f1 = []

for line in raw_output.strip().split("\n"):
    parts = line.split("|")

    if len(parts) != 5:
        continue

    try:
        ks.append(int(parts[0].strip()))
        train_acc.append(float(parts[1].strip()))
        train_f1.append(float(parts[2].strip()))
        test_acc.append(float(parts[3].strip()))
        test_f1.append(float(parts[4].strip()))
    except:
        continue

ks = np.array(ks)
train_acc = np.array(train_acc)
train_f1 = np.array(train_f1)
test_acc = np.array(test_acc)
test_f1 = np.array(test_f1)

# =========================
# Individual graphs
# =========================

# # Train accuracy
# plt.figure(figsize=(8, 4))
# plt.plot(ks, train_acc, marker="o")
# plt.xlabel("K")
# plt.ylabel("Train Accuracy (%)")
# plt.title("Train Accuracy vs K")
# plt.grid(True)
# plt.tight_layout()

# # Test accuracy
# plt.figure(figsize=(8, 4))
# plt.plot(ks, test_acc, marker="o")
# plt.xlabel("K")
# plt.ylabel("Test Accuracy (%)")
# plt.title("Test Accuracy vs K")
# plt.grid(True)
# plt.tight_layout()

# # Train F1
# plt.figure(figsize=(8, 4))
# plt.plot(ks, train_f1, marker="o")
# plt.xlabel("K")
# plt.ylabel("Train Macro-F1")
# plt.title("Train Macro-F1 vs K")
# plt.grid(True)
# plt.tight_layout()

# # Test F1
# plt.figure(figsize=(8, 4))
# plt.plot(ks, test_f1, marker="o")
# plt.xlabel("K")
# plt.ylabel("Test Macro-F1")
# plt.title("Test Macro-F1 vs K")
# plt.grid(True)
# plt.tight_layout()

# =========================
# Comparison graphs
# =========================

# Accuracy comparison
plt.figure(figsize=(8, 4))
plt.plot(ks, train_acc, marker="o", label="Train Accuracy")
plt.plot(ks, test_acc, marker="o", label="Test Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# F1 comparison
plt.figure(figsize=(8, 4))
plt.plot(ks, train_f1, marker="o", label="Train Macro-F1")
plt.plot(ks, test_f1, marker="o", label="Test Macro-F1")
plt.xlabel("K")
plt.ylabel("Macro-F1")
plt.title("Train vs Test Macro-F1")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
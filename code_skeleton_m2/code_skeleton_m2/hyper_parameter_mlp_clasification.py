import numpy as np
import matplotlib.pyplot as plt
from src.methods.mlp import MLP
from src.losses import CrossEntropy
from src.activations import ReLU, Sigmoid, SoftMax
from src.utils import normalize_fn, label_to_onehot, macrof1_fn





feature_data = np.load("Data/features.npz", allow_pickle=True)
train_features = feature_data['xtrain']
train_labels_classif = feature_data['ytrainclassif']

perm = np.random.permutation(train_features.shape[0])
train_features = train_features[perm]
train_labels_classif = train_labels_classif[perm]

val_size = int(0.2 * train_features.shape[0])
x_tr_raw = train_features[val_size:]
x_val_raw = train_features[:val_size]

means = np.mean(x_tr_raw, axis=0)
stds = np.std(x_tr_raw, axis=0)

x_tr = normalize_fn(x_tr_raw, means, stds)
x_val = normalize_fn(x_val_raw, means, stds)
y_tr = label_to_onehot(train_labels_classif[val_size:], C=3)
y_val = label_to_onehot(train_labels_classif[:val_size], C=3)
true_val = train_labels_classif[:val_size]

counts = np.bincount(train_labels_classif[val_size:].astype(int))
class_weights = 1.0 / np.sqrt(counts)
class_weights = class_weights / class_weights.sum()

# --- Learning rate ---
plt.figure(figsize=(12, 5))
lrs = [1e-4, 1e-3, 1e-2, 1e-1]

for lr in lrs:
    model = MLP(dimensions=(13, 64, 32, 3), activations=(ReLU, Sigmoid, SoftMax))
    model.fit(x_tr, y_tr, loss=CrossEntropy, epochs=400, batch_size=32, learning_rate=lr, class_weights=class_weights)
    plt.plot(range(10, 401, 10), model.train_losses, label=f"lr={lr}", marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Effect of Learning Rate(batch size = 32,  arch = simple)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("lr_comparison.png")
plt.show()

# --- Architecture ---
plt.figure(figsize=(12, 5))
configs = {
    "simple (64,32)":   (13, 64, 32, 3),
    "deep (64,32,16)":  (13, 64, 32, 16, 3),
    "wide (128,64)":    (13, 128, 64, 3),
}

for name, dims in configs.items():
    acts = (ReLU,) * (len(dims) - 2) + (SoftMax,)
    model = MLP(dimensions=dims, activations=acts)
    model.fit(x_tr, y_tr, loss=CrossEntropy, epochs=400, batch_size=32, learning_rate=1e-3, class_weights=class_weights)
    plt.plot(range(10, 401, 10), model.train_losses, label=name, marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Effect of Architecture in MLP(lr = 1e-3, batch_size = 32) ")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("arch_comparison.png")
plt.show()

#--- Batch Size ---
plt.figure(figsize=(12, 5))
batches = {
    "small 32":   32,
    "small 64":  64,
    "average 128":  128,
    "average 256": 256,
    "large 512": 512,
}

for name, batch_size in batches.items():
    model = MLP(dimensions=(13, 64, 32, 3), activations=(ReLU, ReLU, SoftMax))
    model.fit(x_tr, y_tr, loss=CrossEntropy, epochs=200, batch_size=batch_size, learning_rate=1e-3, class_weights=class_weights)
    plt.plot(range(10, 201, 10), model.train_losses, label=name, marker="o", markersize=3)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Effect of Batch Size in MLP(lr = 1e-3, arch = 'simple') ")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("batch_comparison.png")
plt.show()

# --- F1 learning rate ---
plt.figure(figsize=(10, 4))
lrs = [1e-4, 1e-3, 1e-2, 1e-1]
f1_scores = []

for lr in lrs:
    model = MLP(dimensions=(13, 64, 32, 3), activations=(ReLU, Sigmoid, SoftMax))
    model.fit(x_tr, y_tr, loss=CrossEntropy, epochs=400, batch_size=32, learning_rate=lr, class_weights=class_weights)
    pred_classes = np.argmax(model.predict(x_val), axis=1)
    f1_scores.append(macrof1_fn(pred_classes, true_val))

plt.bar([str(lr) for lr in lrs], f1_scores, color='steelblue', edgecolor='black')
plt.xlabel("Learning Rate")
plt.ylabel("Macro F1 Score")
plt.title("F1 Score vs Learning Rate")
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.savefig("f1_lr.png")
plt.show()

# --- F1 architecture ---
plt.figure(figsize=(10, 4))
f1_scores = []
for name, dims in configs.items():
    acts = (ReLU,) * (len(dims) - 2) + (SoftMax,)
    model = MLP(dimensions=dims, activations=acts)
    model.fit(x_tr, y_tr, loss=CrossEntropy, epochs=400, batch_size=32, learning_rate=1e-3, class_weights=class_weights)
    pred_classes = np.argmax(model.predict(x_val), axis=1)
    f1_scores.append(macrof1_fn(pred_classes, true_val))

plt.bar(list(configs.keys()), f1_scores, color='steelblue', edgecolor='black')
plt.xlabel("Architecture")
plt.ylabel("Macro F1 Score")
plt.title("F1 Score vs Architecture")
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.savefig("f1_arch.png")
plt.show()
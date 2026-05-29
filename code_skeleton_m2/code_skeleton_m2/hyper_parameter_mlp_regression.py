import numpy as np
import matplotlib.pyplot as plt
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import ReLU, Sigmoid, Linear
from src.utils import normalize_fn

np.random.seed(100)


feature_data = np.load("Data/features.npz", allow_pickle=True)
train_features   = feature_data['xtrain']
train_labels_reg = feature_data['ytrainreg']

perm = np.random.permutation(train_features.shape[0])
train_features = train_features[perm]
train_labels_reg = train_labels_reg[perm]

val_size = int(0.2 * train_features.shape[0])
x_tr_raw  = train_features[val_size:]
x_val_raw = train_features[:val_size]

means = np.mean(x_tr_raw, axis=0)
stds  = np.std(x_tr_raw, axis=0)

x_tr  = normalize_fn(x_tr_raw, means, stds)
x_val = normalize_fn(x_val_raw, means, stds)

y_mean   = train_labels_reg[val_size:].mean()
y_std    = train_labels_reg[val_size:].std()
y_tr_reg  = ((train_labels_reg[val_size:] - y_mean) / y_std).reshape(-1, 1)
y_val_reg = ((train_labels_reg[:val_size]  - y_mean) / y_std).reshape(-1, 1)
y_val_denorm = y_val_reg * y_std + y_mean

EPOCHS = 400
lrs = [1e-4, 1e-3, 1e-2, 1e-1]

reg_configs = {
    "simple (64,32)":  (13, 64, 32, 1),
    "deep (64,32,16)": (13, 64, 32, 16, 1),
    "wide (128,64)":   (13, 128, 64, 1),
}

batches = {
    "32":  32,
    "64":  64,
    "128": 128,
    "256": 256,
    "512": 512,
}

# Loss curve: Learning Rate
plt.figure(figsize=(12, 5))
for lr in lrs:
    model = MLP(dimensions=(13, 64, 32, 1), activations=(ReLU, Sigmoid, Linear))
    model.fit(x_tr, y_tr_reg, loss=MSE, epochs=EPOCHS, batch_size=32, learning_rate=lr)
    plt.plot(range(10, EPOCHS + 1, 10), model.train_losses,
             label=f"lr={lr}", marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Regression — Effect of Learning Rate (arch=simple, batch=32)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("reg_lr_loss.png", dpi=150)
plt.show()

#  MSE bar: Learning Rate
plt.figure(figsize=(10, 4))
mse_scores = []
for lr in lrs:
    model = MLP(dimensions=(13, 64, 32, 1), activations=(ReLU, Sigmoid, Linear))
    model.fit(x_tr, y_tr_reg, loss=MSE, epochs=EPOCHS, batch_size=32, learning_rate=lr)
    pred = model.predict(x_val) * y_std + y_mean
    mse_scores.append(MSE.loss(pred, y_val_denorm))

plt.bar([str(lr) for lr in lrs], mse_scores, color='coral', edgecolor='black')
plt.xlabel("Learning Rate")
plt.ylabel("MSE")
plt.title("Regression — Val MSE vs Learning Rate")
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig("reg_lr_mse.png", dpi=150)
plt.show()

# Loss curve: Architecture
plt.figure(figsize=(12, 5))
for name, dims in reg_configs.items():
    acts = (ReLU,) * (len(dims) - 2) + (Linear,)
    model = MLP(dimensions=dims, activations=acts)
    model.fit(x_tr, y_tr_reg, loss=MSE, epochs=EPOCHS, batch_size=32, learning_rate=1e-3)
    plt.plot(range(10, EPOCHS + 1, 10), model.train_losses,
             label=name, marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Regression — Effect of Architecture (lr=1e-3, batch=32)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("reg_arch_loss.png", dpi=150)
plt.show()

# Loss curve: Batch Size
EPOCHS_BATCH = 200
plt.figure(figsize=(12, 5))
for name, batch_size in batches.items():
    model = MLP(dimensions=(13, 64, 32, 1), activations=(ReLU, Sigmoid, Linear))
    model.fit(x_tr, y_tr_reg, loss=MSE, epochs=EPOCHS_BATCH, batch_size=batch_size, learning_rate=1e-3)
    plt.plot(range(10, EPOCHS_BATCH + 1, 10), model.train_losses,
             label=f"batch={name}", marker='o', markersize=3)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Regression — Effect of Batch Size (lr=1e-3, arch=simple)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("reg_batch_loss.png", dpi=150)
plt.show()
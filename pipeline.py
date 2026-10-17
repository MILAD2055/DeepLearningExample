# =============================================================================
# Step 1 : Setup & Data Preparation
# =============================================================================

# Step 1.2 : Environment Setup — must run before importing cs231nMe.
# sys.path needs the folder that CONTAINS cs231nMe (i.e. My Drive), not
# cs231nMe itself, since we import it as a package via `from cs231nMe import *`.
from google.colab import drive
drive.mount('/content/drive')

import sys, os
drive_root = '/content/drive/My Drive'
if drive_root not in sys.path:
    sys.path.append(drive_root)

# Step 1.1 : Import tools
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231nMe import *   # pulls in get_CIFAR10_data, FullyConnectedNet, Solver,
                          # sgd/sgd_momentum/rmsprop/adam, eval_numerical_gradient, etc.
%matplotlib inline
plt.rcParams["figure.figsize"] = (10.0, 8.0)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Step 1.3 : Download Dataset - CIFAR10
current_working_dir = !pwd
current_working_dir = current_working_dir[0]
%cd /content/drive/My\ Drive/cs231nMe/dataset
!bash get_datasets.sh
%cd {current_working_dir}

# Step 1.4 : Load Dataset
data = get_CIFAR10_data()
for k, v in list(data.items()):
    print(f"{k}: {v.shape}")

# Step 1.5 : Prepare & Create Data -> reshape + mean-subtraction normalization,
# then package the split subsets the solver will train/validate on.
X_train_vectorized = data["X_train"].reshape(data["X_train"].shape[0], -1)
X_val_vectorized = data["X_val"].reshape(data["X_val"].shape[0], -1)
X_test_vectorized = data["X_test"].reshape(data["X_test"].shape[0], -1)

mean_image = np.mean(X_train_vectorized, axis=0)
X_train_vectorized -= mean_image
X_val_vectorized -= mean_image
X_test_vectorized -= mean_image

num_train_small = 50
small_data = {
    "X_train": X_train_vectorized[:num_train_small],
    "y_train": data["y_train"][:num_train_small],
    "X_val": X_val_vectorized,
    "y_val": data["y_val"],
}

full_data = {
    "X_train": X_train_vectorized,
    "y_train": data["y_train"],
    "X_val": X_val_vectorized,
    "y_val": data["y_val"],
}


# =============================================================================
# Step 2 : Model Creation & Initialization  (runs once)
# =============================================================================

# Step 2.1 & 2.2 : Define Architecture & Create Layer Dimensions
input_dim = X_train_vectorized.shape[1]
hidden_dims = [100, 100]
num_classes = 10

# Step 2.3 : Initialize Parameters (W, b, and gamma/beta if normalization is used)
# -- happens inside FullyConnectedNet.__init__ when the model object is built:
model = FullyConnectedNet(
    hidden_dims,
    input_dim=input_dim,
    num_classes=num_classes,
    reg=1e-2,
    weight_scale=5e-2,
    dtype=np.float64,
    normalization="batchnorm",   # set to None to disable
    dropout_keep_ratio=0.9,      # set to 1 to disable dropout
)


# =============================================================================
# Step 3 : Training Setup & Mini-Batch
# =============================================================================

# Step 3.1 : Select training data
train_data = full_data  # swap to small_data for a quick overfit sanity check

# Step 3.2 : Training Hyperparameters
learning_rate = 1e-3
batch_size = 100
num_epochs = 10

# Step 3.3 : Create Solver -> wraps model + data + hyperparameters together.
# This does NOT rebuild the model — it reuses the one created in Step 2.
solver = Solver(
    model,
    train_data,
    update_rule="adam",
    optim_config={"learning_rate": learning_rate},
    lr_decay=0.95,
    batch_size=batch_size,
    num_epochs=num_epochs,
    print_every=100,
)

# Step 3.4 : Start Mini-Batch  <- loop begins here, repeats each iteration
# solver.train() runs Steps 3.4 -> 7.4 in a loop internally:
#   sample mini-batch -> forward pass -> loss -> backward pass -> optimizer
#   update -> back to 3.4, until num_epochs is reached.
solver.train()

# ---------------------------------------------------------------------------
# What happens inside solver.train() -> solver._step(), for reference:
#
#   Step 4 : Forward Pass & Scores          -> model.loss(X_batch)        [scores]
#   Step 5 : Loss Calculation               -> model.loss(X_batch, y_batch)[loss]
#   Step 6 : Backward Pass & Gradient       -> same call returns grads
#   Step 7 : Optimizer & Parameter Update   -> self.update_rule(w, dw, config)
#            applied per-parameter, then loop back to Step 3.4
# ---------------------------------------------------------------------------


# =============================================================================
# Step 8 : Evaluation & Best Model  (runs after the loop exits)
# =============================================================================

# Step 8.1 : Evaluate During Training -> already tracked every epoch inside
# solver.train_acc_history / solver.val_acc_history via check_accuracy()

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history)
plt.title("Training loss")
plt.xlabel("Iteration")

plt.subplot(2, 1, 2)
plt.plot(solver.train_acc_history, label="train")
plt.plot(solver.val_acc_history, label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# Step 8.2 : Compare & Update Best Validation Accuracy -> already handled
# inside solver.train(): model.params now holds solver.best_params

# Step 8.3 : Final Evaluation on Test Set -> runs once, after the loop exits
test_acc = solver.check_accuracy(X_test_vectorized, data["y_test"])
print("Final test accuracy: %f" % test_acc)
print("Best validation accuracy: %f" % solver.best_val_acc)

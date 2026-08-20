from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

#load tensorboard event file
event_path = ("exp_colour_pilot_whiten_foveated\pilot2_final\sub-01_seed0\events.out.tfevents.1787057933.DESKTOP-FDD8AS8.4340.0")

ea = EventAccumulator(event_path)
ea.Reload()


def get_scalar(tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

#get logged metrics
train_steps, train_colour = get_scalar("colour_loss")
val_steps, val_colour = get_scalar("val_colour_loss")
val_f1_steps, val_f1 = get_scalar("val_f1_colour")

train_bce_steps, train_bce = get_scalar("train_bce_loss")
val_bce_steps, val_bce = get_scalar("val_bce_loss")

train_kl_steps, train_kl = get_scalar("train_kl_loss")
val_kl_steps, val_kl = get_scalar("val_kl_loss")

# train_cosine_steps, train_cosine = get_scalar("train_cosine_loss")
# val_cosine_steps, val_cosine = get_scalar("val_cosine_loss")

train_epochs = np.arange(1, len(train_colour) + 1)
val_epochs = np.arange(1, len(val_colour) + 1)
val_f1_epochs = np.arange(1, len(val_f1) + 1)

train_bce_epochs = np.arange(1, len(train_bce) + 1)
val_bce_epochs = np.arange(1, len(val_bce) + 1)

train_kl_epochs = np.arange(1, len(train_kl) + 1)
val_kl_epochs = np.arange(1, len(val_kl) + 1)

# train_cosine_epochs = np.arange(1, len(train_cosine) + 1)
# val_cosine_epochs = np.arange(1, len(val_cosine) + 1)

#plot 1:
#combined colour loss + validation F1

fig, loss_axis = plt.subplots(figsize=(9, 5))

loss_axis.plot(train_epochs, train_colour, label="Train colour loss",)

loss_axis.plot(val_epochs, val_colour, label="Validation colour loss",)

loss_axis.set_xlabel("Epoch")
loss_axis.set_ylabel("Colour loss")


#second y-axis for F1
f1_axis = loss_axis.twinx()

f1_axis.plot(val_f1_epochs, val_f1, linestyle="--", label="Validation colour F1",)

f1_axis.set_ylabel("Validation F1")


#combined legend
loss_lines, loss_labels = (loss_axis.get_legend_handles_labels())

f1_lines, f1_labels = (f1_axis.get_legend_handles_labels())

loss_axis.legend(loss_lines + f1_lines, loss_labels + f1_labels, loc="best",)

plt.title("Colour loss and validation F1 over training")

fig.tight_layout()
plt.show()

#plot 2:
#BCE loss

plt.figure(figsize=(9, 5))

plt.plot(train_bce_epochs, train_bce, label="Train BCE loss",)

plt.plot(val_bce_epochs, val_bce, label="Validation BCE loss",)

plt.xlabel("Epoch")
plt.ylabel("BCE loss")

plt.title("BCE loss over training")

plt.legend()
plt.tight_layout()
plt.show()

#plot 3:
#KL divergence loss
plt.figure(figsize=(9, 5))

plt.plot(train_kl_epochs, train_kl, label="Train KL loss",)

plt.plot(val_kl_epochs, val_kl, label="Validation KL loss",)

plt.xlabel("Epoch")
plt.ylabel("KL divergence")

plt.title("KL divergence loss over training")

plt.legend()
plt.tight_layout()
plt.show()

#plot 4:
#cosine loss
# plt.figure(figsize=(9, 5))

# plt.plot(train_cosine_epochs, train_cosine, label="Train cosine loss",)

# plt.plot(val_cosine_epochs, val_cosine, label="Validation cosine loss",)

# plt.xlabel("Epoch")
# plt.ylabel("Cosine loss")

# plt.title("Cosine loss over training")

# plt.legend()
# plt.tight_layout()
# plt.show()

#plot 5:
#actual BCE and weighted-KL contributions
#colour_loss = BCE + 0.1 * KL

lambda_kl = 0.1

weighted_train_kl = [lambda_kl * x for x in train_kl]

weighted_val_kl = [lambda_kl * x for x in val_kl]

plt.figure(figsize=(9, 5))

plt.plot(train_bce_epochs, train_bce, label="Train BCE",)

plt.plot(train_kl_epochs, weighted_train_kl, label="Train 0.1 × KL",)

plt.plot(val_bce_epochs, val_bce, linestyle="--", label="Validation BCE",)

plt.plot(val_kl_epochs, weighted_val_kl, linestyle="--",label="Validation 0.1 × KL",)

plt.xlabel("Epoch")
plt.ylabel("Contribution to colour loss")

plt.title("BCE and weighted KL contributions")

plt.legend()
plt.tight_layout()
plt.show()

#plot 6:
#actual BCE and weighted-cosine contributions
# colour_loss = BCE + 0.25 * cos

# lambda_cosine = 0.125

# weighted_train_cosine = [lambda_cosine * x for x in train_cosine]

# weighted_val_cosine = [lambda_cosine * x for x in val_cosine]

# plt.figure(figsize=(9, 5))

# plt.plot(train_bce_epochs, train_bce, label="Train BCE",)

# plt.plot(train_cosine_epochs, weighted_train_cosine, label="Train 0.25 × cos",)

# plt.plot(val_bce_epochs, val_bce, linestyle="--", label="Validation BCE",)

# plt.plot(val_cosine_epochs, weighted_val_cosine, linestyle="--",label="Validation 0.25 × cos",)

# plt.xlabel("Epoch")
# plt.ylabel("Contribution to colour loss")

# plt.title("BCE and weighted cosine contributions")

# plt.legend()
# plt.tight_layout()
# plt.show()


#best epochs

min_loss_index = int(np.argmin(val_colour))

max_f1_index = int(np.argmax(val_f1))

print("Best validation-loss epoch:", val_epochs[min_loss_index])

print("Best validation loss:", val_colour[min_loss_index])

print("Best validation-F1 epoch:", val_f1_epochs[max_f1_index])

print("Best validation F1:", val_f1[max_f1_index])


#final component losses

print("\nFinal losses:")

print("Train BCE:", train_bce[-1])

print("Validation BCE:", val_bce[-1])

print("Train KL:", train_kl[-1])

print("Validation KL:", val_kl[-1])

print("Train 0.1 × KL:", lambda_kl * train_kl[-1])

print("Validation 0.1 × KL:", lambda_kl * val_kl[-1])

# print("Train cosine:", train_cosine[-1])

# print("Validation cosine:", val_cosine[-1])

# print("Train 0.25 × cos:", lambda_cosine * train_cosine[-1])

# print("Validation 0.25 × cos:", lambda_cosine * val_cosine[-1])

print("\nAvailable metrics:")
print(ea.Tags()["scalars"])
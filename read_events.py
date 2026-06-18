from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

ea = EventAccumulator("exp_colour_pilot/pilot_full_normalized/sub-01_seed0/events.out.tfevents.1780900701.DESKTOP-FDD8AS8.30260.0")
ea.Reload()

# ea2 = EventAccumulator("exp_colour_pilot/pilot_full_normalized/sub-01_seed0/events.out.tfevents.1780901302.DESKTOP-FDD8AS8.30260.1")
# ea2.Reload()

# print(ea.Tags())
# print(ea2.Tags())

def get_scalar(tag):
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

train_steps, train_colour = get_scalar("colour_loss")
val_steps, val_colour = get_scalar("val_colour_loss")

batches_per_epoch = 20

train_epochs = [s / batches_per_epoch for s in train_steps]
val_epochs = [s / batches_per_epoch for s in val_steps]

plt.figure(figsize=(8, 5))
plt.plot(train_epochs, train_colour, label="Train colour loss")
plt.plot(val_epochs, val_colour, label="Validation colour loss")

plt.xlabel("Epoch")
plt.ylabel("Colour loss")
plt.title("Colour loss over training")
plt.legend()
plt.tight_layout()
plt.show()

min_idx = np.argmin(val_colour)
print("Best epoch:", val_epochs[min_idx])
print("Best val loss:", val_colour[min_idx])

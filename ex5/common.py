import os
import termcolor


RESULTS = 'results/'
SAVED_MODELS = 'saved_models/'
MNIST_CLASSES = 10
IMAGE_SIZE = 28

RANDOM_DIM = 10


def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])


def toGreen(content): return termcolor.colored(content, "green",
                                                 attrs=["bold"])

def toYellow(content): return termcolor.colored(content, "yellow",
                                                 attrs=["bold"])

def toRed(content): return termcolor.colored(content, "red",
                                                 attrs=["bold"])


def toMagenta(content): return termcolor.colored(content, "magenta",
                                                 attrs=["bold"])


def generate_checkpoint_path(model_name: str, epoch: int):
    checkpoint_dir = os.path.join(SAVED_MODELS, model_name, f"{epoch:03d}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}"
                                                f"_{epoch:03d}.ckpt")
    return checkpoint_path


def find_latest_checkpoint(model_name):
    model_checkpoints_dir = os.path.join(SAVED_MODELS, model_name)
    epochs = [int(x) for x in os.listdir(model_checkpoints_dir)]
    last_epoch = max(epochs)
    last_checkpoint = os.path.join(model_checkpoints_dir,
                                   f"{last_epoch:03d}",
                                   f"{model_name}_{last_epoch:03d}.ckpt")
    return last_checkpoint


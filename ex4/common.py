import termcolor


RESULTS = 'results/'
MNIST_CLASSES = 10
IMAGE_SIZE = 28


def toCyan(content): return termcolor.colored(content, "cyan", attrs=["bold"])


def toMagenta(content): return termcolor.colored(content, "magenta",
                                                 attrs=["bold"])

def toGreen(content): return termcolor.colored(content, "green",
                                               attrs=["bold"])

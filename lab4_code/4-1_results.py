from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
from util import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load the data
image_size = [28, 28]
train_images, train_labels, test_images, test_labels = read_mnist()

dbn = DeepBeliefNet(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "pen": 500, "top": 2000, "lbl": 10},
                    image_size=image_size,
                    n_labels=10,
                    batch_size=15
                    )

dbn.recognize(train_images, train_labels, test_images, test_labels)


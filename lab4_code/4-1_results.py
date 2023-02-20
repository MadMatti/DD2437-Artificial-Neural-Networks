from rbm import RestrictedBoltzmannMachine
from util import *
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Load the data
image_size = [28, 28]
train_images, train_labels, test_images, test_labels = read_mnist()

RBM = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                 ndim_hidden=500,
                                 is_bottom=True,
                                 image_size=image_size,
                                 is_top=False,
                                 n_labels=10,
                                 batch_size=20)

RBM.cd1(visible_trainset=train_images, n_iterations=20)


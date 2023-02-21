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

dbn.train_greedylayerwise(train_images, train_labels, 20)
print(dbn.rbm_stack)

iterations = range(20)
plt.plot(iterations, dbn.rbm_stack['vis--hid'].losses, label='vis--hid')
plt.plot(iterations, dbn.rbm_stack['hid--pen'].losses, label='hid--pen')
plt.plot(iterations, dbn.rbm_stack['pen+lbl--top'].losses, label='pen+lbl--top')

plt.xlabel('Iterations')
plt.ylabel('Reconstruction Loss')
plt.legend()
plt.show()


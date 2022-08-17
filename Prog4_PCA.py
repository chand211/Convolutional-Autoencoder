import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

new_data = []
for i in range(len(x_test)):
    new_data.append(x_test[i].flatten())

new_data = np.array(new_data)

data = new_data.data
data.shape

from sklearn.decomposition import PCA

pca = PCA(2)  # we need 2 principal components.
converted_data = pca.fit_transform(new_data.data)

converted_data.shape

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map , c = y_test)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()
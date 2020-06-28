##模式识别课程设计--数据图像展示程序(Fahion-MNIST数据集前49张）

import matplotlib.pyplot as plt
import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

print(x_train_all.shape,y_train_all.shape)#输出训练集样本和标签的大小

#可视化样本，输出训练集中前49个样本
fig, ax = plt.subplots(nrows=7,ncols=7,sharex='all',sharey='all')
ax = ax.flatten()
for i in range(49):
    img = x_train_all[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

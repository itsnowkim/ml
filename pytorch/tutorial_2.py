import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader

import CustomImageDataset

# import dataset
# if we use custom dataset, we have to download from s3 or somewhere...
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# can handle like list.
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 창 size 정의
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    test = training_data[sample_idx]
    
    # training data는 tesnsor array, label id로 구성되어 있다.
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# read custom data
#/data/mnist_test.csv
#/data/mnist_train.csv
path = os.path.join(os.getcwd(),'data')
custom_train = CustomImageDataset.CustomImageDataset(img_dir=path, annotations_file=os.path.join(path,'mnist_train.csv'))
custom_test = CustomImageDataset.CustomImageDataset(img_dir=path, annotations_file=os.path.join(path,'mnist_test.csv'))
# can do same thing with custom data

# Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
# epoch마다 suffle 해서 overfit을 막음, sample들을 minibatch로 전달
# python의 multiprocessing을 이용해서 data 검색 속도 향상
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap="gray")
plt.title(labels_map[label.item()])
plt.show()
print(f"Label: {label}")
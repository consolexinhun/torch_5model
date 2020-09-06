import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet50, resnet34, vgg19, resnet101, inception_v3
from utils import Flatten
from PIL import Image
import os
import pandas as pd
import numpy as np
import tqdm

torch.manual_seed(2020)
device = torch.device('cuda:3' if torch.cuda.is_available() else "cpu")

batchsz = 24
epochs = 50
lr = 1e-4

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer_ImageNet = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


class MyDataset(Dataset):

    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(data_dir, ratio, batchsize=batchsz):
    dataset = ImageFolder(data_dir)
    character = [[] for _ in range(len(dataset.classes))]   # dataset.classes dataset.class_to_idx
    for x, y in dataset.samples:   # (名称， 标签)
        character[y].append(x)

    train_inputs, val_inputs = [], []
    train_labels, val_labels = [], []
    for i, data in enumerate(character): # data表示标签为i的一组图片
        num_sample_train = int(len(data) * ratio[0])
        for x in data[: num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:]:
            val_inputs.append(str(x))
            val_labels.append(i)

    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet),
                                  batch_size=batchsz, drop_last=True, shuffle=True)

    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet),
                                batch_size=batchsz, drop_last=True, shuffle=True)

    loader = {
        'train' :train_dataloader,
        'val'   :val_dataloader
    }
    return loader


def predict(img_path, model_path):
    net = torch.load(model_path)
    net = net.to(device)
    net.eval()  # 一定要加这个 不然结果完全不对
    torch.no_grad()
    img = Image.open(img_path)
    img_ = val_transformer_ImageNet(img).unsqueeze(0)
    # print(img_.shape)
    img_ = img_.to(device)
    outputs = net(img_)
    # print(outputs)
    predicted = outputs.argmax(dim=1)
    return predicted.cpu().item()


def train(model, optimizer, data_loder, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_len = 0

    for index, (fields, labels) in enumerate(data_loder):
        fields, labels = fields.to(device), labels.to(device)

        logits = model(fields)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_len += len(labels)
    loss_mean = total_loss / total_len
    print('epoch:[{}/{}], loss:{:.4f}'.format(epoch, epochs, loss_mean))


def test(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)
            total += len(y)
            correct += pred.eq(y).sum().float().item()
    return correct / total


def main():
    data_dir = './data'
    loader = fetch_dataloaders(data_dir, [0.8, 0.2], batchsize=batchsz)
    print("加载数据集完成")

    train_model = vgg19(pretrained=True)

    net = nn.Sequential(
        *list(train_model.children())[:-1],
        Flatten(),
        # nn.Linear(train_model.fc.in_features, 3)        # 如果是resnet
        nn.Linear(train_model.classifier[0].in_features, 3)  # 如果是VGG
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    criteon = nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        train(net, optimizer, loader['train'], criteon, device, epoch)
        acc = test(net, loader['val'], device)

        print('epoch:', epoch, 'validation acc:', acc)
        if acc >= best_acc:
              torch.save(net, './model/best_model_vgg19')
              best_acc = acc
              best_epoch = epoch
    print("best_epoch:", best_epoch)


if __name__ == '__main__':
    if not os.path.exists("./model"):
        os.makedirs("./model")
    # main()

    path_test = 'car/test/'

    file_list = sorted(os.listdir(path_test), key=lambda x: int(x[:-4]))

    results = [0] * 3450

    tmp = 1

    for image in tqdm(file_list):
        image_index = int(image[:-4])  # int(image.split('.')[0])

        result1 = predict(path_test + image, './model/best_model_vgg19')
        result2 = predict(path_test + image, './model/best_model_18')
        result3 = predict(path_test + image, './model/best_model_34')
        result4 = predict(path_test + image, './model/best_model_50')
        result5 = predict(path_test + image, './model/best_model_101')

        result_list = [result1, result2, result3, result4, result5]

        result = max(result_list, key=result_list.count)

        results[image_index] = result

    np_results = np.array(results)
    pd_data = pd.DataFrame(np_results)
    pd_data.to_csv('./data/result.csv', header=False)

    print("文件保存完成")


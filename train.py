import os
import argparse
from tqdm import tqdm
import sys
import json
import pickle
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# from FCGAFormer import fcga_former0 as create_model
#from coatnet import coatnet_0 as create_model
from poolformer import poolformer_s12 as create_model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
import warnings
from sklearn import metrics
import torch.nn.functional as F
from lion_pytorch import Lion
import csv
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from sklearn import metrics
import matplotlib as mpl
mpl.use('TkAgg')

warnings.filterwarnings("ignore")


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(3407)

def read_split_data(root: str, val_rate: float = 0.2):
    # random.seed(0)  # 保证随机结果可复现
    # torch.manual_seed(3407)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image   = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    prob_all=[]
    lable_all=[]
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    auc = 0.
    f1 = 0.
    recall = 0.
    precision = 0.
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        #Tensor
        pred=F.softmax(pred,1)
        y_scores = pred[:, 1]  #  0代表High
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        y_scores = y_scores.cpu().numpy()
        outputs = pred.cpu().numpy()  # 先把prob转到CPU上，然后再转成numpy，如果本身在CPU上训练的话就不用先转成CPU了
        labels_cpu = labels.cpu().numpy()
        # print(type(y_scores))

        #不使用try会出现数据不均衡造成的错误
        try:
            auc_true=roc_auc_score(labels_cpu, y_scores)
        except ValueError:
             pass   # 或者其它定义，例如roc_auc=0
        auc += auc_true
        outputmax = np.argmax(outputs, axis=1)
        prob_all.extend(outputmax)  # 求每一行的最大值索引
        lable_all.extend(labels)
        f1score = f1_score(lable_all, prob_all, average='macro')
        f1 += f1score
        recallscore = recall_score(lable_all, prob_all, average='macro')
        recall += recallscore
        precisionscore = precision_score(lable_all, prob_all, average='macro')
        precision += precisionscore

        csvFile = open(fr"./result/DBFormer/AUClr0.0001dp0.5/{epoch}.csv", "a", newline="")
        for i,j in zip(y_scores,labels_cpu):
            ss = str(i)+' '+str(j)+'\n'
            csvFile.write(ss)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        # 计算每一轮的指标 和总指标  aucscore这些带score结尾的参数是每一轮的指标的值
        # 不带score的比如auc，f1是总的指标，最后会再去除step+1，来计算出所有的数据的auc

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f},F1-Score: {:.3f},AUC:{:.3f},Recall:{:.3f}," \
                           "Precision:{:.3f}".format(epoch,
                                                     accu_loss.item() / (step + 1),
                                                     accu_num.item() / sample_num,
                                                     f1score,
                                                     auc_true,
                                                     recallscore,
                                                     precisionscore)
    auc_sum=auc / (step + 1)
    f1_sum=f1 / (step + 1)
    recall_sum=recall / (step + 1)
    precision_sum=precision / (step + 1)


    print("这个epoch总的AUC:{:.3f},F1-Score:{:.3f},Recall:{:.3f}，Precision:{:.3f}".format(auc_sum, f1_sum,
                                                                                      recall_sum,
                                                                                      precision_sum))

    folder_path1 = "./result/DBFormer/F1-Scorelr0.0001dp0.5/"
    file_path1 = folder_path1 + "f1_scores.csv"
    if not os.path.exists(folder_path1):
        os.makedirs(folder_path1)
    data = {"epoch": [epoch], "F1-score": [f1_sum]}
    df = pd.DataFrame(data)
    df.to_csv(file_path1, index=False, mode="a", header=not os.path.exists(file_path1))

    folder_path2 = "./result/DBFormer/Recalllr0.0001dp0.5/"
    file_path2 = folder_path2 + r"Recall.csv"
    if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)
    data = {"epoch": [epoch], r"Recall": [recall_sum]}
    df = pd.DataFrame(data)
    df.to_csv(file_path2, index=False, mode="a", header=not os.path.exists(file_path2))

    folder_path3 = "./result/DBFormer/Precisionlr0.0001dp0.5/"
    file_path3 = folder_path3 + "Precision.csv"
    if not os.path.exists(folder_path3):
        os.makedirs(folder_path3)
    data = {"epoch": [epoch], "Precision": [precision_sum]}
    df = pd.DataFrame(data)
    df.to_csv(file_path3, index=False, mode="a", header=not os.path.exists(file_path3))

    # print("F1-Score:{:.3f}".format(f1_score(lable_all, prob_all)))
    # print("AUClr0.0001dp0.2:{:.3f}".format(roc_auc_score(lable_all, prob_all)))
    # print("Recall:{:.3f}".format(recall_score(lable_all, prob_all , average='macro')))
    # print("Precision:{:.3f}".format(precision_score(lable_all, prob_all, average='macro')))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,auc_sum,f1_sum,precision_sum,recall_sum

# def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
#     # 利用sklearn中的函数生成混淆矩阵并归一化
#     cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
#
#     # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
#     pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
#     pl.colorbar()  # 绘制图例
#
#     # 图像标题
#     if title is not None:
#         pl.title(title)
#     # 绘制坐标
#     num_local = np.array(range(len(labels_name)))
#     if axis_labels is None:
#         axis_labels = labels_name
#     pl.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
#     pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
#     pl.ylabel('True label')
#     pl.xlabel('Predicted label')
#
#     # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
#     for i in range(np.shape(cm)[0]):
#         for j in range(np.shape(cm)[1]):
#             if int(cm[i][j] * 100 + 0.5) > 0:
#                 pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
#                         ha="center", va="center",
#                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
#     # 显示
#     pl.savefig("evaluate.jpg", dpi=500, bbox_inches='tight')  # 解决图片不清晰，不完整的问题
#     pl.tight_layout()
#     pl.show()
#
# @torch.no_grad()
# def evaluate(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()
#
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     labels_y = []
#     labels_y_hat = []
#     for step, data in enumerate(data_loader):
#         images, labels = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#         # 将标签和预测值存入列表 制作混淆矩阵
#         labels_y.extend(labels)
#         labels_y_hat.extend(pred_classes.cpu().numpy())
#
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num
#         )
#     # weighted：计算每个标签的指标，并找到它们的平均值，按支持加权（每个标签的真实实例数）。这会改变“宏观”以解决标签不平衡问题; 它可能导致F分数不在精确度和召回之间。
#     accuracy = metrics.accuracy_score(labels_y, labels_y_hat)
#     precision = metrics.precision_score(labels_y, labels_y_hat, average='macro')
#     recall = metrics.recall_score(labels_y, labels_y_hat, average='macro')
#     F1 = metrics.f1_score(labels_y, labels_y_hat, average='macro')
#     print(f"accuracy:{accuracy:.3f},precision:{precision:.3f}, recall:{recall:.3f}, F1_score:{F1:.3f}")
#
#     folder_path1 = "./result/DBFormern_3/F1-Scorelr0.001dp0.2ep30/"
#     file_path1 = folder_path1 + "f1_scores.csv"
#     if not os.path.exists(folder_path1):
#         os.makedirs(folder_path1)
#     data = {"epoch": [epoch], "F1-score": [F1]}
#     df = pd.DataFrame(data)
#     df.to_csv(file_path1, index=False, mode="a", header=not os.path.exists(file_path1))
#
#     folder_path2 = "./result/DBFormern_3/Recalllr0.001dp0.2ep30/"
#     file_path2 = folder_path2 + r"Recall.csv"
#     if not os.path.exists(folder_path2):
#         os.makedirs(folder_path2)
#     data = {"epoch": [epoch], r"Recall": [recall]}
#     df = pd.DataFrame(data)
#     df.to_csv(file_path2, index=False, mode="a", header=not os.path.exists(file_path2))
#
#     folder_path3 = "./result/DBFormern_3/Precisionlr0.001dp0.2ep30/"
#     file_path3 = folder_path3 + "Precision.csv"
#     if not os.path.exists(folder_path3):
#         os.makedirs(folder_path3)
#     data = {"epoch": [epoch], "Precision": [precision]}
#     df = pd.DataFrame(data)
#     df.to_csv(file_path3, index=False, mode="a", header=not os.path.exists(file_path3))
#
#     folder_path4 = "./result/DBFormern_3/Accuracylr0.001dp0.2ep30/"
#     file_path4 = folder_path4 + "Accuracy.csv"
#     if not os.path.exists(folder_path4):
#         os.makedirs(folder_path4)
#     data = {"epoch": [epoch], "accuracy": [accuracy]}
#     df = pd.DataFrame(data)
#     df.to_csv(file_path4, index=False, mode="a", header=not os.path.exists(file_path4))
#
#     if epoch == 19:
#         plot_matrix(labels_y, labels_y_hat, [0, 1, 2], "DBFormer", axis_labels=["High", "Mid", "Low"])
#     # print(labels_y)
#     # print(labels_y_hat)
#     # print(len(labels_y), len(labels_y_hat))
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, F1
#
#     if epoch == 21:
#         plot_matrix(labels_y, labels_y_hat, [0, 1, 2], "DBFormer", axis_labels=["High", "Mid", "Low"])
#     # print(labels_y)
#     # print(labels_y_hat)
#     # print(len(labels_y), len(labels_y_hat))
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, F1

def main(args):
    # 主函数
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter(log_dir="./newruns/DBFormernlr0.0001dp0.5")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model().to(device)
    # model = models.vgg16().to(device)
    # model = models.resnet34().to(device)
    # 使用Inception V3的时候需要将输入变为299
    # model = models.inception_v3().to(device)
    # model = models.inception_v3().to(device)
    # model = models.densenet121().to(device)
    # EfficientNet 与 ViT 系列官方没有实现，需要自己找代码实现

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # # 删除有关分类类别的权重
        # for k in list(weights_dict.keys()):
        #     if "head" in k:
        #         del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    optimizer = Lion(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
        # use_triton=True  # set this to True to use cuda kernel w/ Triton lang (Tillet et al)
    )

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # validate
        val_loss, val_acc,auc,f1score,precision,recall = evaluate(
                                     model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate","AUClr0.0001dp0.2","F1-score","Precision","Recall"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], auc, epoch)
        tb_writer.add_scalar(tags[6], f1score, epoch)
        tb_writer.add_scalar(tags[7], precision, epoch)
        tb_writer.add_scalar(tags[8], recall, epoch)

        # val_loss, val_acc,f1score,precision,recall = evaluate(
        #                              model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate","F1-score","Precision","Recall"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # tb_writer.add_scalar(tags[5], f1score, epoch)
        # tb_writer.add_scalar(tags[6], precision, epoch)
        # tb_writer.add_scalar(tags[7], recall, epoch)



        torch.save(model.state_dict(), "./weights/DBFormern224lr0.0001dp0.5/{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--data-path', type=str,
                        default="E:\data-224-2")
                        # default="E:\data-test")
                        # default = "E:/data-224-3-100")
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

from sklearn.metrics import roc_curve, auc
from constants import *
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import data_normalize
from sklearn.model_selection import KFold
from tqdm import tqdm
from model import MLPModel
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置超参数
    num_epochs = 50000
    warmup_ratio = 0.01
    # learning_rate = 0.01 # for SGD
    learning_rate = 1e-4 # for Adam
    num_folds = 10
    hidden_layer_sizes = (100, 200, 50)

    # 记录最佳结果
    best_roc_auc = 0.0  # 用于跟踪最佳的ROC AUC
    best_epoch = 0  # 用于跟踪最佳的epoch
    best_model_weights = None  # 用于保存最佳模型权重

    # 初始化存储测试结果的列表
    test_losses = []
    test_aucs = []

    # 准备数据
    Train_data = pd.read_csv(Train_1_data_path)
    Train_data = data_normalize(Train_data).values
    Train_label = pd.read_csv(Train_1_label_path).values.ravel()
    Test_data = pd.read_csv(Test_1_data_path)
    Test_data = data_normalize(Test_data).values
    Test_label = pd.read_csv(Test_1_label_path).values.ravel()

    # 设置10折交叉验证
    num_folds = num_folds
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    average_train_losses = []  # 存储每个Fold的训练损失
    average_val_losses = []  # 存储每个Fold的验证损失

    for val_index, (train_indices, val_indices) in enumerate(kf.split(Train_data)):
        # 提取训练数据和验证数据
        train_data_fold = Train_data[train_indices]
        train_label_fold = Train_label[train_indices]
        val_data_fold = Train_data[val_indices]
        val_label_fold = Train_label[val_indices]

        X_train_tensor = torch.tensor(train_data_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(train_label_fold, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(val_data_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(val_label_fold, dtype=torch.float32).to(device)

        # 创建模型和优化器
        input_dim = X_train_tensor.shape[1]
        model = MLPModel(hidden_layer_sizes=hidden_layer_sizes).to(device)
        criterion = nn.BCELoss()
        # # 创建SGD优化器
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # 创建AdamW优化器
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_epochs*warmup_ratio, num_training_steps=num_epochs)

        # 训练循环
        num_epochs = num_epochs
        train_losses = []
        val_losses = []

        tqdm_epochs = tqdm(range(num_epochs), desc=f'Fold [{val_index + 1}/{num_folds}]', dynamic_ncols=True)

        for epoch in tqdm_epochs:
            # 训练阶段
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)[:, 0]
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)[:, 0]
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

                # 计算ROC AUC
                fpr, tpr, thresholds = roc_curve(y_val_tensor.detach().cpu().numpy(), val_outputs.detach().cpu().numpy())
                roc_auc = auc(fpr, tpr)

            tqdm_epochs.set_postfix(train_loss=loss.item(), val_loss=val_loss.item(), val_AUC=roc_auc)

            if best_roc_auc < roc_auc:
                best_roc_auc = roc_auc
                best_epoch = epoch
                best_model_weights = torch.save(model,"wocenima.pth")

        # 测试阶段
        model = torch.load("wocenima.pth")
        X_Test_tensor = torch.tensor(Test_data, dtype=torch.float32).to(device)
        y_Test_tensor = torch.tensor(Test_label, dtype=torch.float32).to(device)
        Test_outputs = model(X_Test_tensor)[:, 0]
        Test_loss = criterion(Test_outputs, y_Test_tensor)
        test_losses.append(Test_loss.item())
        fpr_test, tpr_test, thresholds_test = roc_curve(y_Test_tensor.detach().cpu().numpy(), Test_outputs.detach().cpu().numpy())
        test_auc = auc(fpr_test, tpr_test)
        test_aucs.append(test_auc)

        # 存储每个Fold的训练和验证损失
        average_train_losses.append(train_losses)
        average_val_losses.append(val_losses)

    # 计算10折平均的训练和验证损失
    average_train_losses = np.mean(average_train_losses, axis=0)
    average_val_losses = np.mean(average_val_losses, axis=0)

    # 绘制10折平均训练和验证损失曲线
    plt.plot(average_train_losses, label='Average Train Loss')
    plt.plot(average_val_losses, label='Average Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    param_info = f'Num Epochs: {num_epochs}, Hidden Layer Sizes: {hidden_layer_sizes}'
    plt.text(0.05, 0.1, param_info, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.show()

    # 计算平均测试损失和AUC
    average_test_loss = np.mean(test_losses)
    average_test_auc = np.mean(test_aucs)

    print("optimizer: AdamW")
    print("num_epochs: ", num_epochs)
    print("hidden_layer_sizes: ", hidden_layer_sizes)
    print("lr: ",learning_rate)
    print("num_folds: ", num_folds)
    print(f"test error: {average_test_loss:.4f}")
    print(f"test AUC: {average_test_auc:.4f}")

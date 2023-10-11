from sklearn.metrics import roc_curve, auc
from constants import *
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import data_normalize, LogisticRegressionModel, compute_significance
from sklearn.model_selection import KFold
from tqdm import tqdm



if __name__ == '__main__':

    # 设置超参数
    num_epochs = 100000
    learning_rate = 0.01
    num_folds = 10

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

    val_AUC_areas = []  # 存储每个Fold的AUC面积

    for val_index, (train_indices, val_indices) in enumerate(kf.split(Train_data)):
        # 提取训练数据和验证数据
        train_data_fold = Train_data[train_indices]
        train_label_fold = Train_label[train_indices]
        val_data_fold = Train_data[val_indices]
        val_label_fold = Train_label[val_indices]

        X_train_tensor = torch.tensor(train_data_fold, dtype=torch.float32)
        y_train_tensor = torch.tensor(train_label_fold, dtype=torch.float32)
        X_val_tensor = torch.tensor(val_data_fold, dtype=torch.float32)
        y_val_tensor = torch.tensor(val_label_fold, dtype=torch.float32)

        # 创建模型和优化器
        input_dim = X_train_tensor.shape[1]
        model = LogisticRegressionModel(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
            train_losses.append(loss.item())

            # 验证阶段
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)[:, 0]
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

            # 打印训练和验证损失
            tqdm_epochs.set_postfix(train_loss=loss.item(), val_loss=val_loss.item())

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
    plt.show()

    # 计算test error
    X_Test_tensor = torch.tensor(Test_data, dtype=torch.float32)
    y_Test_tensor = torch.tensor(Test_label, dtype=torch.float32)
    Test_outputs = model(X_Test_tensor)[:, 0]
    Test_loss = criterion(Test_outputs, y_Test_tensor)
    print(f"交叉验证折数：", num_folds)
    print(f"总训练轮数：", num_epochs)
    print(f"test error: {Test_loss.detach().numpy():.4f}")

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_Test_tensor.detach().numpy(), Test_outputs.detach().numpy())
    roc_auc = auc(fpr, tpr)
    print(f"test ROC area: {roc_auc:.4f}")

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

    # 计算特征重要程度
    compute_significance(model)
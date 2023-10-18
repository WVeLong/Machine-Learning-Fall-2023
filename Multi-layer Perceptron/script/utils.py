from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# 计算各特征重要性程度
def compute_significance(model):
    # coefficients 包含了模型的权重参数
    coefficients = model.linear.weight.detach().squeeze().numpy()
    # 计算特征的重要性
    feature_importance = coefficients
    # 将特征重要性标准化到 -1 到 1 之间
    min_importance = feature_importance.min()
    max_importance = feature_importance.max()
    feature_importance_normalized = -1 + 2 * (feature_importance - min_importance) / (max_importance - min_importance)

    num_features = len(feature_importance_normalized)
    feature_indices = range(1, num_features + 1)

    step = 10
    plt.figure(figsize=(10, 6))
    plt.bar(feature_indices, feature_importance_normalized, align='center')
    plt.xlabel('Feature')
    plt.ylabel('Normalized Significance')
    plt.title('The Significance of Association Between Each Feature And Patients’ Survival')
    plt.xticks(feature_indices[::step], feature_indices[::step])
    plt.show()
    pass

# 数据归一化
def data_normalize(df):
    normalized_df = pd.DataFrame()
    for column in df.columns:
        # 获取列的数据类型
        data_type = df[column].dtype

        # 判断数据类型是否为数值型或整数型
        if data_type in [int, float]:
            # 如果需要标准化，使用StandardScaler进行标准化
            scaler = MinMaxScaler()
            normalized_column = scaler.fit_transform(df[[column]])
            # 将标准化后的数据添加到新的DataFrame中
            normalized_df[column] = normalized_column.flatten()
        else:
            # 如果不需要标准化，将原始数据添加到新的DataFrame中
            normalized_df[column] = df[column]
    return normalized_df
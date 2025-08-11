from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier # 0.84
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm
import numpy as np

def data_load(data_path: str):
    """
    Load data and split it into trainset and testset
    """
    X = []
    Y = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        info = line.strip().split(';')
        y = info[0].split(',')
        x = eval(info[1])
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def process_data(data: np.ndarray, labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray):
    """
    Process data
    """
    # 合并标签为字符串（例如 "12_3" 代表服务12错误类型3）
    y_combined = [f"{svc}_{err}" if svc !=0 else "0_0" for svc, err in labels]
    y_test_combined = [f"{svc}_{err}" if svc !=0 else "0_0" for svc, err in test_labels]

    # 编码为整数标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)  # 形状 (n_samples,)
    y_test_encoded = label_encoder.transform(y_test_combined)

    # 为每个服务的特征拼接服务ID（1-48）
    def data_flat(data):
        X_enhanced = []
        for sample in data:
            enhanced = []
            for svc_id in range(1, 46):
                svc_features = sample[svc_id-1]  # 当前服务的5维特征
                enhanced.append(np.concatenate([[svc_id], svc_features]))  # [ID, f1, f2, ..., f5]
            X_enhanced.append(np.stack(enhanced))
        X_enhanced = np.array(X_enhanced)  # 形状 (n_samples, 48, 6)
        X_flat = X_enhanced.reshape(X_enhanced.shape[0], -1)
        return X_flat


    X_flat = data_flat(data)
    X_test_flat = data_flat(test_data)


    model = lgb.LGBMClassifier(
        num_class=len(label_encoder.classes_),  # 总类别数
        objective="multiclass",
        metric="multi_logloss",
        boosting_type="gbdt",
        n_estimators=200,
        learning_rate=0.05
    ) 
    from sklearn.model_selection import train_test_split

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2)
    # 训练
    model.fit(X_train, y_train)

    # 预测概率（获取Top-5）
    # y_proba = model.predict_proba(X_test)  # 形状 (n_test_samples, 294)
    y_proba = model.predict_proba(X_test_flat)  # 形状 (n_test_samples, 294)
    top5_indices = np.argsort(-y_proba, axis=1)[:, :5]  # 每样本取概率最高的5个类别
    top5_labels = label_encoder.inverse_transform(top5_indices.flatten()).reshape(-1, 5)

    from sklearn.metrics import top_k_accuracy_score

    # 计算Top-5准确率
    top_5_accuracy = custom_top_k_accuracy(y_test_encoded, y_proba, k=5)
    top_3_accuracy = custom_top_k_accuracy(y_test_encoded, y_proba, k=3)
    top_1_accuracy = custom_top_k_accuracy(y_test_encoded, y_proba, k=1)
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}, {top_3_accuracy:.2f}, {top_1_accuracy:.2f}")
    print_predictions_with_labels(y_test_encoded, y_proba, label_encoder, k=5)

def print_predictions_with_labels(y_true_encoded, y_proba, label_encoder, k=5):
    """
    输出每条测试数据的真实标签和预测Top-K标签
    
    参数:
        y_true_encoded: 测试集真实标签（已编码的整数）
        y_proba: 模型预测的概率矩阵 (n_samples, n_classes)
        label_encoder: 标签编码器（用于解码整数标签）
        k: 需要输出的Top-K预测数
    """
    # 获取Top-K预测的类别索引
    top_k_indices = np.argsort(-y_proba, axis=1)[:, :k]  # 形状: (n_samples, k)
    
    # 解码标签
    y_true_decoded = label_encoder.inverse_transform(y_true_encoded)  # 真实标签字符串（如 "12_3"）
    top_k_decoded = label_encoder.inverse_transform(top_k_indices.flatten()).reshape(-1, k)  # Top-K预测标签
    
    # 逐条输出
    for i in range(len(y_true_encoded)):
        true_label = y_true_decoded[i]
        pred_labels = top_k_decoded[i]
        print(f"样本 {i+1}: 真实标签 = {true_label}, Top-{k}预测 = {pred_labels}")

def custom_top_k_accuracy(y_true, y_proba, k=5):
    top_k_preds = np.argsort(-y_proba, axis=1)[:, :k]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    return correct / len(y_true)


def decision_merger(train_path: str, test_path: str):
    data, labels = data_load(train_path)
    test_data, test_label = data_load(test_path)
    process_data(data, labels, test_data, test_label)
    return
    train_data, train_label = data_load(test_path)
    # Split trainset and test set
    X_train, _, y_train, _ = train_test_split(data, labels, test_size=0.3, random_state=1002)
    X_test, _, y_test, _ = train_test_split(train_data, train_label, test_size=0.05, random_state=1002)
    # Train the model
    clf2 = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, l2_regularization=0.3, min_samples_leaf=3, verbose=1, max_leaf_nodes=256, early_stopping=False)
    clf2.fit(X_train, y_train)
    scores = clf2.score(X_test, y_test)
    scores.mean()
    res = clf2.predict(X_test)
    r = recall_score(y_test, res, pos_label=0)
    p = precision_score(y_test, res, pos_label=0)
    print('Accuracy:', scores)
    print('Recall: ', r)
    print('Precision:', p)

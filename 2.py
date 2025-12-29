from gensim.models import FastText
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
import pandas as pd
import numpy as np

# ===================== 1. 数据准备（适配论文的子词特征） =====================
# 下载AG_NEWS数据集（论文常用的文本分类数据集，自动下载）
# 也可以用本地数据集，替换成你的文件路径即可
def load_ag_news():
    # 模拟AG_NEWS数据格式：标签（1-4）+ 文本
    train_url = "./train.csv"
    test_url = "./test.csv"
    
    # 读取数据
    train_df = pd.read_csv(train_url, header=None, names=['label', 'title', 'text'])
    test_df = pd.read_csv(test_url, header=None, names=['label', 'title', 'text'])
    
    # 合并标题和文本，生成训练/测试语料
    train_corpus = (train_df['title'] + " " + train_df['text']).tolist()
    test_corpus = (test_df['title'] + " " + test_df['text']).tolist()
    train_labels = train_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    # 分词（适配FastText的子词处理）
    train_tokens = [simple_preprocess(text) for text in train_corpus]
    test_tokens = [simple_preprocess(text) for text in test_corpus]
    
    return train_tokens, train_labels, test_tokens, test_labels

# ===================== 2. 训练FastText模型（论文核心逻辑） =====================
# 加载数据
train_tokens, train_labels, test_tokens, test_labels = load_ag_news()

# 训练FastText模型（复现论文的子词特征）
model = FastText(
    sentences=train_tokens,
    vector_size=100,  # 词向量维度（论文默认值）
    window=5,         # 上下文窗口
    min_count=1,      # 最小词频
    epochs=25,        # 训练轮数
    sg=1,             # Skip-gram（论文对比的核心模型）
    word_ngrams=2,    # 子词长度（论文的核心改进点）
    min_n=3,          # 最小子词长度
    max_n=6           # 最大子词长度
)

# ===================== 3. 模型评估（复现论文的实验结果） =====================
# 计算文本向量（平均词向量，FastText分类的常用方式）
def get_text_vector(tokens, model):
    vecs = [model.wv[token] for token in tokens if token in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

# 生成训练/测试集的文本向量
train_vectors = np.array([get_text_vector(tokens, model) for tokens in train_tokens])
test_vectors = np.array([get_text_vector(tokens, model) for tokens in test_tokens])

# 训练分类器（逻辑回归，和FastText官方分类逻辑一致）
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(train_vectors, train_labels)

# 评估准确率
test_acc = clf.score(test_vectors, test_labels)
print(f"FastText模型测试集准确率：{test_acc:.4f}")

# ===================== 4. 预测示例（大作业展示用） =====================
test_text = "China launches new AI model for natural language processing"
test_token = simple_preprocess(test_text)
test_vec = get_text_vector(test_token, model)
pred_label = clf.predict([test_vec])[0]

label_mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Tech"}
print(f"测试文本：{test_text}")
print(f"预测类别：{label_mapping[pred_label]}")
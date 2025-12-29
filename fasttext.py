from gensim.models import FastText
from gensim.utils import simple_preprocess
import pandas as pd
import numpy as np
import urllib.request
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_ag_news():
    train_url = "./train.csv"
    test_url = "./test.csv"
    
    train_df = pd.read_csv(train_url, header=None, names=['label', 'title', 'text'])
    test_df = pd.read_csv(test_url, header=None, names=['label', 'title', 'text'])
    
    print("\n📝 正在进行文本分词预处理...")
    train_corpus = (train_df['title'] + " " + train_df['text']).tolist()
    test_corpus = (test_df['title'] + " " + test_df['text']).tolist()
    train_labels = train_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    train_tokens = []
    for text in tqdm(train_corpus, desc="训练集分词"):
        train_tokens.append(simple_preprocess(text))
    test_tokens = []
    for text in tqdm(test_corpus, desc="测试集分词"):
        test_tokens.append(simple_preprocess(text))
    
    combined = list(zip(train_tokens, train_labels))
    random.seed(42)
    random.shuffle(combined)
    train_tokens[:], train_labels[:] = zip(*combined)

    #print("\n=== 数据标签检查 ===")
    #print("训练集标签的数据类型:", type(train_labels[0]))
    #print("训练集标签样例:", train_labels[:10])
    #print("训练集标签分布:")
    #from collections import Counter
    #print(Counter(train_labels))
    #print("=== 检查完毕 ===\n")

    return train_tokens, train_labels, test_tokens, test_labels


class FastTextTrainingMonitor:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_progress = tqdm(total=epochs, desc="FastText训练进度", unit='epoch')
    
    def on_epoch_end(self):
        self.epoch_progress.update(1)
        self.epoch_progress.set_postfix({"训练状态": "正常进行中"})
    
    def close(self):
        self.epoch_progress.close()

train_tokens, train_labels, test_tokens, test_labels = load_ag_news()

monitor = FastTextTrainingMonitor(epochs=25)

print("\n🚀 开始训练FastText模型...")

model = FastText(
    vector_size=100, 
    window=5, 
    min_count=1, 
    sg=1,
    min_n=3, 
    max_n=6, 
    workers=1, 
    seed=42
)

model.build_vocab(corpus_iterable=train_tokens)

for epoch in range(25):
    model.train(
        corpus_iterable=train_tokens,  
        total_examples=model.corpus_count,
        epochs=1
    )
    monitor.on_epoch_end()  
monitor.close()

def get_text_vector(tokens, model):
    vecs = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

print("\n🔢 正在生成文本向量...")
train_vectors = []
for tokens in tqdm(train_tokens, desc="训练集向量生成"):
    train_vectors.append(get_text_vector(tokens, model))
train_vectors = np.array(train_vectors)

test_vectors = []
for tokens in tqdm(test_tokens, desc="测试集向量生成"):
    test_vectors.append(get_text_vector(tokens, model))
test_vectors = np.array(test_vectors)

print("\n🧠 训练逻辑回归分类器...")
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

with tqdm(total=1, desc="分类器训练") as pbar:
    clf.fit(train_vectors, train_labels)
    pbar.update(1)

test_acc = clf.score(test_vectors, test_labels)
print(f"\n📊 最终测试集准确率：{test_acc:.4f}")

plt.figure(figsize=(10, 6))

test_texts_en = [
    "AI model improves natural language processing efficiency",
    "Chinese football team beats South Korea to qualify for Asian Cup",
    "Central bank cuts interest rates to boost stock market",
    "UN adopts resolution on climate cooperation"
]
test_texts_cn = [
    "AI模型提升自然语言处理效率",
    "国足击败韩国队晋级亚洲杯",
    "央行降息提振股市上涨",
    "联合国通过气候合作决议"
]

label_mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Tech"}
pred_results = []
pred_confidence = []  

for text in test_texts_en:
    tokens = simple_preprocess(text)
    vec = get_text_vector(tokens, model)
    pred_label = clf.predict([vec])[0]
    pred_proba = clf.predict_proba([vec])[0]
    pred_results.append(label_mapping[pred_label])
    pred_confidence.append(max(pred_proba))  

bars = plt.bar(
    range(len(test_texts_en)), 
    pred_confidence, 
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    alpha=0.8
)

plt.xticks(
    range(len(test_texts_cn)), 
    [f"示例{i+1}\n{text[:10]}..." for i, text in enumerate(test_texts_cn)], 
    rotation=15,
    fontsize=10
)
plt.ylabel('预测置信度', fontsize=12)
plt.title('FastText文本分类预测结果', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)

for i, (bar, res, conf) in enumerate(zip(bars, pred_results, pred_confidence)):
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 0.02, 
        f"{res}\n{conf:.2f}", 
        ha='center', 
        va='bottom', 
        fontsize=11,
        fontweight='bold'
    )

plt.text(
    0.5, 1.05, 
    f"模型测试集整体准确率：{test_acc:.4f}", 
    ha='center', 
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7)
)

plt.tight_layout()
plt.savefig('fasttext_final_result.png', dpi=300, bbox_inches='tight')
print("\n✅ 可视化结果图已保存为：fasttext_final_result.png")
plt.show()

print("\n===== 📋 详细预测结果 =====")
for i, (text_cn, text_en, res, conf) in enumerate(zip(test_texts_cn, test_texts_en, pred_results, pred_confidence)):
    print(f"【示例{i+1}】")
    print(f"中文文本：{text_cn}")
    print(f"英文文本：{text_en}")
    print(f"预测类别：{res}")
    print(f"预测置信度：{conf:.4f}\n")

#print("\n=== 词向量质量诊断 ===")
#try:
    #print("与 'football' 最相似的词:")
    #print(model.wv.most_similar('football', topn=5))
    #print("\n与 'economy' 最相似的词:")
    #print(model.wv.most_similar('economy', topn=5))
    #print("\n与 'technology' 最相似的词:")
    #print(model.wv.most_similar('technology', topn=5))
#except KeyError as e:
    #print(f"错误：词汇 '{e}' 不在词汇表中，可能是因为文本预处理时被过滤掉了。")
#print("=== 诊断完毕 ===")
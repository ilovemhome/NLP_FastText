# -*- coding: utf-8 -*-
import jieba
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np
import nltk
import re

# 一、基于二十大报告文本的中文词云可视化
# 1. 读取文本并分词
def read_and_cut_text(text_path, stopwords_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = jieba.cut(text, cut_all=False)
    # 加载停用词
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    # 过滤停用词和长度小于2的词
    filtered_words = [w for w in words if w not in stopwords and len(w) >= 2]
    return filtered_words

# 2. 词频统计
def get_word_freq(words):
    return Counter(words)

# 3. 生成词云
def generate_wordcloud(word_freq, mask_path, font_path, output_path):
    mask = np.array(Image.open(mask_path))
    wc = WordCloud(
        font_path=font_path,
        background_color='white',
        max_words=400,
        mask=mask,
        width=800,
        height=600
    )
    wc.generate_from_frequencies(word_freq)
    wc.to_file(output_path)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# ------------------- 二、英文人名统计 -------------------
# 1. 读取文本并分句
def read_english_text(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # 分句
    sentences = nltk.sent_tokenize(text)
    return sentences, text

# 2. 统计人名
def extract_names(text):
    # 使用nltk的命名实体识别
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    chunks = nltk.ne_chunk(pos_tags, binary=False)
    names = []
    for chunk in chunks:
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
            name = ' '.join(c[0] for c in chunk)
            names.append(name)
    return names

def plot_top_names(name_freq):
    top10 = name_freq.most_common(10)
    names, counts = zip(*top10)
    plt.figure(figsize=(10,6))
    plt.bar(names, counts, color='skyblue')
    plt.title('Top 10 Person Names in Emma')
    plt.ylabel('Frequency')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 一、中文词云
    chinese_words = read_and_cut_text('20Congress.txt', 'stopwords.txt')
    word_freq = Counter(chinese_words)
    print('高频词汇及频次：')
    for word, freq in word_freq.most_common(20):
        print(f'{word}: {freq}')
    generate_wordcloud(word_freq, 'mapofChina.jpg', 'SimHei.ttf', 'china_wordcloud.png')

    # 二、英文人名统计
    sentences, eng_text = read_english_text('austen-emma.txt')
    names = extract_names(eng_text)
    name_freq = Counter(names)
    print('\n出现频率最高的10个人名：')
    for name, freq in name_freq.most_common(10):
        print(f'{name}: {freq}')
    plot_top_names(name_freq)
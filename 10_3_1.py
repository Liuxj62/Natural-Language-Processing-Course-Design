# 10.3.1 文本分类
# 代码10-1 自定义语料预处理函数
import tensorflow as tf
from collections import Counter
from tensorflow import keras
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# ===================== 第一步：GPU配置 =====================
physical_gpus = tf.config.list_physical_devices('GPU')
print(f"检测到的所有物理GPU：{physical_gpus}")

if physical_gpus:
    tf.config.set_visible_devices(physical_gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_gpus[0], True)
    print(f"✅ 已启用GPU：{physical_gpus[0]}")
else:
    print("❌ 未找到GPU，使用CPU")


# ===================== 第二步：工具函数 =====================
# 打开文件
def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


# 读取文件数据（原生读取，避免pandas版本问题）
def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


# 构建词汇表
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    data_train, lab = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, temp = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


# 读取词汇表
def read_vocab(vocab_dir):
    with open_file(vocab_dir) as fp:
        words = [i.strip() for i in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 读取分类目录
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


# 将id表示的内容转换为文字
def to_words(content, words):
    return ''.join(words[x] for x in content)


# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    return x_pad, y_pad


# ===================== 第三步：数据加载 =====================
# 路径配置（统一绝对路径，避免相对路径混乱）
base_dir = os.path.abspath('../data')
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = os.path.abspath('../tmp/')
save_path = os.path.join(save_dir, 'best_validation.h5')
cache_dir = os.path.abspath('../data/cache/')

# 创建必要目录
os.makedirs(save_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# 构建/读取词汇表
vocab_size = 5000
if not os.path.exists(vocab_dir):
    build_vocab(train_dir, vocab_dir, vocab_size)

# 读取分类和词汇表
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
vocab_size = len(words)
seq_length = 600

# 加载数据（优先缓存）
if os.path.exists(os.path.join(cache_dir, 'x_train.npy')):
    x_train = np.load(os.path.join(cache_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(cache_dir, 'y_train.npy'))
    x_val = np.load(os.path.join(cache_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(cache_dir, 'y_val.npy'))
    x_test = np.load(os.path.join(cache_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(cache_dir, 'y_test.npy'))
else:
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, seq_length)
    # 保存缓存
    np.save(os.path.join(cache_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(cache_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(cache_dir, 'x_val.npy'), x_val)
    np.save(os.path.join(cache_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(cache_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(cache_dir, 'y_test.npy'), y_test)


# ===================== 第四步：模型构建+编译（核心修复） =====================
def TextRNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=600))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6, axis=1))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


# 模型实例化+编译（核心！补全缺失的编译步骤）
model = TextRNN()
model.compile(
    optimizer='rmsprop',  # 优化器
    loss='categorical_crossentropy',  # 损失函数（适配独热编码标签）
    metrics=['categorical_accuracy']  # 评估指标（和绘图键名匹配）
)

# ===================== 第五步：模型训练 =====================
# 配置最优模型保存回调
checkpoint = ModelCheckpoint(
    filepath=save_path,
    monitor='val_categorical_accuracy',  # 匹配编译的指标，避免监控失效
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# 训练模型
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint]
)

# ===================== 第六步：训练曲线绘图（修复空白+重叠） =====================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def plot_acc_loss(history):
    # 动态获取训练轮数，避免硬编码
    epochs = len(history.history['loss'])
    x_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5), dpi=100)  # 增大画布避免重叠

    # 准确率曲线
    plt.subplot(121)
    plt.title('准确率趋势图')
    plt.plot(x_range, history.history['categorical_accuracy'], linestyle='-', color='g', label='训练集')
    plt.plot(x_range, history.history['val_categorical_accuracy'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')

    # 损失曲线
    plt.subplot(122)
    plt.title('损失趋势图')
    plt.plot(x_range, history.history['loss'], linestyle='-', color='g', label='训练集')
    plt.plot(x_range, history.history['val_loss'], linestyle='-.', color='b', label='验证集')
    plt.legend(loc='best')
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')

    plt.tight_layout()
    plt.savefig("3.png", bbox_inches='tight')  # 先保存再显示，避免空白
    plt.show()


# 调用绘图函数（取消注释）
plot_acc_loss(history)

# ===================== 第七步：模型保存+测试 =====================
# 查看模型架构
model.summary()

# 保存模型
model.save(os.path.join(save_dir, 'my_model.h5'))
del model  # 释放内存

# 加载模型并测试
model1 = load_model(save_path)  # 加载最优模型，而非最后一轮模型
y_pre = model1.predict(x_test, verbose=0)

# 计算混淆矩阵
y_pre_arg = np.argmax(y_pre, axis=1)
y_test_arg = np.argmax(y_test, axis=1)
confm = confusion_matrix(y_test_arg, y_pre_arg)  # 注意顺序：真实标签在前，预测在后

# 打印评价报告
print(classification_report(y_test_arg, y_pre_arg, target_names=categories))

# 混淆矩阵可视化（修复标签重叠）
plt.figure(figsize=(8, 8), dpi=600)
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False, linewidths=.8,
            cmap='YlGnBu')
plt.xlabel('真实标签', size=14)
plt.ylabel('预测标签', size=14)
plt.xticks(np.arange(10) + 0.5, categories, size=12, rotation=45, ha='right')
plt.yticks(np.arange(10) + 0.5, categories, size=12)  # 统一偏移量0.5，避免重叠
plt.savefig("1.png", bbox_inches='tight')
plt.show()
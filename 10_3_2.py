# 10.3.2 情感分析
# ===================== GPU加速核心配置 =====================
# 完整整合（GPU配置 + 训练参数）
import tensorflow as tf
import os
# 代码最开头添加
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 禁用所有GPU
import tensorflow as tf
print("✅ 已禁用GPU，使用CPU训练（规避CUDA环境故障）")


# 代码10-7 读取语料数据
import pandas as pd
from tensorflow.keras.preprocessing import sequence
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras import Input
import time
from sklearn import metrics
# 读取正负情感语料
neg = pd.read_excel('../data/neg.xls', header=None, index_col=None)
pos = pd.read_excel('../data/pos.xls', header=None, index_col=None)

# 给训练语料贴标签
pos['mark'] = 1
neg['mark'] = 0

# 代码10-8 词语
# 分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all = pd.concat([pos, neg], ignore_index=True)  # 合并正负情感语料
# 先确保所有输入都是字符串类型
pn_all[0] = pn_all[0].astype(str)

# 然后应用分词
cut_word = lambda x: list(jieba.cut(x))  # 定义分词函数
pn_all['words'] = pn_all[0].apply(cut_word)  # 对情感语料分词
comment = pd.read_excel('../data/sum.xls')  # 读入评论内容,增加语料
comment = comment[comment['rateContent'].notnull()]  # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cut_word)  # 对评论语料分词
pn_comment = pd.concat([pn_all['words'], comment['words']], ignore_index=True)  # 合并所有的数据

# 正负情感评论词语向量化
w = [] 
for i in pn_comment:
    w.extend(i)    
dicts = pd.DataFrame(pd.Series(w).value_counts())  # 建立统计词典
del w, pn_comment  # 删除临时文件 w，d2v_train
dicts['id'] = list(range(1, len(dicts)+1))
get_sent = lambda x: list(dicts['id'][x])
pn_all['sent'] = pn_all['words'].apply(get_sent)

# 评论词语向量标准化，对样本进行padding填充和truncating修剪
maxlen = 50  # 设置评论词语最大长度
pn_all['sent'] = list(sequence.pad_sequences(pn_all['sent'], maxlen=maxlen))  # 正负情感评论词语向量化

# 训练集、测试集
x_all = np.array(list(pn_all['sent']))
y_all = np.array(list(pn_all['mark']))
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.25)

print('训练集的特征数据形状为：', x_train.shape)
print('训练集的标签数据形状为：', y_train.shape)
print('测试集的特征数据形状为：', x_test.shape)
print('测试集的标签数据形状为：', y_test.shape)
print('训练集的特征数据为：\n', x_train)

# 代码10-9 模型构建
# 搭建LSTM模型
def TextRNN():
    model = tf.keras.Sequential()
    # 降低Embedding维度（256→64）
    model.add(tf.keras.layers.Embedding(len(dicts)+1, 64, input_length=50))
    # 降低LSTM维度（128→32）
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dropout(0.3))  # 降低dropout，减少计算
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model

# 代码10-10 模型训练
# 设置超参
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
timeA = time.time()
# 模型训练（减小batch_size）
model.fit(
    x_train, y_train,
    batch_size=4,  # 关键：从128/256降到16，降低显存占用
    epochs=10,
    shuffle=False,
    use_multiprocessing=False,  # 关闭多进程，避免显存碎片化
    workers=0
)
timeB = time.time()
print('time cost: ', int(timeB-timeA))

# 代码10-11 模型测试
# 使用测试数据进行模型测试
y_pred = model.predict(x_test).round().astype(int)
# 模型评价
acc = metrics.accuracy_score(y_test, y_pred)
print('测试集的准确率为：', acc)
print('精确率，召回率，F1值分别为：')
print(metrics.classification_report(y_test, y_pred))
print('混淆矩阵为：')
cm = metrics.confusion_matrix(y_test, y_pred)  # 混淆矩阵
print(cm)
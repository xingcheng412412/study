# 基本库
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
import numpy as np
import random
import tensorflow
import glob
from collections import Counter
# 搭建分类模型所需要的库
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tqdm import tqdm
# 加载音频处理库
import librosa
import librosa.display


def set_seeds(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
set_seeds()
feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

##提取训练集数据特征
def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)  # 计算mfcc,并把它作为特征
            feature.extend([mels])
    return [feature, label]

##提取测试集数据特征
def extract_features_test(test_dir, file_ext="*.wav"):
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):  # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        mels = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=128).T, axis=0)  # 计算mfcc,并把它作为特征
        feature.extend([mels])
    return feature


def generate_train():
    # 自己更改目录
    parent_dir = r'./train'                          #########训练集文件夹路径

    sub_dirs = np.array(['aloe', 'burger', 'cabbage', 'candied_fruits',
                                 'carrots', 'chips', 'chocolate', 'drinks', 'fries',
                                 'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
                                 'pizza', 'ribs', 'salmon', 'soup', 'wings'])
    temp = extract_features(parent_dir, sub_dirs, max_file=1000)

    temp = np.array(temp)
    data = temp.transpose()
    np.save('fusai_train_mfcc_128', data)
    print(data.shape)
    # 获取特征
    X = np.vstack(data[:, 0])
    # 获取标签
    Y = np.array(data[:, 1])
    print('X的特征尺寸是：', X.shape)
    print('Y的特征尺寸是：', Y.shape)
    a = np.load('./fusai_train_mfcc_128.npy', allow_pickle=True)
    print(a.shape)
    Y = to_categorical(Y)


def generate_test():
    X_test = extract_features_test('./test_b')       #########测试集文件夹路径
    X_test = np.vstack(X_test)
    temp = np.array(X_test)
    np.save('./fusai_test_mfcc_128', temp)
    print(temp.shape)

generate_train()   #生成128维的训练集mfcc数据
generate_test()    #生成128维的测试集mfcc数据

for _ in tqdm(range(6677, 6687)):  ###生成10个模型用于融合，类似于10折效果
    print(_)
    a = np.load('./fusai_train_mfcc_128.npy', allow_pickle=True)
    # 获取特征
    X = np.vstack(a[:, 0])
    # 获取标签
    Y = np.array(a[:, 1])
    Y = to_categorical(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=int(_), stratify=Y,
                                                        test_size=0.1)  ###通过随机数种子，按照9:1的比例划分训练集和验证集
    X_train = X_train.reshape(-1, int(X.shape[-1]), 1)
    X_test = X_test.reshape(-1, int(X.shape[-1]), 1)

    model = Sequential()
    input_dim = (int(X.shape[-1]), 1)
    model.add(Conv1D(16, (3), padding="same", activation="relu", input_shape=input_dim))  # 卷积层
    model.add(Conv1D(16, (3), padding="same", activation="relu"))  # 卷积层
    model.add(Conv1D(16, (3), padding="same", activation="relu"))  # 卷积层
    model.add(BatchNormalization())  # BN层
    model.add(Dropout(0.52, seed=66))
    model.add(Flatten())  # 展开
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.52, seed=66))
    model.add(Dense(20, activation="softmax"))  # 输出层：20个units输出20个类的概率

    # 编译模型，设置损失函数，优化方法以及评价标准
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    filepath = "weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    # model.load_weights("weights_best.hdf5")
    history = model.fit(X_train, Y_train, epochs=1234, batch_size=128, validation_data=(X_test, Y_test),
                        callbacks=callbacks_list, verbose=0)  # validation_split = 0.2
    max_val = "{:.4f}".format(max(history.history['val_accuracy']))

    new_name = str(max_val) + "_" + str(_) + '_' + "weights_best.hdf5"
    os.rename(filepath, new_name)

########## 使用10个模型，生成10个预测表格 ##########
best_list = glob.glob('./*.hdf5')
print(best_list)

for index, _ in enumerate(best_list):
    model = load_model(_)  # 加载模型准备预测
    X_test = np.load('./fusai_test_mfcc_128.npy', allow_pickle=True)
    predictions = model.predict(X_test.reshape(-1, 128, 1))
    preds = np.argmax(predictions, axis=1)
    preds = [label_dict_inv[x] for x in preds]

    path = glob.glob('./test_b/*.wav')
    result = pd.DataFrame({'name': path, f'{index}_label': preds})

    result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
    result.to_csv(f'{index}_submit.csv', index=None)

########################## 对10个预测表格进行简单的投票，筛选出频率最高的值，作为最终预测结果 ##############################
aa = pd.read_csv('0_submit.csv')
for _ in range(1, 10):##############
    aa = pd.merge(aa, pd.read_csv(f'{_}_submit.csv'), on='name')

label = aa.columns.drop('name')
all_label = pd.DataFrame(aa, columns=label)

label_merge = []
for _ in all_label.values:
    c = Counter(_)
    label_merge.append(c.most_common(1)[0][0])

result_merge = pd.DataFrame({'name': aa['name'], 'label': label_merge})
result_merge.to_csv('submit.csv', index=None)
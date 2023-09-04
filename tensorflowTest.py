import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os

# tensorflow Log 출력 제어
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 난수 생성 방식을 고정
np.random.seed(100)
tf.random.set_seed(100)

# data load , drop, split
df = pd.read_csv("data\\data_Advertising.csv")
df.drop(columns=['Unnamed: 0'])
X = df.drop(columns=['Sales'])
Y = df['Sales']
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)

# Pandas의 DataFrame을 Tensorflow의 Dataset 형태로 변환
train_ds = tf.data.Dataset.from_tensor_slices((train_X.values, train_Y.values))

# shuffle로 섞어주고 Batch 나누기
train_ds = train_ds.shuffle(len(train_X)).batch(batch_size=5)

# take method로 batch를 꺼내서 feature와 label에 나워서 저장하기
[(train_features_batch, label_batch)] = train_ds.take(1)

print('\nFB, TV, Newspaper batch Data:\n',train_features_batch)
print('Sales batch Data:',label_batch)
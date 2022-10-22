'''
자동차 연비 예측하기 : 회귀
=>  회귀(regression)는 가격이나 확률 같이 연속된 출력값을 예측하는것이 목적이다.
    이와 달리 분류(classification)는 여러개의 클래스 중 하나의 클래스를 선택하는것이 목적이다.
    아래 코드는 Auto MPG 데이터셋을 사용하여 1970년대 후반과 1980년대 초반의 자동차 연비를 예측한다.
    이 기간에 출시된 자동차 정보를 모델에 제공할것이다.
    이 정보에는 '실린더 수', '배기량', '마력', '공차 중량'같은 속성이 포함된다.
'''

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__, "\n")

##  Auto MPG 데이터셋
#   이 데이터셋은 UCI 머신 러닝 저장소에서 다운로드 할 수 있다.
#   1. 데이터 구하기

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

#       판다스를 사용하여 데이터를 읽는다.

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail(), "\n")

#   2. 데이터 정제하기
#      이 데이터셋은 일부 데이터가 누락되어있다.
print(dataset.isna().sum(), "\n")

#      문제를 간단하게 만들기 위해 누락된 행을 삭제해보자.
dataset = dataset.dropna()

#      "Origin"열은 수치형이 아니고 범주형이므로 "ont-hot encodding"으로 변환해보자.
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japen'] = (origin == 3)*1.0
print(dataset.tail(), "\n")

#   3. 데이터셋을 훈련세트와 테스트세트로 분할하기
#      이제 데이터를 훈련세트와 테스트세트로 분할해보자. 테스트 세트는 모델을 최종적으로 평가할 때 사용

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#   4. 데이터 조사하기
#      훈련 세트에서 몇 개의 열을 선택해 산점도 행렬을 만들어 살펴보자.
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

#      전반적인 통계도 확인해보자.
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)

#   5. 특성과 레이블 분리하기
#      특성에서 타깃 값 또는 '레이블'을 분리한다. 이 레이블을 예측하기 위해 모델을 훈련시킬것이다.
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

#   6. 데이터 정규화
#      위 train_stats 통계를 다시 살펴보고 각 특성의 범위가 얼마나 다른지 확인해보자.
#      특성의 스케일과 범위가 다르면 '정규화(normalization)'하는것이 권장된다.
#      특성을 정규화하지 않아도 모델이 수렴할 수 있지만, 훈련시키기가 어렵고 
#      입력단위에 의존적인 모델이 만들어진다.
#      note ; 의도적으로 훈련세트만 사용하여 통계치를 생성하였다.
#             이 통계는 테스트 세트를 정규화 할 때에도 사용된다.
#             이는 테스트 세트를 모델이 훈련에 사용했던 것과 동일한 분포로 투영하기 위함이다.

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#      정규화된 데이터를 사용하여 모델을 훈련한다.
#      주의 : 여기에서 입력 데이터를 정규화하기 위해 사용한 통계치(평균과 표준편차)는
#             'one-hot encoding'과 마찬가지로 모델에 주입되는 모든 데이터에 적용되어야 한다.
#             여기에는 테스트세트는 물론 모델이 실전에 투입되어 얻은 라이브 데이터도 포함된다.

##  모델
#   1.  모델 만들기
#       모델을 구성해 보자.
#       여기에서는 두 개의 완전연결(Densely connected) 은닉층으로 'Sequential'모델을 만들것이다.
#       출력층은 하나의 연속적인 값을 반환한다.
#       나중에 두 번째 모델을 만들기 쉽도록 'build_model'함수로 모델 구성 단계를 감쌀것이다.

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
        ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()

print(model.summary(),"\n")

#       모델을 한번 실행해 보자. 
#       훈련세트에서 10개의 샘플을 하나의 배치로 만들어 model.predict 메서드를 호출해볼것이다.

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

#   2.  모델 훈련
#       이 모델을 1000번의 에포크동안 훈련한다.
#       훈련 정확도와 검증 정확도는 history 객체에 기록된다.

# 에프코가 끝날 때마다 dot(.)을 출력하여 훈련 진행 과정을 표시.
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS=1000

history = model.fit(normed_train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

#       history 객체에 저장된 통계치를 사용해 모델의 훈련 과정을 시각화해보자.

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail(), "\n")

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8, 12))

    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
            label = 'Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)

#       이 그래프를 보면 수백번 에포크를 진행한 후에는 모델이 거의 향상되지 않는 것 같다.
#       model.fit 메서드를 수정하여 검증점수가 향상되지 않으면 자동으로 훈련을 멈추도록 만들어보자.
#       에포크마다 훈련 상태를 점검하기 위해 EarlyStopping callback을 사용해보자.
#       지정된 에포크 횟수 동안 성능 향상이 없으면 자동으로 훈련이 멈춘다.

model = build_model()

# patience 매개변수는 성능 향상을 체크할 에포크 횟수이다.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data,
                    train_labels,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)

#       이 그래프를 보면 검증세트의 평균 오차가 약 +/-2 MPG이다.

#       모델을 훈련할 때 사용하지 않았던 테스트 세트에서 모델의 성능을 확인해보자.
#       이를 통해 모델이 실전에 투입되었을 때 모델의 성능을 짐작할 수 있다.

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차 : {:5.2f} MPG".format(mae))

##  예측
#   마지막으로 테스트 세트에 있는 샘플을 사용해 MPG값을 예측해 보자.

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

plt.clf()
#   모델이 꽤 잘 예측한 것 같다. 다음으로 오차의 분포를 살펴보자.

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error[MPG]')
_ = plt.ylabel('Count')
plt.show()

#   가우시안 분포가 아니지만 아마도 훈련 샘플의 수가 매우 작기 때문일것이다.

'''
결론 : 이 노트북은 회귀 문제를 위한 기법을 소개한다.
       1. 평균 제곱 오차(Mean Square Error;MSE)는 회귀 문제에서 자주 사용하는 손실함수이다.
          (분류 문제에서 사용하는 손실함수와 다르다)
       2. 비슷하게 회귀에서 사용되는 평가 지표도 분류와는 다르다.
          많이 사용하는 회귀 지표는 평균 절댓값 오차(Mean Absolute Error;MAE)이다.
       3. 수치 입력 데이터의 특성이 여러 가지 범위를 가질 때 동일한 범위가 되도록
          각 특성의 스케일을 독립적으로 조정해야 한다.
       4. 훈련 데이터가 많지 않다면 과대적합을 피하기 위해
          은닉층의 개수가 적은 소규모 네트워크를 선택하는 방법이 좋다.
       5. 조기종료(Early stopping)은 과대적합을 방지하기 위한 좋은 방법이다.
'''

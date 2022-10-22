'''
영화 리뷰를 이용한 텍스트 분류 -- IMDB DATASET
=>  아래 코드는 영화리뷰 텍스트를 긍정 또는 부정으로 분류한다
    이 예제는 이진(binary)(또는 클래스가 두개인)분류 문제이다.
    이진 분류는 머신러닝에서 중요하고 널리 사용된다.
    여기에서는 인터넷 영화 데이터베이스(Internet Movie Database)에서 수집한
    50000개의 영화 리뷰 텍스트를 담은 IMDB 데이터셋을 사용할것이다.
    25000개의 리뷰는 훈련용으로, 25000개의 리뷰는 테스트용으로 나뉘어져 있다.
    훈련세트와 테스트세트의 클래스에는 균형이 잡혀있다.
    즉, 긍정적인 리뷰와 부정적인 리뷰의 개수가 동일하다.
'''

##  텐서플로우, 케라스 모듈 Import 후 IMDB데이터셋 다운로드
#   IMDB 데이터셋은 텐서플로우와 함께 제공된다.
#   리뷰(단어의 Sequence)는 미리 전처리 해서 정수 시퀀스로 변환되어있다.
#   각 정수는 어휘 사전에 있는 특정 단어를 의미한다.
#   매개변수 'num_words=10000'은 훈련 데이터에서 가장 많이 등장하는 상위 10000개의 단어를 선택.
import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)



##  데이터 탐색
#   잠시 데이터 형태를 알아보자. 이 데이터셋의 샘플은 전처리된 정수 배열이다.
#   이 정수는 영화 리뷰에 나오는 단어를 나타낸다.
#   레이블(label)은 정수 0 또는 1이다.
#   0은 부정적인 리뷰, 1은 긍정적인 리뷰이다.

print("훈련 샘플 : {}, 레이블 : {}".format(len(train_data), len(train_labels)))

#   리뷰 텍스트는 어휘 사전의 특정 단어를 나타내는 정수로 변환되어있다.
#   첫번째 리뷰를 확인해보자.

print(train_data[0])

#   영화리뷰들은 길이가 다르다. 다음 코드는 첫 번째 리뷰와 두 번째 리뷰에서 단어의 개수를 출력한다.
#   신경망의 입력은 길이가 같아야 하기 때문에 나중에 이 문제를 해결할것이다.

print(len(train_data[0])) # 218
print(len(train_data[1])) # 189


##  정수를 단어로 다시 변환하기
#   정수를 다시 텍스트로 변환하는 방법이 있다면 유용할 것이다.
#   여기에서는 정수와 문자열을 매핑한 딕셔너리 객체에 질의하는 헬퍼(helper)함수를 만들것이다.

# 단어와 정수 인덱스를 매핑한 딕셔너리

word_index = imdb.get_word_index()

# 처음 몇 개 인덱스는 사전에 정의되어있다.

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#   이제 decode_review 함수를 사용해 첫 번째 리뷰 텍스트를 출력해보자.

print(decode_review(train_data[0]))


##  데이터 준비
#   리뷰(정수배열)는 신경망에 주입되기 전에 텐서로 변환되어야 한다.
#   변환하는 방법에는 몇 가지가 있다.
#   1. ont-hot-encoding : 정수 배열을 0과 1로 이루어진 벡터로 변환한다.
#                         예를들어 배열 [3, 5]을 인덱스 3과 5만 1이고 나머지는 모두 0인
#                         10,000차원 벡터로 변환할 수 있다.
#                         그 다음 실수 벡터 데이터를 다룰수 있는 Dense층을 신경망의 첫번째 층으로 사용
#                         이 방법은 'num_words * num_reviews'크기의 행렬이 필요함
#                         --> 메모리를 많이 사용하게됨
#   2. 다른 방법으로는 정수 배열의 길이가 모두 같도록 padding을 추가하여
#      'max_length * num_reviews' 크기의 정수 텐서를 만든다.
#      이런 형태의 텐서를 다룰수있는 embedding 층을 신경망의 첫 번째 층으로 사용할 수 있다.
#   이 튜토리얼에서는 두 번째 방법을 사용할것이다.
#   영화 리뷰의 길이가 같아야 하므로 'pad_sequences' 함수를 사용해 길이를 맞춰보자.

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#   샘플의 길이를 확인해보자.

print(len(train_data[0])) # 256
print(len(train_data[1])) # 256

#   패딩된 첫번째 리뷰 내용을 확인해보자.

print(train_data[0])

##  모델 구성
#   신경망은 층(layer)을 쌓아서 만든다. 이 구조에서는 두 가지를 결정해야한다.
#   1. 모델에서 얼마나 많은 층을 사용할 것인가?
#   2. 각 층에서 얼마나 많은 은닉 유닛(hidden unit)을 사용할 것인가?
#   이 예제의 입력데이터는 단어 인덱스의 배열이다.
#   예측할 레이블은 0 또는 1 이다.
#   이 문제에 맞는 모델을 구성해보자.

# 입력 크기는 영화 리뷰 데이터셋에 적용된 어휘 사전의 크기이다.(1만개의 단어)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

#   층을 순서대로 쌓아 분류기(classifier)를 만든다.
#   1. 첫 번째 층은 'Embedding'층이다.
#      이 층은 정수로 인코딩된 단어를 입력받고 각 단어 인덱스에 해당하는 임베딩 벡터를 찾는다.
#      이 벡터는 모델이 훈련되면서 학습된다.
#      또한, 이 벡터는 출력배열에 새로운 차원으로 추가된다.
#      최종 차원은 (batch, sequence, embedding)이 된다.
#   2. 그 다음 'GlobalAveragePooling1D'층은 sequence 차원에 대해 평균을 계산하여
#      각 샘플에 대해 고정된 길이의 출력 벡터를 반환한다.
#      이는 길이가 다른 입력을 다루는 가장 간단한 방법이다.
#   3. 이 고정 길이의 출력 벡터는 16개의 은닉 유닛을 가진 완전연결층(Dense)을 거친다.
#   4. 마지막 층은 하나의 출력 노드를 가진 완전연결층이다.
#      sigmoid 활성화 함수를 사용하여 0과 1 사이의 실수를 출력한다.
#      이 값은 확률 또는 신뢰도를 나타낸다.

##  은닉 유닛
#   위 모델에는 입력과 출력 사이에 두 개의 중간(혹은 은닉)층이 있다.
#   출력(유닛 또는 노드,뉴런)의 개수는 층이 가진 표현 공간(representational space)의 차원이 된다.
#   다른 말로 하면, 내부 표현을 학습할 때 허용되는 네트워크 자유도의 양이다.
#   모델에 많은 은닉 유닛(고차원의 표현 공간)과 층이 있다면 네트워크는 더 복잡한 표현을 학습할 수 있다
#   하지만 네트워크의 계산 비용이 많이 들고 원치않는 패턴을 학습할 수도 있다.
#   이런 표현은 훈련 데이터의 성능을 향상시키지만 테스트 데이터에서는 그렇지 못한다.
#   이를 과대적합(overfitting)이라 한다.

##  손실함수와 옵티마이저
#   모델이 훈련하려면 '손실함수(loss function)'과 '옵티마이저(optimizer)'가 필요하다.
#   이 예제는 이진분류문제이고 모델이 확률을 출력하므로(출력층의 유닛이 하나이고 sigmoid함수 사용)
#   'binary_crossentropy' 함수를 사용할것이다.
#   다른 손실함수를 선택 할 수도 있다. 예를 들어 'mean_squared_error'을 선택 할 수도 있다.
#   하지만 일반적으로 'binary_crossentropy'가 확률을 다루는데 적합하다.
#   이 함수는 확률분포간의 거리를 측정한다.
#   여기에서는 정답인 타깃 분포와 예측 분포 사이의 거리이다.
#   나중에 회귀(regression)문제에 대해 살펴 볼 때 평균 제곱 오차 손실함수를 어떻게 사용하는지 알아보자
#   이제 모델이 사용할 옵티마이저와 손실 함수를 설정해보자.

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

##  검증 세트 만들기
#   모델을 훈련 할 때 모델이 만난 적 없는 데이터에서 정확도를 확인하는것이 좋다.
#   원본 훈련 데이터에서 10,000개의 샘플을 떼어네어 검증세트를 만들것이다.
#   왜 테스트 세트를 사용하지 않을까?
#   => 훈련 데이터만을 사용하여 개발하고 튜닝하는것이 목표이다.
#      그 다음 테스트 세트를 사용해서 딱 한번만 정확도를 평가한다.

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

##  모델 훈련
#   이 모델을 512개의 샘플로 이루어진 미니배치에서 40번의 에포크동안 훈련한다.
#   x_train과 y_train 텐서에 있는 모든 샘플에 대해 40번 반복한다는 뜻 이다.
#   훈련 하는 동안 10,000개의 검증 세트에서 모델의 손실과 정확도를 모니터링한다.

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

##  모델 평가
#   모델의 성능을 확인해보자. 두 개의 값이 반환된다.
#   손실과 정확도이다.

results = model.evaluate(test_data, test_labels, verbose=2)

print(results)

#   이 예제는 매우 단순한 방식을 사용하므로 87%의 정확도를 달성하였다.
#   고급 방법을 사용한 모델은 95%의 정확도를 얻는다.

##  정확도와 손실 그래프 그리기
#   model.fit()은 History 객체를 반환한다.
#   여기에는 훈련하는 동안 일어난 모든 정보가 담긴 딕셔너리가 들어있다.

history_dict = history.history
print(history_dict.keys())

#   4개의 항목이 있다. 훈련과 검증 단게에서 모니터링하는 지표들이다.
#   훈련 손실과 검증 손실을 그래프로 그려보고, 훈련 정확도와 검증 정확도도 그래프로 그려서 비교해보자.

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

# bo는 파란색 점이다.
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 파란색 실선이다.
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#   이 그래프에서 점선은 훈련 손실과 훈련 정확도를 나타낸다
#   실선은 검증 손실과 검증 정확도이다.
#   훈련 손실은 에포크마다 '감소'하고 훈련 정확도는 '증가'한다는것을 주목하자.
#   경사 하강법 최적화를 사용할 때 볼 수 있는 현상이다.
#   매 반복마다 최적화 대상의 값을 최소화한다.
#   하지만 검증 손실과 검증 정확도에서는 그렇지 못한다.
#   약 20번째 에포크 이후가 최적점 인 것 같다.
#   이는 과대적합 때문이다.
#   이전에 본 적 없는 데이터보다 훈련 데이터에서 더 잘 동작한다.
#   이 지점부터는 모델이 과도하게 최적화되어
#   테스트 데이터에서 일반화 되기 어려운 훈련 데이터의 특정 표현을 학습한다.
#   여기에서는 과대적합을 막기 위해 단순히 20번째 에포크 근처에서 훈련을 멈출 수 있다.
#   나중에 callback을 사용하여 자동으로 이렇게 하는 방법을 배워보자.


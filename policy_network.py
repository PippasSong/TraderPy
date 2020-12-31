import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import SGD


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr  # 기본 학습 속도(learning rate = 0.01)

        # LSTM 신경망
        self.model = Sequential()  # 전체 신경망을 구성하는 클래스

        self.model.add(LSTM(256, input_shape=(1, input_dim), return_sequences=True, stateful=False,
                            dropout=0.5))  # 256차원 드롭아웃을 50%로 정하여 과적합 피한다
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        # 최적화 알고리즘과 학습 속도를 정함. 기본 학습 알고리즘은 SGD(확률적 경사 하강법), 기본 학습 속도는 0.01
        self.model.compile(optimizer=SGD(lr=lr), loss='mse')
        self.prob = None

    def reset(self):
        self.prob = None

    # 신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17차원의 입력을 받아서 매수와 매도가 수익을 높일 것으로 판단되는 확률을 구한다.
    def predict(self, sample):
        # array() 함수는 파이썬 리스트를 n차원 배열(ndarray)형식으로 만든다
        # sample의 크기는 self.input_dim의 값인 17. 이 배열을 1행 17열인 2차원 배열로 변경
        # reshape 함수는 ndarray를 다른 차원으로 변환(-1:열은 알아서 재배열해준다)
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    # 입력으로 들어온 학습 데이터 집합 x와 레이블 y로 정책 신경망을 학습시킴. x, y는 정책 학습기에서 준비
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)
    
    #학습한 정책 신경망을 파일로 저장. model_path는 저장할 파일명을 의미
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True) # 인공 신경망을 구성하기 위한 값들을 HDF5파일로 저장
    
    # 저장한 정책 신경망을 불러오기 위한 함수  
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
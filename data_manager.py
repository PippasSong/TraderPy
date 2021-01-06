import pandas as pd
import numpy as np


# CSV 파일 경로를 입력으로 받는다
def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)  # thousands 파라미터로 ','를 넣어주면 천 단위로 콤마가 붙은 값을 숫자로 인식
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data


def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()  # rolling 함수는 window 크기만큼
        # 데이터를 묶어서 합, 평균, 표준편차 등을 계산할 수 있게 준비함
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean()
        )
    return prep_data


def build_training_data(prep_data):
    training_data = prep_data
    # 시가/전일종가 비율
    training_data['open_lastclose_ratio'] = np.zeros((len(training_data)))
    training_data['open_lastclose_ratio'].iloc[1:] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    # 고가/종가 비율
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    # 저가/종가 비율
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    # 종가/전일종가 비율
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data['close_lastclose_ratio'].iloc[1:] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    # 거래량/전일거래량 비율. 거래량 값이 0이면 이전의 0이 아닌 값으로 바꾸어 줌
    # ffill: 특정 값을 이전의 값으로 변경
    # bfill: 특정 값을 이후의 값으로 변경
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data['volume_lastvolume_ratio'].iloc[1:] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1] \
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values

    # 각 원도우에 대하여 이동평균 종가비율((현재종가-이동평균 값)/이동평균 값), 이동평균 거래량 비율을 구한다
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data

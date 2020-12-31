import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Visualizer:

    def __init__(self):
        self.fig = None  # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.axes = None  # 차트를 그리기 위한 Matplotib의 Axes 클래스 객체. fig에 포함되는 차트의 배열

    def prepare(self, chart_data):
        # 캔버스를 초기화하고 4개의 차트를 그릴 준비

        # 4행 1열의 구조를 가지는 figure 생성. nrows: 행 개수, ncols: 열 개수 sharex: 각 그래프가 x축의 범위 공유
        # 2개의 변수를 튜플로 반환. 1. Figure객체 2. Figure객체에 포함된 Axes 객체의 배열
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:
            # 보기 어려운 과학적 표기 비활성화
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
        # 차트 1. 일봉 차트
        self.axes[0].set_ylabel('Env.')  # y축 레이블 표시
        # 거래량 가시화
        x = np.arange(len(chart_data))  # arrange: 입력으로 들어온 값만큼 순차적으로 값을 생성해 배열로 반환
        volume = np.array(chart_data)[:, -1].tolist()  # array를 슬라이스 [표시할 행,표시할 열] ':': 모든 행 또는 열 '-1': 마지막 index-1(
        # len-2)의미
        self.axes[0].bar(x, volume, color='b', alpha=0.3)  # 거래량 표시 위해 막대 차트를 그림
        # ohlc란 open, high, low, close의 약자로 이 순서로 구성된 2차원 배열
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))  # N*5차원의 배열. 첫번째 열은 인덱스, N은 일봉의 수. 1차원
        # 인덱스 배열을 만들고 chart_data의 4번째 열까지를 수평적으로 붙인다

        # self.axes[0]에 봉 차트 출력
        # 양봉은 빨간색으로, 음봉은 파란색으로 표시
        candlestick_ohlc(ax, ohlc, colorup='r', colordown='b')  # Axes객체, ohlc데이터 입력 뒤는 옵션

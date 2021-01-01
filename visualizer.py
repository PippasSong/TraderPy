import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc

from agent import Agent


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

    # epoch_str: figure 제목으로 표시할 에포크
    # num_epoches: 총 수행할 에포크 수
    # epsilon: 탐험률
    # action_list: 에이전트가 수행할 수 있는 전체 행동 리스트
    # actions: 에이전트가 수행한 행동 배열
    # num_stocks: 주식 보유 수 배열
    # outvals: 정책 신경망의 출력 배열
    # initial_balance: 초기 자본금
    # pvs: 포트폴리오 가치 배열

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, action_list=None, actions=None, num_stocks=None,
             outvals=None, exps=None, initial_balance=None, learning_idxes=None, pvs=None):
        x = np.arange(len(actions))  # 모든 차트가 공유할 x축 데이터
        actions = np.array(actions)  # 에이전트의 행동 배열
        outvals = np.array(outvals)  # 정책 신경망의 출력 배열
        pvs_base = np.zeros(len(actions)) + initial_balance  # 초기 자본금 배열. zeros: 인자로 배열의 형태인 shape를 받아서 0으로 구성된
        # NumPy배열을 반환. 다차원 배열의 경우 튜플로 넘겨야 한다

        # 차트 2. 에이전트 상태 (행동, 보유 주식 수)
        colors = ['r', 'b']
        for actiontype, color in zip(action_list, colors):  # zip: 두 개의 배열에서 같은 인덱스의 요소를 순서대로 묶어준다
            for i in x[actions == actiontype]:
                self.axes[1].axvline(i, color=color, alpha=0.1)  # 배경 색으로 행동 표시. x축 위치에서 세로로 선을 긋는 함수. alpha는 투명도
        self.axes[1].plot(x, num_stocks, '-k')  # 보유 주식 수 그리기. x축 데이터, y축 데이터, 스타일을 인자로 받는다

        # 차트 3. 정책 신경망의 출력 및 탐험
        for exp_idx in exps:
            # 탐험을 노란색 배경으로 그리기
            self.axes[2].axvline(exp_idx, color='y')
        for idx, outval in zip(x, outvals):
            color = 'white'
            if outval.argmax() == 0:
                color = 'r'  # 매수면 빨간색
            elif outval.argmax() == 1:
                color = 'b'  # 매도면 파란색
            # 행동을 빨간색 또는 파란색 배경으로 그리기
            self.axes[2].axvline(idx, color=color, alpha=0.1)
        styles = ['.r', '.b']  # 빨간 점, 파란 점 의미
        for action, style in zip(action_list, styles):
            # 정책 신경망의 출력을 빨간색, 파란색 점으로 그리기
            self.axes[2].plot(x, outvals[:, action], style)

        # 차트 4. 포트폴리오 가치
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs > pvs_base, facecolor='r', alpha=0.1)
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor='b', alpha=0.1)
        self.axes[3].plot(x, pvs, '-k')  # 포트폴리오 가치를 검정색 실선으로 그린다
        # 학습 위치 표시
        for learning_idx in learning_idxes:
            self.axes[3].axvline(learning_idx, color='y')

        # 에포크 및 탐험 비율
        self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))  # 문자열에 값 넣기
        # 캔버스 레이아웃 조정
        plt.tight_layout()  # figure의 크기에 알맞게 내부 차트들의 크기를 조정해 준다
        plt.subplots_adjust(top=0.85)

    # xlim: 모든 차트의 x축 값 범위를 설정해 줄 튜플
    # 학습 과정에서 변하지 않는 환경에 관한 차트를 제외하고 그 외 차트들을 초기화
    def clear(self, xlim):
        for ax in self.axes[1:]:
            ax.cla()  # 그린 차트 지우기
            ax.relim()  # limit를 초기화
            ax.autoscale()  # 스케일 재설정
        # y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('PG')
        self.axes[3].set_ylabel('PV')
        for ax in self.axes:
            ax.set_xlim(xlim)  # x축 limit 재설정
            ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            ax.get_yaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            # x축 간격을 일정하게 설정
            ax.ticklabel_format(useOffset=False)

    def save(self, path):
        plt.savefig(path)  # Figure을 그림파일로 저장

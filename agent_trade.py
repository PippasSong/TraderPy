import numpy as np


class Agent:
    # 클래스 속성들 모든 클래스가 공유한다
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 에이전트 상태의 차원. 주식 보유 비율, 포트폴리오 가치 비율로 2가지 이므로 2차원이다

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 미고려(일반적으로 0.015%)
    TRADING_TAX = 0.003  # 거래세 미고려(실제 0.3%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들. list로, 수정 가능하다
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연 보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연 보상 임계치

        # Agent클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance+num_stocks*{현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수
        self.immediate_reward = 0  # 즉시 보상

        # Agent 클래스의 상태
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 한 에포크마다 에이전트의 상태를 초기화
    def reset(self, cur_stock_value=0, num_stocks=0):
        self.balance = self.initial_balance
        self.num_stocks = num_stocks
        self.portfolio_value = self.initial_balance + cur_stock_value  # 307927 # balance+num_stocks*{현재 주식 가격}
        self.base_portfolio_value = self.portfolio_value  # 직전 학습 시점의 PV. initial balance가 과거의 balance가 되도록 해야 함. 매개변수로 과거 주식을 받을게 하니라 전체 포트폴리오를 받아 대입할 것. 에이전트 파일을 하나 더 만들기 or 지연보상을 고려하지 않으므로 무시?(포트폴리오 가치와 같은 값 주기)
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 에이전트의 초기 자본금을 설정
    def set_balance(self, balance):
        self.initial_balance = balance

    # 직전 학습 시점의 pv를 설정
    def set_base_portfolio_value(self, base_portfolio_value):
        self.base_portfolio_value = base_portfolio_value

    # 에이전트의 상태를 반환
    def get_states(self):
        # 주식 보유 비율 = 보유 주식 수/(포트폴리오 가치/현재 주가)---->최대 보유 가능한 주식 중 얼마나 보유하고 있는가
        self.ratio_hold = self.num_stocks / int(self.portfolio_value / self.environment.get_price())
        # 포트폴리오 가치 비율 = 포트폴리오 가치 / 기준 포트폴리오 가치---->0에 가까울수록 큰 손실 1보다 크면 수익 발생 의미
        # 기준 포트폴리오 가치는 직전에 목표 수익 또는 손익률을 달성했을 때의 포트폴리오 가치
        # 수익률이 목표 수익률에 가까우면 매도의 관점에서 투자
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value

        # 튜플로 반환
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # 엡실론의 확률로 무작위로 행동을 결정, 그렇지 않은 경우 정책 신경망을 통해 행동을 결정
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        if np.random.rand() < epsilon:  # rand : 0~1사이의 값을 생성하여 반환
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정
        else:
            exploration = False
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            action = np.argmax(probs)  # argmax:array에서 가장 큰 값의 위치(index)를 반환
            confidence = probs[action]
        return action, confidence, exploration

    # 행동의 유효성 판단. 결정을 했는데 포트폴리오에서 그 결정을 따를 자산이 있는가
    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                validity = False
        return validity

    # 매수/매도 단위 결정 함수. 정책 신경망이 결정한 행동의 확률이 높을수록 매수 또는 매도하는 단위를 크게 정한다
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(int(confidence * (self.max_trading_unit - self.min_trading_unit)),
                                self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading

    # 투자 행동 수행 함수
    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD  # 유효한 행동이 아니면 유보

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit)

            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신
            self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
            self.balance += invest_amount  # 보유 현금을 갱신
            self.num_sell += 1  # 매도 횟수 증가

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        # 기준 포트폴리오 가치(학습을 수행한 시점의 포트폴리오 가치)에서 현재 포트폴리오 가치의 등락률을 계산
        profitloss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value

        # 즉시 보상 판단
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률을 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward

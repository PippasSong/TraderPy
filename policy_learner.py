import os
import locale
import logging
import numpy as np
import settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer

locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None, min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체
        # 에이전트 객체
        self.agent = Agent(self.environment, min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data  # 학습 데이터
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망. 입력 크기 = 학습 데이터의 크기 + 에이전트 상태 크기
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # 가시화 모듈

    # 학습 데이터를 다시 읽기 위해 idx를 -1로 재설정
    def reset(self):
        self.sample = None
        self.training_data_idx = -1

    # num_epoches: 수행할 반복 학습의 전체 횟수
    # max_memory: 배치 학습 데이터를 만들기 위해 과거 데이터를 저장할 배열
    # balance: 에이전트의 초기 투자 자본금을 정해주기 위한 인자
    # discount_factor: 먼 과거의 행동일수록 할인 요인을 적용하여 지연 보상을 약하게 적용
    # start_epsilon: 초기 탐험 비율
    # learning: 학습 유무, 학습된 모델을 가지고 투자 시뮬레이션만 하려 한다면 False.
    # init_stocks: 초기에 보유한 주식의 수
    def fit(self, num_epoches=1000, max_memory=60, balance=1000000, discount_factor=0, start_epsilon=.5, learning=True, past_stock_value=0, cur_stock_value=0, init_stocks=0):
        logging.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}],"
                    "DRT: {delayed_reward_threshold}".format(lr=self.policy_network.lr, discount_factor=discount_factor,
                                                             min_trading_unit=self.agent.min_trading_unit,
                                                             max_trading_unit=self.agent.max_trading_unit,
                                                             delayed_reward_threshold=self.agent.
                                                             delayed_reward_threshold))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        # join(parent, child) 로 폴더 생성
        epoch_summary_dir = os.path.join(settings.BASE_DIR,
                                         'epoch_summary/%s/epoch_summary_%s' % (
                                             self.stock_code, settings.timestr))  # 가시화 결과를 저장시 날짜와 시간 지정
        if not os.path.isdir(epoch_summary_dir):  # path 가 존재하고 폴더인지 확인
            os.makedirs(epoch_summary_dir)  # path 에 포함된 폴더들이 없을 경우 생성해 줌

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)


        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # 에포크 관련 정보 초기화
            loss = 0.  # 정책 신경망의 결과가 학습 데이터와 얼마나 차이가 있는지를 저장
            itr_cnt = 0  # 수행한 에포크 수를 저장
            win_cnt = 0  # 수행한 에포크 중에서 수익이 발생한 에포크 수를 저장
            exploration_cnt = 0  # 무작위 투자를 수행한 횟수를 저장
            batch_size = 0
            pos_learning_cnt = 0  # 수익이 발생하여 긍정적 지연 보상을 준 수
            neg_learning_cnt = 0  # 손실이 발생하여 부정적 지연 보상을 준 수

            # 메모리 초기화
            memory_sample = []
            memory_action = []
            memory_reward = []  # 즉시보상
            memory_prob = []  # 정책 신경망의 출력
            memory_pv = []  # 포트폴리오 가치
            memory_num_stocks = []  # 보유 주식 수
            memory_exp_idx = []  # 탐험 위치
            memory_learning_idx = []  # 학습 위치

            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset(cur_stock_value, init_stocks)
            self.policy_network.reset()
            self.reset()

            # 가시화기 초기화
            self.visualizer.clear([0, len(self.chart_data)])  # x축 데이터 범위를 파라미터로 넣어준다

            # 학습을 진행할수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self._build_sample()
                if next_sample is None:
                    break

                # 정책 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(self.policy_network, self.sample, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                memory_sample.append(next_sample)
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)

                # 학습 데이터의 샘플, 에이전트 행동, 즉시보상, 포트폴리오 가치, 보유 주식 수를 저장하는 2차원 배열
                memory = [(memory_sample[i], memory_action[i], memory_reward[i])
                          for i in list(range(len(memory_action)))[-max_memory:]]
                if exploration:
                    memory_exp_idx.append(itr_cnt)  # 무작위 행동을 결정한 경우 현재의 인덱스를 저장
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS)  # 무작위 투자에서는 정책 신경망의 출력이 없기 때문에 nan값을 넣는다. 리스트에
                    # 곱하기를 하면 똑같은 리스트를 뒤에 붙여준다
                else:
                    memory_prob.append(self.policy_network.prob)  # 무작위 투자가 아닌 경우 신경망의 출력을 그대로 저장

                # 반복에 대한 정보 갱신
                batch_size += 1
                itr_cnt += 1
                exploration_cnt += 1 if exploration else 0  # 탐험을 한 경우에만 1을 증가시키고 그렇지 않으면 0을 더함
                win_cnt += 1 if delayed_reward > 0 else 0  # 지연 보상이 0보다 큰 경우에만 1을 증가시킴

                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:
                    # 배치 학습 데이터 크기
                    batch_size = min(batch_size, max_memory)
                    # 배치 학습 데이터 생성
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    if len(x) > 0:
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # 정책 신경망 갱신
                        loss += self.policy_network.train_on_batch(x, y)
                        memory_learning_idx.append([itr_cnt, delayed_reward])
                    batch_size = 0  # 학습 수행 후 배치 데이터 크기를 초기화

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')  # 문자열을 자리수에 맞게 오른쪽으로 정렬해 주는 함수

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning_idxes=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            # 수행 결과를 파일로 저장
            self.visualizer.save(
                os.path.join(epoch_summary_dir, 'epoch_summary_%s_%s.png' % (settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            # 콘솔창에 뜨는 정보
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logging.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t(%s+%s*%s)\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),self.agent.balance, self.environment.get_price(), self.agent.num_stocks,
                            pos_learning_cnt, neg_learning_cnt, loss
                        ))

            # 학습 관련 정보 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 학습 관련 정보 로그 기록
        logging.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt
        ))

    # 미니 배치 데이터 생성
    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):
        x = np.zeros((batch_size, 1, self.num_features))  # 특징벡터를 지정
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)  # 지연 보상으로 정답(레이블)을 설정
        for i, (sample, action, reward) in enumerate(reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features))
            y[i, action] = (delayed_reward + 1) / 2
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i
        return x, y

    # 학습 데이터를 구성하는 샘플 하나를 생성
    def _build_sample(self):
        self.environment.observe()  # 다음 인덱스 데이터를 읽도록 한다
        if len(
                self.training_data) > self.training_data_idx + 1:  # 다음 인덱스 데이터가 존재하면 training_data_idx인덱스의 데이터를 받아와서
            # sample로 저장
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    # 학습된 정책 신경망 모델로 주식투자 시뮬레이션
    # init_stocks: 초기에 보유한 주식 수
    def trade(self, model_path=None, balance=2000000, cur_stock_value=0, init_stocks=0):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)  # 학습된 신경망 모델을 적용
        self.fit(balance=balance, num_epoches=1, learning=False, cur_stock_value=cur_stock_value, init_stocks=init_stocks)  # 학습을 진행하지 않고 정책 신경망에만 의존하여 투자 시뮬레이션을 진행

import logging
import sys
import os.path
import settings

stock_code = '217270'  # 넵튠
model_ver = '20210107181206'

# 로그 기록
log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
timestr = settings.get_time_str()
if not os.path.exists('logs/%s' % stock_code):
    os.makedirs('logs/%s' % stock_code)
file_handler = logging.FileHandler(filename=os.path.join(
    log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
file_handler.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s",
                    handlers=[file_handler, stream_handler], level=logging.DEBUG)

import data_manager
from policy_learner import PolicyLearner

# 파이썬의 main 구문
if __name__ == '__main__':
    # 강화학습에 사용할 주식 데이터를 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code))
    )
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2021-01-08') & (training_data['date'] <= '2021-01-08')]
    training_data = training_data.dropna()

    # 준비한 주식 데이터를 차트 데이터와 학습 데이터로 분리
    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio',
    ]
    training_data = training_data[features_training_data]

    # 비학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data, min_trading_unit=1,
        max_trading_unit=10)
    policy_learner.trade(balance=52777, model_path=os.path.join(settings.BASE_DIR,
                                                                 'model\{}\model_{}.h5'.format(stock_code, model_ver)),
                         cur_stock_value=9 * 27650, init_stocks=9)

    # 정책 신경망을 파일로 저장. 추가적인 학습을 수행하여 모델을 새로 저장하고 싶다면 코드 블록을 그대로 두면 된다
    # model_dir = os.path.join(settings.BASE_DIR, 'model/%s' % stock_code)
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)
    # model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    # policy_learner.policy_network.save_model(model_path)

class Environment:
    PRICE_IDX = 4  # 종가의 위치

    # 생성자
    def __init__(self, chart_data=None):
        self.chart_data = chart_data  # 주식 종목의 차트 데이터
        self.observation = None  # 현재 관측치
        self.idx = -1  # 차트 데이터에서의 현재 위치

    # idx와 observation을 초기화
    def reset(self):  # self는 객체의 인스턴스 그 자체를 말한다. 즉 객체 자기 자신을 참조하는 매개변수. 인스턴스를 사용하면 self는 파이썬이 자동으로 사용해서 명시할 필요 없다.
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + 1:  # 차트 데이터가 현재 위치보다 많은 경우
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]  # iloc: []행의 데이터를 가저온다
            return self.observation
        return None
    # 현재 observation에서 종가를 획득. 종가 close의 위치가 5번째 열이기 때문에 PRICE_IDX값은 4이다
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

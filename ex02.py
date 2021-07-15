# pandas 라이브러리 불러오기
import pandas as pd
# 랜덤 포레스트 불러오기
from sklearn.ensemble import RandomForestRegressor

# data폴데이 있는 csv 파일 데이터 불러오기

train = pd.read_csv('따릉이/train.csv') 
test = pd.read_csv('따릉이/test.csv')

# isnull() 메소드는 관측치가 결측이면 True, 결측이 아니면 False의 boollean 값을 반환한다
# notnull() 메소드는 isnull()과 반대
#print(train.isnull().sum())

# 원하는 피쳐의 결측치를 해당 피쳐의 평균값으로 대체하는 코드
#--> df.fillna({'칼럼명':int(df['칼럼명'].mean)}, inplace=True)
#train.fillna({'hour_bef_temperature':int(train['hour_bef_temperature'].mean())}, inplace=True)
#train.fillna({'hour_bef_precipitation':int(train['hour_bef_precipitation'].mean())}, inplace=True)
#train.fillna({'hour_bef_windspeed':int(train['hour_bef_windspeed'].mean())}, inplace=True)
#train.fillna({'hour_bef_humidity':int(train['hour_bef_humidity'].mean())}, inplace=True)
#train.fillna({'hour_bef_visibility':int(train['hour_bef_visibility'].mean())}, inplace=True)
#train.fillna({'hour_bef_ozone':int(train['hour_bef_ozone'].mean())}, inplace=True)
#train.fillna({'hour_bef_pm10':int(train['hour_bef_pm10'].mean())}, inplace=True)
#train.fillna({'hour_bef_pm2.5':int(train['hour_bef_pm2.5'].mean())}, inplace=True)
#print(train.isnull().sum()) 피쳐 평균값 대체 출력

# 결측치 보간 대체 함수로 전처리. 결측치란, 누락된 데이터를 말한다.
# 보간법이란 알고 있는 데이터 값들을 이용하여 모르는 값을 추정하는 방법의 한 종류이다
# 이 보간법이 fillna()에 비해 더 지능적이다.

train.interpolate(inplace=True)
test.fillna(0, inplace=True)
#print(train.isnull().sum())

# count 피쳐를 제외한 X_train df를 생성하는 코드
X_train = train.drop(['count'], axis=1)

# count 피쳐만을 가진 Y_train df를 생성하는 코드
Y_train = train['count']

# 랜덤포레스트 모듈의 옵션 중 criterion 옵션을 통해 
# 어떤 평가척도를 기준으로 훈련할 것인지 정할 수 있다
# RMSE 는 MSE 평가지표의 루트를 씌운 것으로서, 
# 모델을 선언할 때 criterion = ‘mse’ 옵션으로 구현할 수 있습니다.
model = RandomForestRegressor(criterion = 'mse')

#fit으로 모델 학습
model.fit(X_train, Y_train)
#print(model)

# 랜덤포레스트모델 예측변수의 중요도를 출력하는 코드
model.feature_importances_

# fit() 으로 모델이 학습되고 나면 feature_importances_ 속성(attribute) 으로 
# 변수의 중요도를 파악할 수 있습니다.변수의 중요도란 예측변수를 결정할 때 
# 각 피쳐가 얼마나 중요한 역할을 하는지에 대한 척도입니다.
# 변수의 중요도가 낮다면 해당 피쳐를 제거하는 것이 모델의 성능을 높일 수 있습니다.
#print(model.feature_importances_)

# X_train 에서 drop(제거) 할 피쳐의 경우에 수 대로 3개의 X_train 을 생성.

X_train_1 = train.drop(['count','id'], axis=1)
X_train_2 = train.drop(['count', 'id', 'hour_bef_windspeed'], axis=1)
X_train_3 = train.drop(['count', 'id', 'hour_bef_windspeed', 'hour_bef_pm2.5'], axis=1)

# 각 train 에 따라 동일하게 피쳐를 drop 한 test 셋들을 생성.
# 예측을 할 때 test 는 훈련 셋과 동일한 피쳐를 가져야 한다

test_1 = test.drop(['id'], axis=1)
test_2 = test.drop(['id', 'hour_bef_windspeed'], axis=1)
test_3 = test.drop(['id', 'hour_bef_windspeed', 'hour_bef_pm2.5'], axis=1)

# 각 X_train에 대해 모델 훈련

model_input_var1 = RandomForestRegressor(criterion = 'mse')
model_input_var1.fit(X_train_1, Y_train)

model_input_var2 = RandomForestRegressor(criterion = 'mse')
model_input_var2.fit(X_train_2, Y_train)

model_input_var3 = RandomForestRegressor(criterion = 'mse')
model_input_var3.fit(X_train_3, Y_train)

# 각 모델로 test 셋들을 예측

y_pred_1 = model_input_var1.predict(test_1)
y_pred_2 = model_input_var2.predict(test_2)
y_pred_3 = model_input_var3.predict(test_3)

# 각 결과들을 submission 파일로 저장

submission_1 = pd.read_csv('따릉이/submission.csv')
submission_2 = pd.read_csv('따릉이/submission.csv')
submission_3 = pd.read_csv('따릉이/submission.csv')

submission_1['count'] = y_pred_1
submission_2['count'] = y_pred_2
submission_3['count'] = y_pred_3

submission_1.to_csv('sub_1.csv',index=False)
submission_2.to_csv('sub_2.csv',index=False)
submission_3.to_csv('sub_3.csv',index=False)

#저장 후 제출한 결과 sub_1 ->id피쳐를 제거한 결과가 가장 순위가 높았다.
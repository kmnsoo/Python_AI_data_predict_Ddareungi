#pandas와 사이킷런_DecisionTreeRegressor 라이브러리 불러오기
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#import numpy as np

# 정답 코드
# 저장해둔 폴더 경로를 통해 csv파일 데이터를 DataFrame객체로 불러오기
train = pd.read_csv('따릉이/train.csv') 
test = pd.read_csv('따릉이/test.csv')

#print(train.shape)
#print(test.shape)
#print(train.head(10)) 
#print(train.tail()) 


#df = pd.DataFrame({
#        'name': ['kwon', 'park', 'kim'],
#        'age':[30, np.nan, 19],
#        'class':[np.nan, np.nan, 1]
#})
#df.info()
#
#df.isnull().sum()


#print(train.isnull())
#print('\n--------------------------------train.csv 각 열 별 결측치 수--------------------------------\n')
#print(test.isnull().sum())


#train.info()
#test.info()

#결측치 전처리. train의 결측치는 제거하고, teset의 결측치는 모두 0으로 대체
train = train.dropna()
test = test.fillna(0)

print(train.isnull().sum())
print(test.isnull().sum())

#모델 훈련
X_train = train.drop(['count'], axis=1)
Y_train = train['count']

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
#print(model.fit(X_train, Y_train))

#테스트 예측. predict함수를 이용하여 test data를 훈련된 모델로 예측한 
#data array를 생성합니다.
pred = model.predict(test)
#pred[:5]
#print(pred[:5])


#따릉이 폴더 내부에 있는 submmison.csv 파일을 df파일로 불러오기
submission = pd.read_csv('따릉이/submission.csv')
#submission df파일의 count 피쳐에 예측결과 할당하기
submission['count'] = pred
#제출 파일 csv로 생성 후 저장하기.
submission.to_csv('sub.csv',index=False)
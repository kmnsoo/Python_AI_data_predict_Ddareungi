import pandas as pd
from sklearn.tree import DecisionTreeRegressor
#import numpy as np

# 정답 코드
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

train = train.dropna()
test = test.fillna(0)

print(train.isnull().sum())
print(test.isnull().sum())

X_train = train.drop(['count'], axis=1)
Y_train = train['count']

model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
#print(model.fit(X_train, Y_train))

pred = model.predict(test)
#pred[:5]
#print(pred[:5])

submission = pd.read_csv('따릉이/submission.csv')
submission['count'] = pred
submission.to_csv('sub.csv',index=False)
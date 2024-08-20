import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns


# 데이터 불러오기 (인덱스 지정)
header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
          'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv',
                   delim_whitespace=True, names=header)


# 'RM'은 방의 개수, 'MEDV'는 주택가격을 나타냄
array = data.values
X1 = data['RM'].values.reshape(-1, 1)  # X를 2차원 배열로 변환해야 합니다.
Y1 = data['MEDV'].values

# 데이터 분포 시각화 (전체 데이터)
plt.scatter(X1, Y1, color='LightPink', label='Actual Data Points', marker='*', s=30, alpha=0.5)
plt.title("Number of Rooms vs. Housing Price")
plt.xlabel("Number of Rooms")
plt.ylabel("Median Value of Homes ($1000s)")
plt.legend()
plt.show()
#__________________________________________________________________________
# 'LSTAT'은 하위계층의 비율, 'MEDV'는 주택가격을 나타냄
array = data.values
X2 = data['LSTAT'].values.reshape(-1, 1)  # X를 2차원 배열로 변환해야 합니다.
Y2 = data['MEDV'].values

# 데이터 분포 시각화 (전체 데이터)
plt.scatter(X2, Y2, color='skyblue', label='Actual Data Points', marker='*', s=30, alpha=0.5)
plt.title("Lower Status of the Population vs. Housing Price")
plt.xlabel("Lower Status of the Population (per capita)")
plt.ylabel("Median Value of Homes ($1000s)")
plt.legend()
# plt.show()
#__________________________________________________________________________

#데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3)

#선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, Y_train)

#예측
y_pred = model.predict(X_train)

#잔차 플롯(모델의 예측과 실제 값의 차이 시각화)
# plt.figure(figsize=(10, 6))
# residuals = Y_train - y_pred
# plt.scatter(y_pred, residuals, color='blue', marker='o', alpha=0.6)
# plt.hlines(0, min(y_pred), max(y_pred), colors='red', linestyles='--')
# plt.title("Residuals vs. Predicted Values")
# plt.xlabel("Predicted Values")
# plt.ylabel("Residuals")
# plt.grid(True)

#히스토그램
plt.figure(figsize=(10, 6))
plt.hist(Y_train, bins=30, alpha=0.5, label='Actual Values')
plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted Values')
plt.title("Histogram of Actual vs. Predicted Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()
# print(model.coef_, model.intercept_)->가중치 찾기. 방정식을 만들어내는 것. y=ax+b에서 a와 b의 값을 찾아낸 것.
# kfold = kFold(n_splits=5)
#mse = cross_val_score(model, X, Y, scoring='neg_mean_squared')
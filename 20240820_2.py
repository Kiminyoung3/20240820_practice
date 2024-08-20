import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 데이터 불러오기 (첫 번째 컬럼을 인덱스로 사용)
data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)

# 단위 변환 (인치 -> 센티미터, 파운드 -> 킬로그램)
data['Height(cm)'] = data['Height(Inches)'] * 2.54
data['Weight(kg)'] = data['Weight(Pounds)'] * 0.453592

# X는 키, Y는 몸무게
X = data['Height(cm)'].values.reshape(-1, 1)
Y = data['Weight(kg)'].values

# 데이터셋을 훈련 세트와 테스트 세트로 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# 데이터 분포 시각화 (전체 데이터)
plt.scatter(X, Y, color='skyblue', label='Actual Data Points', marker='*', s=30, alpha=0.5)
plt.title("Height vs. Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.show()

# 모델 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# X_test에서 10개의 랜덤 샘플 추출
sample_indices = np.random.choice(X_test.shape[0], 100, replace=False)
sample_X = X_test[sample_indices]
sample_Y = Y_test[sample_indices]

# 샘플에 대한 예측 수행
y_pred = model.predict(sample_X)

# 성능 평가 (샘플링된 데이터에 대해)
#MAE: 모델 예측 값과 실제 값의 차이의 절대값을 평균한 값(값이 작을수록 예측이 실제 값에 가깝다)
mae = mean_absolute_error(sample_Y, y_pred)
#MSE: 예측 오차의 제곱을 평균한 값(값이 작을수록 예측이 실제 값에 가깝다. 값이 크면 몇몇 예측이 크게 벗어난다)
mse = mean_squared_error(sample_Y, y_pred)
#RMSE: MSE의 제곱근. MSE의 단위를 실제 데이터와 동일하게 만들어 해석 쉽도록 함. 큰 오차에 더 민감한 반응
rmse = np.sqrt(mse)
#R2은 결정계수라고도 하며, 모델이 실제 데이터를 얼마나 잘 설명하는 지를 나타냄. 1에 가까울수록 데이터 잘 설명함(음수는 평균보다 못함)
r2 = r2_score(sample_Y, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R^2): {r2:.2f}")

# 결과 시각화 (예측 결과와 실제 값 비교)
plt.figure(figsize=(10, 6))
#plt.scatter(X_tesst[:100], y_pred[:100], color='red') ->이런식으로 100개만 추출해서 그래프그릴수도있음
plt.scatter(range(len(sample_Y)), sample_Y, color='green', label='Actual Values', marker='o')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='*')
plt.title("Comparison of Actual vs. Predicted Weight (300 Random Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Weight (kg)")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()

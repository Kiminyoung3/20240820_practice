import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#index_col=0 은 첫 번째 컬럼을 인덱스로 사용한다는 뜻이다.
data=pd.read_csv('./data/5.HeightWeight.csv', index_col=0)

array = data.values
#print(array)

#인덱스를 추출해서 단위를 변환한 후 다시 대입하는 방법으로 균일한 데이터 변환
#data['Height(Inches)'] = data['Height(Inches)']*2.54
data['Height(Inches)'] = data['Height(Inches)'] * 2.54
data['Weight(Pounds)'] = data['Weight(Pounds)'] * 0.453592
print(data)


#데이터프레임의 인덱스를 추출하기. X는 키, Y는 몸무게
X=array[:, 0]
Y=array[:, 1]

print(X)
print(Y)

plt.scatter(X, Y, color='skyblue', label='Actual Data Points', marker='*', s=30, alpha=0.5)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_train)

model.coef_
model.intercept_

y_pred = model.predict(X_test)
error = mean_absolute_error(y_pred, Y_test)
print(error)

fig, ax=plt.subplots()
plt.clf()
plt.scatter(X, Y, label="Actual Data Points", color="green", marker="x", s=30, alpha=0.5)
plt.title("Actual Data Points")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# 성능 평가
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R^2): {r2:.2f}")

plt.figure(figsize=(10, 6))

plt.scatter(range(len(Y_test)), Y_test, color='green', label='Actual Values', marker='o')

plt.plot(range(len(y_pred)), y_pred, color='red', label='predictted Values', marker='*')

#여기에 초기화 넣으면 안돼~~~그래프 안나온다.
plt.title("Scatter Plot of Salary vs. Experience Years")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()




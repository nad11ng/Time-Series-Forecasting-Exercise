import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("co2.csv")
data ["time"] = pd.to_datetime(data["time"], yearfirst=True)
data["co2"] = data["co2"].interpolate()


# fig, ax = plt.subplots()
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()
# # print(data)
# # print(data.info())
window_size=5

def create_recursive_data(inp_data, input_window_size):
    i = 1
    while i < input_window_size:
        inp_data["co2_{}".format(i)] = inp_data["co2"].shift(-i)
        i += 1
    inp_data["target"] = inp_data["co2"].shift(-i)
    inp_data = data.dropna(axis=0)
    return inp_data

data = create_recursive_data(data, window_size)

# print(data.drop("time", axis=1).corr())

x = data.drop(["time", "target"], axis=1)
y = data["target"]

train_size = 0.8

num_samples = len(x)
x_train = x[:int(num_samples*train_size)]
y_train = y[:int(num_samples*train_size)]
x_test = x[int(num_samples*train_size):]
y_test = y[int(num_samples*train_size):]

model = LinearRegression()
# model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(x_train, y_train)


y_predict = model.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print("Prediction: {}. Actual value: {}".format(i, j))

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2 score: {}".format(r2_score(y_test, y_predict)))

fig, ax = plt.subplots()
ax.plot(data["time"][:int(num_samples*train_size)], y_train, label="train")
ax.plot(data["time"][int(num_samples*train_size):], y_test, label="test")
ax.plot(data["time"][int(num_samples*train_size):], y_predict, label="prediction")
ax.set_xlabel("Time")
ax.set_ylabel("CO2")
ax.legend()
ax.grid()
plt.show()

sample_data = [400.1, 400.3, 401, 400.2, 401.5]
for i in range(10):
    output = model.predict([sample_data]).tolist()
    print(sample_data, output)
    sample_data = sample_data[1:] + output
#Посчитать коэффициент линейной регрессии при заработной плате (zp), 
#используя градиентный спуск (без intercept).

#Решение без intercept:


import numpy as np
import matplotlib.pyplot as plt

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

n= 10

alpha = 1e-6
b1 = 0.1



def mse_(b1, y=ks, X=zp, n=10):
    return np.sum((b1 * X - y) ** 2) / n

for i in range(1000):
    fp = (1 / n) * np.sum(2 * (b1 * zp - ks) * zp)
    b1 -= alpha * fp
    if i % 100 == 0:
        print(f'Итерация: {i}, b1 : {b1}, mse: {mse_(b1) }')

y_pred = b1 * zp

print('_')
print(f"Без intercept: {y_pred}")
print('_')
def main(zp, ks):
    plt.scatter(zp, ks)
    plt.plot(zp, y_pred, 'r:', label = 'без интерсепта')
    plt.legend()
    plt.show()
main(zp, ks)    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Загружаем датасет
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)

feature = 's5'

# Проверяем минимальное значение
min_val = X[feature].min()
print(f"Минимальное значение {feature}: {min_val:.3f}")

# Чтобы применить np.log(), все значения должны быть > 0,
# поэтмоу смещаем весь столбец на abs(min_val)+1
# и теперь минимальное значение станет 1, а остальные > 1
X[feature] = X[feature] + abs(min_val) + 1

new_min = X[feature].min()
print(f"Минимальное значение после смещения {feature}: {new_min:.3f}")

# Применяем логарифмирование
X[f'{feature}_log'] = np.log(X[feature])

# Распределения до и после
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(X[feature], bins=30, color='skyblue', edgecolor='black')
plt.title(f'Распределение {feature} (до логарифмирования)')

plt.subplot(1,2,2)
plt.hist(X[f'{feature}_log'], bins=30, color='salmon', edgecolor='black')
plt.title(f'Распределение {feature} (после логарифмирования)')

plt.tight_layout()
plt.show()

# Метрики
print(f"Среднее {feature} до логарифмирования: {X[feature].mean():.3f}")
print(f"Среднее {feature} после логарифмирования: {X[f'{feature}_log'].mean():.3f}")

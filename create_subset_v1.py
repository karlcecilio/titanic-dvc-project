import pandas as pd
import os

# Чтение исходных данных
df = pd.read_csv('data/titanic_raw.csv')

# Создание подмножества: Pclass, Sex, Age
subset = df[['pclass', 'sex', 'age']].copy()

# Сохранение версии 1 (исходные данные, включая пропущенные значения)
os.makedirs('data/versions', exist_ok=True)
subset.to_csv('data/versions/titanic_subset_v1.csv', index=False)

print(f"Версия 1 сохранена, размер: {subset.shape}")
print(f"Количество пропущенных значений в Age: {subset['age'].isnull().sum()}")
print("\nПредпросмотр данных версии 1:")
print(subset.head())

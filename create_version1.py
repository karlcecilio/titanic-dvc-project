import pandas as pd
import os

# Создание директории для версий
os.makedirs('data/versions', exist_ok=True)

# Чтение исходных данных
df = pd.read_csv('data/titanic_raw.csv')

# Создание поднабора данных: Pclass, Sex, Age
subset_v1 = df[['Pclass', 'Sex', 'Age']].copy()

# Сохранение версии 1
subset_v1.to_csv('data/versions/titanic_subset_v1.csv', index=False)

print("Версия 1: Исходный поднабор данных (с пропусками)")
print("=" * 40)
print(f"Размер данных: {subset_v1.shape}")
print(f"Пропуски в Age: {subset_v1['Age'].isnull().sum()}")
print(f"Распределение Sex: {subset_v1['Sex'].value_counts().to_dict()}")
print(f"Распределение Pclass: {subset_v1['Pclass'].value_counts().sort_index().to_dict()}")
print("\nПредпросмотр данных:")
print(subset_v1.head(10))

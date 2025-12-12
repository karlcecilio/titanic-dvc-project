import pandas as pd

# Чтение версии 1
df_v1 = pd.read_csv('data/versions/titanic_subset_v1.csv')

# Вычисление среднего значения Age (игнорируя NaN)
age_mean = df_v1['Age'].mean()

print("Версия 2: Заполнение пропущенных значений в Age средним значением")
print("=" * 40)
print(f"Среднее значение Age: {age_mean:.2f}")
print(f"Количество пропущенных значений до заполнения: {df_v1['Age'].isnull().sum()}")

# Создание версии 2 (заполнение пропущенных значений)
df_v2 = df_v1.copy()
df_v2['Age'] = df_v1['Age'].fillna(age_mean)

# Сохранение версии 2
df_v2.to_csv('data/versions/titanic_subset_v2.csv', index=False)

print(f"Количество пропущенных значений после заполнения: {df_v2['Age'].isnull().sum()}")
print("\nПредварительный просмотр данных версии 2:")
print(df_v2.head(10))
print("\nСтатистическая информация о Age:")
print(df_v2['Age'].describe())

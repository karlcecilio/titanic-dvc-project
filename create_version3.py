import pandas as pd

# Чтение версии 2 (с заполненными пропусками)
df_v2 = pd.read_csv('data/versions/titanic_subset_v2.csv')

print("Версия 3: One-Hot Encoding для столбца Sex")
print("=" * 40)
print("Данные до преобразования:")
print(df_v2.head())
print(f"\nНазвания столбцов до преобразования: {list(df_v2.columns)}")

# Применение One-Hot Encoding
df_v3 = pd.get_dummies(df_v2, columns=['Sex'], prefix='Sex', dtype=int)

print("\nДанные после преобразования:")
print(df_v3.head())
print(f"\nНазвания столбцов после преобразования: {list(df_v3.columns)}")
print(f"Статистика нового столбца Sex_female: {df_v3['Sex_female'].sum()} / {len(df_v3)}")
print(f"Статистика нового столбца Sex_male: {df_v3['Sex_male'].sum()} / {len(df_v3)}")

# Проверка корректности кодирования
print("\nПроверка кодирования:")
for i in range(3):
    original_sex = df_v2.iloc[i]['Sex']
    encoded_female = df_v3.iloc[i]['Sex_female']
    encoded_male = df_v3.iloc[i]['Sex_male']
    print(f"Пассажир {i+1}: Sex='{original_sex}' → female={encoded_female}, male={encoded_male}")

# Сохранение версии 3
df_v3.to_csv('data/versions/titanic_subset_v3.csv', index=False)

print(f"\nВерсия 3 сохранена, размер: {df_v3.shape}")

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

# Способ 1: Получение набора данных "Титаник" через sklearn
def create_titanic_dataset():
    # Загрузка набора данных "Титаник" из OpenML
    titanic = fetch_openml(name='titanic', version=1, as_frame=True)
    df = titanic.frame
    
    # Сохранение исходного набора данных
    df.to_csv('data/titanic_raw.csv', index=False)
    print(f"Исходный набор данных сохранён, размер: {df.shape}")
    print(f"Названия колонок: {list(df.columns)}")
    
    return df

# Способ 2: Создание локальных данных в случае неудачи fetch_openml
def create_sample_dataset():
    # Создание примерных данных (если загрузка из сети не удалась)
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.randint(0, 2, n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 15, n_samples),
        'SibSp': np.random.randint(0, 4, n_samples),
        'Parch': np.random.randint(0, 3, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.uniform(0, 100, n_samples),
        'Cabin': np.random.choice([f'C{i}' for i in range(1, 51)] + [None], n_samples),
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples)
    }
    
    # Добавление пропущенных значений
    df = pd.DataFrame(data)
    age_missing = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    df.loc[age_missing, 'Age'] = np.nan
    
    # Сохранение
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/titanic_raw.csv', index=False)
    print(f"Примерный набор данных сохранён, размер: {df.shape}")
    return df

# Основная программа
if __name__ == "__main__":
    import os
    
    # Создание директории data
    os.makedirs('data', exist_ok=True)
    
    try:
        print("Попытка загрузки набора данных 'Титаник' из OpenML...")
        df = create_titanic_dataset()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("Создание примерного набора данных...")
        df = create_sample_dataset()
    
    # Отображение информации о данных
    print("\nПредпросмотр данных:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nСтатистика пропущенных значений:")
    print(df.isnull().sum())

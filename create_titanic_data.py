import pandas as pd
import numpy as np
import os

def create_titanic_dataset():
    """Создание набора данных Титаника"""
    # Создание примерных данных (имитация набора данных Титаника)
    np.random.seed(42)
    n_passengers = 300
    
    data = {
        'PassengerId': range(1, n_passengers + 1),
        'Survived': np.random.randint(0, 2, n_passengers),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.2, 0.3, 0.5]),
        'Name': [f'Passenger_{i}' for i in range(1, n_passengers + 1)],
        'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.6, 0.4]),
        'Age': np.random.normal(30, 15, n_passengers),
        'SibSp': np.random.randint(0, 4, n_passengers),
        'Parch': np.random.randint(0, 3, n_passengers),
        'Ticket': [f'TICKET{i:04d}' for i in range(1, n_passengers + 1)],
        'Fare': np.random.exponential(50, n_passengers),
        'Cabin': [f'C{np.random.randint(1, 51):03d}' if np.random.random() > 0.7 else np.nan 
                  for _ in range(n_passengers)],
        'Embarked': np.random.choice(['C', 'Q', 'S', None], n_passengers, p=[0.3, 0.2, 0.4, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Добавление пропущенных значений в Age (около 20%)
    age_missing_idx = np.random.choice(n_passengers, size=int(n_passengers * 0.2), replace=False)
    df.loc[age_missing_idx, 'Age'] = np.nan
    
    # Создание директории для данных
    os.makedirs('data', exist_ok=True)
    
    # Сохранение исходных данных
    df.to_csv('data/titanic_raw.csv', index=False)
    
    # Вывод информации о наборе данных
    print("=" * 50)
    print("Набор данных 'Титаник' успешно создан")
    print("=" * 50)
    print(f"Размер набора данных: {df.shape[0]} строк × {df.shape[1]} столбцов")
    print(f"Пропущенные значения в столбце Age: {df['Age'].isnull().sum()} ({df['Age'].isnull().mean()*100:.1f}%)")
    print(f"Распределение в столбце Sex: {df['Sex'].value_counts().to_dict()}")
    print(f"Распределение в столбце Pclass: {df['Pclass'].value_counts().sort_index().to_dict()}")
    print("\nПредварительный просмотр данных:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    create_titanic_dataset()

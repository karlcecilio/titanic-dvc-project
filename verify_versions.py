import pandas as pd
import os
import sys

def load_and_describe_version(filepath, version_name):
    """Загрузка и описание версии данных"""
    print(f"\n{'='*60}")
    print(f"Версия: {version_name}")
    print(f"Файл: {filepath}")
    print('='*60)
    
    if not os.path.exists(filepath):
        print("Файл не существует!")
        return None
    
    df = pd.read_csv(filepath)
    
    print(f"Размер данных: {df.shape}")
    print(f"Названия столбцов: {list(df.columns)}")
    
    if 'Age' in df.columns:
        missing_age = df['Age'].isnull().sum()
        print(f"Пропуски в Age: {missing_age}")
        if missing_age == 0:
            print(f"Статистика Age: среднее={df['Age'].mean():.2f}, ст.откл.={df['Age'].std():.2f}")
    
    # Проверка столбцов с кодировкой
    sex_cols = [c for c in df.columns if c.startswith('Sex_')]
    if sex_cols:
        print(f"Столбцы с кодировкой: {sex_cols}")
        for col in sex_cols:
            print(f"  {col}: {df[col].sum()} пассажиров")
    
    print("\nПервые 5 строк данных:")
    print(df.head())
    
    return df

def compare_versions():
    """Сравнение всех версий"""
    versions = [
        ('data/versions/titanic_subset_v1.csv', 'Версия 1 - Исходные данные (с пропусками)'),
        ('data/versions/titanic_subset_v2.csv', 'Версия 2 - Заполненные пропуски'),
        ('data/versions/titanic_subset_v3.csv', 'Версия 3 - One-Hot Encoding'),
    ]
    
    print("Сравнение версий набора данных 'Титаник'")
    print("="*60)
    
    all_dfs = []
    for filepath, name in versions:
        df = load_and_describe_version(filepath, name)
        if df is not None:
            all_dfs.append((name, df))
    
    # Сравнение различий
    if len(all_dfs) >= 2:
        print("\n" + "="*60)
        print("Сводка различий между версиями")
        print("="*60)
        
        # Сравнение версии 1 и версии 2
        if len(all_dfs) >= 2:
            name1, df1 = all_dfs[0]
            name2, df2 = all_dfs[1]
            
            print(f"\n{name1} vs {name2}:")
            age_diff = df2['Age'].mean() - df1['Age'].mean()
            print(f"  Разница в среднем Age: {age_diff:.4f}")
            print(f"  Заполнено {df1['Age'].isnull().sum()} пропущенных значений")
        
        # Сравнение версии 2 и версии 3
        if len(all_dfs) >= 3:
            name2, df2 = all_dfs[1]
            name3, df3 = all_dfs[2]
            
            print(f"\n{name2} vs {name3}:")
            print(f"  Добавлено столбцов: {len(df3.columns) - len(df2.columns)}")
            new_cols = set(df3.columns) - set(df2.columns)
            print(f"  Новые столбцы: {list(new_cols)}")

if __name__ == "__main__":
    compare_versions()

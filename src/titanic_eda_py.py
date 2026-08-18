#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titanic EDA: разведочный анализ и проверка статистических гипотез.

Скрипт запускается из терминала. Графики сохраняются в папку ./plots/ в виде PNG-файлов.
"""

import os
import matplotlib
matplotlib.use('Agg')  # неинтерактивный backend — без GUI-окон, только сохранение в файл

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy import stats

# Папка для сохранения графиков
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_plot(filename):
    """Сохраняет текущий график в папку plots/ и закрывает фигуру."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[saved] {path}')


# ======================================================================
# План работы:
# 1. Первичный осмотр данных
# 2. Анализ пропусков
# 3. Дубликаты и выбросы (аномалии)
# 4. Univariate-анализ числовых переменных
# 5. Univariate-анализ категориальных переменных
# 6. Bivariate-анализ: признаки vs Survived
# 7. Корреляционный анализ
# 8. Multivariate-паттерны
# 9. Статистическая проверка гипотез
# ======================================================================

df = sns.load_dataset('titanic')

# ======================================================================
# Пункт 1. Первичный осмотр данных
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 1: Первичный осмотр данных')
print('=' * 70)

print(df.shape)
print(df.head())

df.info()
print(df.describe())
print(df.describe(include=['object', 'category']))

# ======================================================================
# Пункт 2. Анализ пропусков
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 2: Анализ пропусков')
print('=' * 70)

miss_count = df.isnull().sum()
miss_perc = (miss_count * 100) / len(df)
miss_df = pd.DataFrame({
    'Количество пропусков': miss_count,
    'Процент пропусков': miss_perc
})
mask = miss_df['Количество пропусков'] > 0
miss_df = miss_df[mask].sort_values(by='Процент пропусков', ascending=False)
print(miss_df)

# Визуализация пропусков
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Карта пропущенных значений')
save_plot('01_missing_heatmap.png')

plt.figure(figsize=(8, 5))
miss_df['Процент пропусков'].plot(kind='barh')
plt.xlabel('% пропусков')
plt.title('Доля пропущенных значений')
save_plot('02_missing_barh.png')

# Гипотеза: пропуски в deck не случайны — связаны с pclass
df['miss_d'] = df['deck'].isnull()
print('\nДоля пропусков deck по классам:')
print(df.groupby('pclass')['miss_d'].mean())

# --- Корректировка (очистка) ---
df['age_m'] = df['age'].fillna(df['age'].median())  # новая колонка, оригинал age сохраняем для пункта 9
df['FamilySize'] = df['sibsp'] + df['parch'] + 1
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

print('\nПропуски после обработки:')
print(df.isnull().sum())

df = df.drop(columns=['alive', 'class', 'embark_town', 'adult_male', 'deck', 'miss_d'])
print('\nИтоговые столбцы:', df.columns.tolist())

# ======================================================================
# Пункт 3: Дубликаты и выбросы (аномалии)
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 3: Дубликаты и выбросы')
print('=' * 70)

print(f"Количество полных дубликатов: {df.duplicated().sum()}")
# Случайное совпадение всех значений у разных пассажиров статистически вероятно
# (мало признаков низкой кардинальности после очистки, уникального ID нет) — не удаляем.

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df['age_m'], ax=axes[0])
axes[0].set_title('Age')
sns.boxplot(y=df['fare'], ax=axes[1])
axes[1].set_title('Fare')
sns.boxplot(y=df['FamilySize'], ax=axes[2])
axes[2].set_title('FamilySize')
plt.tight_layout()
save_plot('03_outliers_boxplots.png')


def count_outliers_iqr(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound


print('\nВыбросы по правилу IQR:')
for col in ['age_m', 'fare', 'sibsp', 'parch', 'FamilySize']:
    count, low, up = count_outliers_iqr(col)
    print(f"{col}: {count} выбросов (границы: {low:.2f} — {up:.2f})")

# Вывод: sibsp/parch/FamilySize сильно дискретны, IQR формально завышает число
# "выбросов" (например, parch: границы 0.00-0.00). Ни один выброс не удаляется —
# они не ошибки данных, а содержательная информация (богатые пассажиры, большие семьи).

# ======================================================================
# Пункт 4: Univariate-анализ числовых переменных
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 4: Univariate — числовые переменные')
print('=' * 70)

numeric_cols = ['age_m', 'fare', 'sibsp', 'parch', 'FamilySize']

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30, color='steelblue')
    axes[i].set_title(f'Распределение: {col}')
fig.delaxes(axes[5])
plt.tight_layout()
save_plot('04_numeric_distributions.png')

print('\nАсимметрия и эксцесс:')
for col in numeric_cols:
    print(f"{col}: skewness = {df[col].skew():.2f}, kurtosis = {df[col].kurtosis():.2f}")
# fare — самый проблемный признак (skew 4.79, kurtosis 33.4 — аномально высокие
# значения, у нормального распределения kurtosis ≈ 0)

# Распределение age с разбивкой по who
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='age', hue='who', multiple='stack', bins=30)
plt.title('Распределение возраста по категории who')
save_plot('05_age_by_who.png')

# ======================================================================
# Пункт 5: Univariate — категориальные переменные
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 5: Univariate — категориальные переменные')
print('=' * 70)

cat_cols = ['sex', 'pclass', 'embarked', 'who', 'alone']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(data=df, x=col, hue=col, legend=False, ax=axes[i], palette='muted')
    axes[i].set_title(f'Распределение: {col}')
fig.delaxes(axes[5])
plt.tight_layout()
save_plot('06_categorical_distributions.png')
# Явный дисбаланс: мужчин почти вдвое больше женщин, большинство село в
# Southampton, преобладает 3 класс, категория who доминируется "man".

# ======================================================================
# Пункт 6: Bivariate — признаки vs survived
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 6: Bivariate — признаки vs survived')
print('=' * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
sns.barplot(data=df, x='sex', y='survived', ax=axes[0])
axes[0].set_title('Выживаемость по полу')
sns.barplot(data=df, x='pclass', y='survived', ax=axes[1])
axes[1].set_title('Выживаемость по классу')
sns.barplot(data=df, x='embarked', y='survived', ax=axes[2])
axes[2].set_title('Выживаемость по порту')
sns.barplot(data=df, x='who', y='survived', ax=axes[3])
axes[3].set_title('Выживаемость по категории who')
sns.barplot(data=df, x='alone', y='survived', ax=axes[4])
axes[4].set_title('Выживаемость: один/не один')
fig.delaxes(axes[5])
plt.tight_layout()
save_plot('07_bivariate_categorical.png')
# Женщины ≈74%, мужчины ≈19%. Классы: 1≈63%, 2≈47%, 3≈24%. who: женщины ≈76%,
# дети ≈59%, мужчины ≈16%. Не одни ≈51%, одиночки ≈30%.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x='survived', y='age_m', ax=axes[0])
axes[0].set_title('Возраст vs Выживаемость')
sns.boxplot(data=df, x='survived', y='fare', ax=axes[1])
axes[1].set_title('Тариф vs Выживаемость')
plt.tight_layout()
save_plot('08_bivariate_numeric.png')
# age почти не различается (медиана ≈28 в обеих группах), fare у выживших заметно выше.

# ======================================================================
# Пункт 7: Корреляционный анализ
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 7: Корреляционный анализ')
print('=' * 70)

plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number, 'bool']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Матрица корреляций')
save_plot('09_correlation_matrix.png')
# survived сильнее всего коррелирует с pclass (-0.34) и fare (+0.26).
# Мультиколлинеарность: sibsp-FamilySize=0.89, parch-FamilySize=0.78,
# alone-FamilySize=-0.69 (FamilySize построен из sibsp и parch).

# ======================================================================
# Пункт 8: Multivariate — комбинации признаков
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 8: Multivariate — класс + пол')
print('=' * 70)

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='pclass', y='survived', hue='sex')
plt.title('Выживаемость по классу и полу одновременно')
save_plot('10_multivariate_class_sex.png')
# Женщины 1-2 класса выживали почти всегда (≈97%, ≈92%), мужчины 3 класса — реже
# всего (≈14%). Пол и класс усиливают друг друга.

# ======================================================================
# Пункт 9: Статистическая проверка гипотез
# ======================================================================
print('\n' + '=' * 70)
print('ПУНКТ 9: Статистическая проверка гипотез')
print('=' * 70)

# --- Гипотеза 1: Выживаемость зависит от пола ---
# H0: пол и выживаемость независимы / H1: пол влияет на выживаемость
print('\n--- Гипотеза 1: пол → survived (хи-квадрат) ---')
contingency_sex = pd.crosstab(df['sex'], df['survived'])
print(contingency_sex)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_sex)
print(f"Chi2 = {chi2:.2f}, p-value = {p_value:.10f}, dof = {dof}")
n = contingency_sex.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_sex.shape) - 1)))
print(f"Cramer's V = {cramers_v:.3f}")
# Результат: p≈0, Cramer's V=0.541 → H0 отвергается, сильная связь.

# --- Гипотеза 2: Выживаемость зависит от класса ---
print('\n--- Гипотеза 2: класс → survived (хи-квадрат) ---')
contingency_class = pd.crosstab(df['pclass'], df['survived'])
print(contingency_class)
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_class)
print(f"Chi2 = {chi2:.2f}, p-value = {p_value:.10f}, dof = {dof}")
n = contingency_class.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_class.shape) - 1)))
print(f"Cramer's V = {cramers_v:.3f}")
# Результат: p≈0, Cramer's V=0.340 → H0 отвергается, умеренная связь.

# --- Гипотеза 3: Возраст выживших и погибших отличается ---
print('\n--- Гипотеза 3: возраст выживших vs погибших ---')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=df, x='age', hue='survived', kde=True, ax=axes[0])
axes[0].set_title('Гист. Выживаемость')
sns.boxplot(data=df, x='survived', y='age', ax=axes[1])
axes[1].set_title('Боксплот Выживаемость')
plt.tight_layout()
save_plot('11_hypothesis_age.png')

survived_age = df[df['survived'] == 1]['age'].dropna()
died_age = df[df['survived'] == 0]['age'].dropna()

print("Shapiro-Wilk для выживших:", stats.shapiro(survived_age))
print("Shapiro-Wilk для погибших:", stats.shapiro(died_age))
# Оба p < 0.05 → распределение не нормальное → используем Mann-Whitney U

stat, p_value = stats.mannwhitneyu(survived_age, died_age, alternative='two-sided')
print(f"Mann-Whitney U: statistic = {stat:.2f}, p-value = {p_value:.4f}")
print(f"Средний возраст выживших: {survived_age.mean():.2f}")
print(f"Средний возраст погибших: {died_age.mean():.2f}")
# Результат: p=0.1605 > 0.05 → H0 НЕ отвергается, значимой разницы в возрасте нет.

# --- Гипотеза 4: Тариф выживших и погибших отличается ---
print('\n--- Гипотеза 4: тариф выживших vs погибших ---')
survived_fare = df[df['survived'] == 1]['fare']
died_fare = df[df['survived'] == 0]['fare']

stat, p_value = stats.mannwhitneyu(survived_fare, died_fare, alternative='two-sided')
print(f"Mann-Whitney U: statistic = {stat:.2f}, p-value = {p_value:.10f}")
print(f"Медианный тариф выживших: {survived_fare.median():.2f}")
print(f"Медианный тариф погибших: {died_fare.median():.2f}")
# Результат: p≈0 → H0 отвергается. Медианы 26.0 vs 10.5 значимо различаются.

# ======================================================================
# ИТОГОВЫЕ ВЫВОДЫ EDA
# ======================================================================
print('\n' + '=' * 70)
print("""
ИТОГОВЫЕ ВЫВОДЫ EDA

Описательная часть:
1. Пол — главный предиктор выживаемости: женщины ~74% против ~19% у мужчин.
2. Класс имеет чёткий градиент: 63% (1) -> 47% (2) -> 24% (3).
3. Комбинация пол+класс усиливает эффект: женщины 1-2 класса выживали
   почти всегда (92-97%), мужчины 3 класса - реже всего (14%).
4. Возраст сам по себе слабо влияет на выживание (корреляция -0.06).
5. Fare коррелирует с выживанием (+0.26), отражая влияние класса.
6. Пропуски в deck не случайны (MAR): связаны с pclass (19% в 1 классе
   против 91-97.5% во 2-3 классах).
7. Выбросы в данных - не ошибки, а содержательная неоднородность
   пассажиров, поэтому не удалялись.
8. Обнаружена мультиколлинеарность FamilySize с sibsp/parch/alone.

Статистическая проверка гипотез:
1. Пол -> survived: chi2=260.72, p≈0, Cramer's V=0.541 - H0 отвергается.
2. Класс -> survived: chi2=102.89, p≈0, Cramer's V=0.340 - H0 отвергается.
3. Возраст выживших vs погибших: Mann-Whitney p=0.1605 - H0 НЕ отвергается.
4. Тариф выживших vs погибших: Mann-Whitney p≈0 - H0 отвергается.
""")
print('=' * 70)
print(f'\nВсе графики сохранены в папку ./{PLOTS_DIR}/')

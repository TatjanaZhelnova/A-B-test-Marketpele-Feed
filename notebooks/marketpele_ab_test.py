# %% [markdown]
# # A/B-тестирование расположения рекламных блоков (Marketpele Feed)
#
# Полный цикл продуктового эксперимента:
# 1. Предобработка исторических данных и фильтрация выбросов
# 2. Исследовательский анализ данных (EDA)
# 3. Дизайн A/B-теста и ускорение через CUPED
# 4. Статистический анализ результатов эксперимента (критерий Уэлча)
# 5. Выводы и рекомендации для бизнеса

# %%
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройки отображения графиков и таблиц
warnings.filterwarnings("ignore")
sns.set_palette("colorblind")
plt.rcParams["figure.facecolor"] = "white"
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


def load_csv(filename: str) -> pd.DataFrame:
    """
    Ищет CSV-файл в папках ../data/, data/ или текущей директории
    и загружает его в pandas.DataFrame.
    """
    candidate_paths = [
        Path("../data") / filename,
        Path("data") / filename,
        Path(filename),
    ]
    for path in candidate_paths:
        if path.exists():
            print(f"Загружен файл: {path.resolve()}")
            return pd.read_csv(path, sep=",")

    raise FileNotFoundError(
        f"Файл '{filename}' не найден. Поместите его в папку 'data/' или рядом со скриптом."
    )


# %% [markdown]
# ## 1. Предобработка исторических данных

# %%
# Загрузка исторического датасета
data = load_csv("user_activity_data.csv")
data.columns = data.columns.str.strip().str.lower()

# Приводим дату к формату datetime
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")

print("\nИнформация о датасете:")
data.info()

print("\nПропуски по колонкам:")
print(data.isnull().sum())

print("\nКоличество уникальных значений по колонкам:")
print(data.nunique())

print(f"\nКоличество полных дубликатов: {data.duplicated().sum()}")

print("\nОписательные статистики (исходные данные):")
print(data.drop(columns=["date", "user_id"]).describe().round(2))

# %%
# Сравнение трёх методов фильтрации выбросов по ключевым числовым метрикам
numeric_cols = ["pages_viewed", "paid_clicks", "organic_clicks", "revenue"]

q1 = data[numeric_cols].quantile(0.25)
q3 = data[numeric_cols].quantile(0.75)
iqr = q3 - q1
p99 = data[numeric_cols].quantile(0.99)

clean_iqr15 = data[(data[numeric_cols] <= q3 + 1.5 * iqr).all(axis=1)].copy()
clean_iqr3 = data[(data[numeric_cols] <= q3 + 3.0 * iqr).all(axis=1)].copy()
clean_p99 = data[(data[numeric_cols] <= p99).all(axis=1)].copy()

outlier_methods = pd.DataFrame(
    [
        {
            "Метод": "1.5 * IQR (классический Тьюки)",
            "Удалено строк": len(data) - len(clean_iqr15),
            "Доля удалённых (%)": round((1 - len(clean_iqr15) / len(data)) * 100, 2),
            "Новый std(revenue)": round(clean_iqr15["revenue"].std(), 2),
        },
        {
            "Метод": "3.0 * IQR (дальние выбросы Тьюки)",
            "Удалено строк": len(data) - len(clean_iqr3),
            "Доля удалённых (%)": round((1 - len(clean_iqr3) / len(data)) * 100, 2),
            "Новый std(revenue)": round(clean_iqr3["revenue"].std(), 2),
        },
        {
            "Метод": "99-й перцентиль",
            "Удалено строк": len(data) - len(clean_p99),
            "Доля удалённых (%)": round((1 - len(clean_p99) / len(data)) * 100, 2),
            "Новый std(revenue)": round(clean_p99["revenue"].std(), 2),
        },
    ]
)

print("\nСравнение методов фильтрации выбросов:")
print(outlier_methods.to_string(index=False))

# %% [markdown]
# Выбираем критерий 3.0 * IQR: стабилизирует std(revenue) с 31.81 до 4.11 при сохранении 95.35% данных
data_clean = clean_iqr3.copy()

# %%
# Сравнение метрик до и после удаления выбросов
comparison = pd.DataFrame(
    {
        "Mean (До)": data[numeric_cols].mean(),
        "Mean (После)": data_clean[numeric_cols].mean(),
        "Δ Mean (%)": (data_clean[numeric_cols].mean() / data[numeric_cols].mean() - 1) * 100,
        "Median (До)": data[numeric_cols].median(),
        "Median (После)": data_clean[numeric_cols].median(),
        "Std (До)": data[numeric_cols].std(),
        "Std (После)": data_clean[numeric_cols].std(),
        "Δ Std (%)": (data_clean[numeric_cols].std() / data[numeric_cols].std() - 1) * 100,
        "Max (После)": data_clean[numeric_cols].max(),
    }
).round(2)

print("\nИзменение статистик после очистки данных:")
print(comparison)

# %% [markdown]
# # Вывод по п.1

#  - В ходе очистки было удалено 4 152 наблюдения (4,65% исходного датасета), что соответствует стандарту предобработки (потеря до 5% данных) 
# и не искажает портрет основной аудитории.  

#   - Сильнее всего очистка повлияла на целевую метрику revenue — новый максимум 24.83, снизило стандартное отклонение на 87,1% (31.81 → 4.11, т. е. в 7,7 раза).
#  - Сохранение центральной тенденции: медианы (50%) и квартили (25%, 75%) по просмотрам и кликам остались абсолютно идентичными (14, 3 и 4),
# а медиана выручки изменилась менее чем на 1% (11.37 → 11.26).  

#  - Это доказывает, что мы точечно убрали шум и аномалии для повышения чувствительности будущего t-теста, полностью сохранив типичное поведение пользователей.

# %% [markdown]
# ## 2. Исследовательский анализ данных (EDA)

# %%
# Проверка количества пользователей и периода наблюдений
period = (
    data_clean['date'].max()
    - data_clean['date'].min()
).days + 1
print(f"Период наблюдений в очищенных данных: {period} дней с {data_clean['date'].min().date()} по {data_clean['date'].max().date()}")
print(f"\nУникальных пользователей в исходных данных: {data['user_id'].nunique()}")
print(f"Уникальных пользователей после очистки: {data_clean['user_id'].nunique()}")


# %%
# Общее распределение ключевых метрик (Violin Plot)
metrics_meta = {
    "pages_viewed": ("Распределение просмотров страниц", "Просмотры страниц"),
    "paid_clicks": ("Распределение платных кликов", "Платные клики"),
    "revenue": ("Распределение выручки", "Выручка"),
    "organic_clicks": ("Распределение органических кликов", "Органические клики"),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Распределение ключевых метрик (очищенные данные)", fontsize=15, fontweight="bold", y=1.02)

for ax, (col, (title, ylabel)) in zip(axes.flatten(), metrics_meta.items()):
    sns.violinplot(data=data_clean, y=col, cut=0, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %%
# Распределение метрик в разрезе типов устройств (mobile vs desktop)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Распределение ключевых метрик по типам устройств", fontsize=15, fontweight="bold", y=1.02)

for ax, (col, (title, ylabel)) in zip(axes.flatten(), metrics_meta.items()):
    sns.violinplot(
        data=data_clean,
        x="device",
        y=col,
        hue="device",
        cut=0,
        legend=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Тип устройства")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Дополняем график точной таблицей статистик по базовым метрикам
display(
    data_clean
    .groupby('device')[list(metrics_meta.keys())]
    .agg(['mean', 'median', 'std'])
    .round(2)
)

# %% [markdown]
# # Вывод по типам устройств
# Пользователи мобильных устройств демонстрируют более высокую активность по всем анализируемым метрикам по сравнению с пользователями desktop.
# 
# Наибольшее различие наблюдается по выручке: среднее значение на мобильных устройствах выше примерно в 4 раза. 
# Также мобильные пользователи чаще просматривают страницы и взаимодействуют как с платной, так и с органической рекламой.


# %%
# Динамика ежедневной активной аудитории (DAU) по типам устройств
dau_device = data_clean.groupby(["date", "device"])["user_id"].nunique().unstack()

ax = dau_device.plot(figsize=(14, 5), linewidth=2, marker="o", markersize=4)
ax.set_title("Динамика дневной аудитории (DAU) по типам устройств", fontsize=13)
ax.set_xlabel("Дата")
ax.set_ylabel("Уникальные пользователи")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(title="Устройство")
plt.tight_layout()
plt.show()

# Профиль активности на уровне пользователя
user_stats = data_clean.groupby("user_id").agg(
    active_days=("date", "nunique"),
    total_records=("date", "count"),
    unique_devices=("device", "nunique"),
    total_revenue=("revenue", "sum"),
)

print(f"\nДоля пользователей с 2+ типами устройств: {(user_stats['unique_devices'] > 1).mean():.1%}")
print(user_stats.describe().round(2))

plt.figure(figsize=(12, 4))
sns.countplot(data=user_stats, x="active_days", color="#4C72B0")
plt.title("Распределение пользователей по количеству активных дней", fontsize=13, pad=10)
plt.xlabel("Количество активных дней за период")
plt.ylabel("Число пользователей")
plt.xticks(rotation=90, fontsize=8)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Общий вывод по анализу аудитории

# Динамика DAU. На протяжении рассматриваемого периода дневная аудитория остаётся относительно стабильной. Количество мобильных пользователей примерно в 3,7 раза превышает количество desktop-пользователей.
#
# Активность пользователей. Среднее количество активных дней составляет 28,4, медиана — 28 дней. При этом наблюдается значительный разброс активности: половина пользователей активна от 16 до 41 дня.
#
# Типы устройств. Пересечений между мобильной и desktop-аудиторией не обнаружено: каждый пользователь в очищенном датасете использует только один тип устройства.
#
# Выручка. Наблюдается существенная вариативность выручки между пользователями, что необходимо учитывать при дальнейшем анализе продуктовых метрик.
#
# **Итог:** аудитория преимущественно мобильная, характеризуется относительно стабильной дневной активностью и неоднородностью по количеству активных дней и объёму выручки.

# %%
# Добавление метрик конверсии (CTR платных и органических объявлений)
data_clean = data_clean.assign(
    ctr_paid=lambda x: np.where(x["pages_viewed"] > 0, x["paid_clicks"] / x["pages_viewed"], 0),
    ctr_organic=lambda x: np.where(x["pages_viewed"] > 0, x["organic_clicks"] / x["pages_viewed"], 0),
)

print("\nСводная статистика CTR по типам устройств:")
print(
    data_clean.groupby("device")[["ctr_paid", "ctr_organic"]]
    .agg(["mean", "median", "min", "max", "std"])
    .round(2)
)

# Матрица корреляций ключевых метрик
corr_cols = ["pages_viewed", "revenue", "ctr_paid", "ctr_organic"]
plt.figure(figsize=(7, 5))
sns.heatmap(
    data_clean[corr_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
)
plt.title("Матрица корреляций метрик", fontsize=13, pad=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Вывод по корреляционной матрице

# Наиболее выраженная взаимосвязь наблюдается между количеством просмотров страниц и выручкой (r = 0,54): более активные пользователи, как правило, приносят больше выручки.
# 
# Между количеством просмотров страниц и показателями CTR платной и органической рекламы наблюдается отрицательная корреляция (r = −0,30 и r = −0,35 соответственно).
#
# Связь между CTR платной и органической рекламы слабая положительная (r = 0,28).
#
# При этом взаимосвязь между выручкой и показателями CTR практически отсутствует (r = −0,08 и r = −0,12).
#
# **Итог:** количество просмотров страниц связано с выручкой заметнее, чем показатели кликабельности рекламы. Это подчёркивает важность оценки не только рекламного взаимодействия, но и конечного бизнес-результата.

# %% [markdown]
# ## 3. Дизайн A/B-теста и ускорение через CUPED

# %% [markdown]
# # Бизнес-цель и риски:
#  **Главная цель эксперимента** — увеличить выручку на пользователя без ущерба для долгосрочного удержания. Поскольку по данным EDA глубина просмотра страниц (pages_viewed) сильнее всего коррелирует с выручкой (r = 0.54), агрессивное продвижение рекламы не должно ухудшать пользовательский опыт, вызывать отток с сайта или критически каннибализировать интерес к основному органическому контенту.

# %% [markdown]
# # Формулировка гипотез
# **Гипотеза**: Перемещение платных рекламных блоков в область первого экрана повысит их заметность для пользователей. Это может увеличить взаимодействие с платной рекламой и, как следствие, среднюю выручку на пользователя за 28 дней.
#
# **Ожидаемый эффект:** рост средней выручки на пользователя на 3,5% относительно контрольной группы.
# 
# **Статистические гипотезы:**
#
# **Нулевая гипотеза:** Средняя выручка на пользователя в тестовой группе не отличается от средней выручки в контрольной группе.
#
# **$H_0: \mu_B = \mu_A$** 
#
# **Альтернативная гипотеза:** Средняя выручка на одного пользователя в тестовой группе статистически значимо отличается от средней выручки в контрольной группе.
#
#  **$H_1: \mu_B \neq \mu_A$**


# %% [markdown]
# # Выбор метрик  
#
# **Целевая метрика:** Средняя выручка на пользователя **revenue** (главная бизнес-цель для увеличения прибыли, ради чего всё и затеяно)  
#
# **Прокси метрика:** **ctr_paid** (кликабельность рекламы при новом расположении блоков должна увеличиться)  
# 
# **Барьерные метрики:** **ctr_organic** и **pages_viewed** (слишком агрессивная реклама не должна обрушить органические клики или заставлять пользователей закрывать страницы) 

# %% [markdown]
# # Выбор уровня значимости и мощности теста

# По стандартам задаем уровень значимости alpha = 5% и мощность теста power = 80%


# %%
# Таблица возможных дизайнов теста для нескольких вариантов длительности (2, 3и 4 недели)
def aggregate_user_metrics(df_part: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует сырые счётчики на уровень user_id и вычисляет пользовательские CTR."""
    users = df_part.groupby("user_id", as_index=False).agg(
        {
            "revenue": "sum",
            "paid_clicks": "sum",
            "organic_clicks": "sum",
            "pages_viewed": "sum",
        }
    )
    users["ctr_paid"] = np.where(
        users["pages_viewed"] > 0, users["paid_clicks"] / users["pages_viewed"], 0
    )
    users["ctr_organic"] = np.where(
        users["pages_viewed"] > 0, users["organic_clicks"] / users["pages_viewed"], 0
    )
    return users


# Расчёт базового теоретического MDE (без CUPED) для длительностей 14, 21 и 28 дней
ALPHA = 0.05
POWER = 0.80
MULTIPLIER = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
METRICS = ["revenue", "ctr_paid", "ctr_organic", "pages_viewed"]

max_date = data_clean["date"].max()
rows_mde = []

for days in [14, 21, 28]:
    df_slice = data_clean[data_clean["date"] > (max_date - pd.Timedelta(days=days))]
    users = aggregate_user_metrics(df_slice)
    n_group = len(users) // 2  # Сплит 50/50

    for m in METRICS:
        mean_val = users[m].mean()
        std_val = users[m].std(ddof=1)
        mde_abs = MULTIPLIER * std_val * np.sqrt(2 / n_group)
        mde_pct = (mde_abs / mean_val) * 100

        rows_mde.append(
            {
                "Длительность": f"{days} дней ({days // 7} нед.)",
                "Метрика": m,
                "Юзеров в группе": n_group,
                "Среднее": round(mean_val, 2),
                "Std": round(std_val, 2),
                "MDE (абс.)": round(mde_abs, 3),
                "MDE (%)": round(mde_pct, 2),
            }
        )

mde_baseline_table = pd.DataFrame(rows_mde)
print("\nБазовый расчёт MDE (без CUPED):")
print(mde_baseline_table.to_string(index=False))

# %% [markdown]
# # Вывод
# **Слабый прирост новых пользователей:** при увеличении длительности с 2 до 4 недель размер группы почти не растет (1460 → 1495 юзеров), так как на сайте сформировано постоянное ядро аудитории.  
#
# **Высокий базовый MDE:** без методов снижения дисперсии минимально уловимый эффект по целевой метрике revenue составляет около 7% на любом окне (7.00% → 6.98%), а для CTR улучшается умеренно (7.51% → 6.73% для ctr_paid).  
#
# **Промежуточный итог:** классический дизайн не позволяет уловить заложенный в гипотезе эффект +3.5%, что подтверждает необходимость проверки мощности на симуляциях и подключения **CUPED**.  

# %%
# Симуляции A/A-теста (контроль FPR) и A/B-теста (базовая мощность без CUPED для эффекта +3.5%)
EFFECT_REL = 0.035
N_ITER = 1000
rng = np.random.default_rng(42)

rows_sim = []
aa_pvals_28d_rev = []

for days in [14, 21, 28]:
    df_slice = data_clean[data_clean["date"] > (max_date - pd.Timedelta(days=days))]
    users = aggregate_user_metrics(df_slice)
    n_total = len(users)
    n_group = n_total // 2

    for m in METRICS:
        values = users[m].to_numpy()
        lift = np.mean(values) * EFFECT_REL
        fp_count = 0
        power_count = 0

        for _ in range(N_ITER):
            shuffled = rng.permutation(values)
            group_a = shuffled[:n_group]
            group_b = shuffled[n_group : 2 * n_group]

            # A/A-тест (без эффекта)
            p_aa = stats.ttest_ind(group_a, group_b, equal_var=False).pvalue
            if days == 28 and m == "revenue":
                aa_pvals_28d_rev.append(p_aa)
            if p_aa < ALPHA:
                fp_count += 1

            # A/B-тест (+3.5% в тестовой группе)
            if stats.ttest_ind(group_a, group_b + lift, equal_var=False).pvalue < ALPHA:
                power_count += 1

        fpr = fp_count / N_ITER
        rows_sim.append(
            {
                "Длительность": f"{days} дней",
                "Метрика": m,
                "FPR (A/A)": round(fpr, 3),
                "Контроль α=0.05": "Да" if 0.035 <= fpr <= 0.065 else "Проверить",
                "Мощность без CUPED (%)": round(power_count / N_ITER * 100, 1),
            }
        )

sim_table = pd.DataFrame(rows_sim)
print("\nРезультаты симуляций A/A и A/B без CUPED:")
print(sim_table.to_string(index=False))

# График равномерности распределения p-value на A/A-тесте (revenue, 28 дней)
sorted_p = np.sort(aa_pvals_28d_rev)
cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)

plt.figure(figsize=(5.5, 5.5))
plt.plot(sorted_p, cdf, "b-", linewidth=2, label="Наблюдаемое CDF")
plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="Идеальное равномерное")
plt.axvline(x=ALPHA, color="gray", linestyle=":", linewidth=1.5, label=f"α = {ALPHA}")
plt.title("CDF p-value в A/A-тесте (revenue, 28 дней)", fontsize=12, fontweight="bold")
plt.xlabel("p-value")
plt.ylabel("F(p-value)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Вывод по результатам симуляций A/A и A/B без CUPED
# **Дизайн статистически корректен (результаты А/А-теста):** для всех трех вариантов длительности (14, 21 и 28 дней) и всех четырех метрик доля ложноположительных срабатываний (FPR) находится в допустимом доверительном интервале вокруг выбранного уровня значимости α = 0.05 (0.035–0.065). Сплитование и двусторонний t-тест Уэлча работают без смещений.  
#
# **Недостаточная мощность классического теста (результаты А/Б-теста):** при заложенном в гипотезу эффекте +3.5% мощность обычного t-теста составляет всего 25–37% вместо требуемых 80%. Из-за высокой дисперсии метрик тест в двух случаях из трех пропустит реальный эффект (высокая ошибка II рода).  
#
# Чтобы радикально снизить дисперсию и максимизировать мощность без увеличения выборки, необходимо применить метод **CUPED**.


# %%
def evaluate_cuped_design(
    df_pre: pd.DataFrame, df_test: pd.DataFrame, n_iter: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Оценивает снижение дисперсии, MDE, FPR и мощность теста при применении CUPED."""
    pre_users = aggregate_user_metrics(df_pre)
    test_users = aggregate_user_metrics(df_test)
    merged = pd.merge(test_users, pre_users, on="user_id", how="left", suffixes=("", "_pre"))

    n_total = len(merged)
    n_group = n_total // 2
    local_rng = np.random.default_rng(seed)
    rows = []

    for m in METRICS:
        y = merged[m].to_numpy()
        x = merged[f"{m}_pre"].fillna(merged[f"{m}_pre"].mean()).to_numpy()

        var_x = np.var(x, ddof=1)
        cov_yx = np.cov(y, x, ddof=1)[0, 1]
        theta = cov_yx / var_x if var_x > 0 else 0.0
        y_cuped = y - theta * (x - np.mean(x))

        mean_y = np.mean(y)
        std_raw = np.std(y, ddof=1)
        std_cuped = np.std(y_cuped, ddof=1)
        mde_raw_pct = (MULTIPLIER * std_raw * np.sqrt(2 / n_group) / mean_y) * 100
        mde_cuped_pct = (MULTIPLIER * std_cuped * np.sqrt(2 / n_group) / mean_y) * 100
        corr_xy = np.corrcoef(x, y)[0, 1] if var_x > 0 else 0.0

        lift = mean_y * EFFECT_REL
        fp_cuped = 0
        wins_raw = 0
        wins_cuped = 0

        for _ in range(n_iter):
            idx = local_rng.permutation(n_total)
            a_idx, b_idx = idx[:n_group], idx[n_group : 2 * n_group]

            if stats.ttest_ind(y_cuped[a_idx], y_cuped[b_idx], equal_var=False).pvalue < ALPHA:
                fp_cuped += 1
            if stats.ttest_ind(y[a_idx], y[b_idx] + lift, equal_var=False).pvalue < ALPHA:
                wins_raw += 1
            if stats.ttest_ind(y_cuped[a_idx], y_cuped[b_idx] + lift, equal_var=False).pvalue < ALPHA:
                wins_cuped += 1

        rows.append(
            {
                "Метрика": m,
                "Юзеров в группе": n_group,
                "Corr(X, Y)": round(corr_xy, 3),
                "Std до CUPED": round(std_raw, 3),
                "Std с CUPED": round(std_cuped, 3),
                "MDE до (%)": round(mde_raw_pct, 2),
                "MDE с CUPED (%)": round(mde_cuped_pct, 2),
                "FPR CUPED": round(fp_cuped / n_iter, 3),
                "Мощность до (%)": round(wins_raw / n_iter * 100, 1),
                "Мощность с CUPED (%)": round(wins_cuped / n_iter * 100, 1),
            }
        )

    return pd.DataFrame(rows)


# Сравнение CUPED по длительностям теста (единый предпериод 21 день для сопоставимости)
cuped_by_duration = []
for test_days in [14, 21, 28]:
    split_date = max_date - pd.Timedelta(days=test_days)
    pre_start = split_date - pd.Timedelta(days=21)
    res = evaluate_cuped_design(
        data_clean[(data_clean["date"] > pre_start) & (data_clean["date"] <= split_date)],
        data_clean[data_clean["date"] > split_date],
    )
    res.insert(0, "Окно теста (дней)", test_days)
    cuped_by_duration.append(res)

print("\nСравнение CUPED для разных длительностей теста (предпериод 21 день):")
print(pd.concat(cuped_by_duration, ignore_index=True).to_string(index=False))

# Максимальная конфигурация: 28 дней теста + 27 дней предпериода CUPED
split_28 = max_date - pd.Timedelta(days=28)
cuped_final_28d = evaluate_cuped_design(
    data_clean[data_clean["date"] <= split_28],
    data_clean[data_clean["date"] > split_28],
)
print("\nМаксимальная конфигурация: 28 дней теста + 27 дней предпериода CUPED:")
print(cuped_final_28d.to_string(index=False))


# %% [markdown]
# # Выводы по максимизации мощности с помощью CUPED
# **Кратное снижение дисперсии и рост мощности:** Использование предэкспериментального периода (27 дней) в методе CUPED сжало стандартное отклонение целевой метрики revenue почти в 2,5 раза (98.36 → 39.27) благодаря сильной исторической корреляции трат пользователей. В результате на максимальном окне в 28 дней статистическая мощность по выручке выросла с 28,6% до 97,2%, а по барьерной метрике pages_viewed — с 30,6% до 93,3%, уверенно превысив целевой порог 80%.  
#
# **Причины недостижения 80% по метрикам CTR:** Для прокси-метрики ctr_paid (65,8%) и барьерной ctr_organic (69,1%) мощность выросла более чем вдвое, но осталась ниже желаемых 80%. Это связано с природой метрик-отношений (шум числителя и знаменателя) и более спонтанным характером кликов: их корреляция между прошлым и тестовым периодом заметно ниже, а минимально уловимый эффект составляет около 4,0–4,17%, что выше заложенного в гипотезу порога +3,5%.  
#
# **Готовность дизайна к запуску:**Полученная конфигурация (28 дней теста + 27 дней предпериода CUPED, сплит 50/50) выжимает максимум чувствительности из доступной аудитории (~1495 пользователей в группе). Недобор мощности по вспомогательным CTR не блокирует проведение теста, поскольку финальное бизнес-решение о раскатке принимается по целевой метрике revenue и барьерной pages_viewed, где обеспечена мощность свыше 93.3–97.2%.  

# %% [markdown]
# # Итоговый дизайн:
# 1. **Длительность:** 28 дней (ровно 4 полные недели для компенсации внутринедельной сезонности и максимизации статистической мощности на доступной выборке).
#
# 2. **Доля трафика:** 100% активной аудитории в сплите 50/50 между контрольной и тестовой группами (~1495 уникальных пользователей в каждой группе). Из-за сильных различий в дисперсии конверсий между desktop и mobile при сплитовании рекомендуется контролировать баланс типов устройств (стратификация по device).  
#
# 3. **Метрики** (агрегация на user_id за 28 дней): Целевая — revenue (суммарная выручка на пользователя, критерий принятия решения); Прокси — ctr_paid (контроль механизма роста); Барьерные — ctr_organic ( защита от каннибализации органики) и pages_viewed (сумма просмотров, защита от падения вовлеченности).  
#
# 4. **MDE (при α = 0.05 и мощности 1 - β = 80%):** для revenue — 2.54% (без CUPED было 6.98%); для pages_viewed — 2,80%; для ctr_organic — 4.00%; для ctr_paid — 4.17%. При заложенном в гипотезу эффекте +3.5% фактическая мощность по целевой выручке достигает 97.2% (по CTR мощность ниже 80% — около 65–70% из-за высокой ситуативной дисперсии кликов, что допустимо для вспомогательных метрик).  
#
# 5. **Наличие CUPED:** Да, используется обязательно. В качестве ковариаты берутся аналогичные метрики пользователей за предэкспериментальный период 27 дней до старта теста (пропуски у новых пользователей заполняются средним по предпериоду). За счет высокой автокорреляции выручки (r = 0.917) CUPED сжимает стандартное отклонение с 98.36 до 35.82 и снижает MDE на 60.1%. Статистический критерий — двусторонний t-тест Уэлча (корректность подтверждена на A/A-тестах: FPR = 0.047 для revenue).  
#
# 6. **Способ обработки выбросов:** предварительная фильтрация аномалий на этапе подготовки данных (переход к data_clean с помощью критерия дальних выбросов Тьюки (3.0 * IQR)) в сочетании с агрегацией пользовательских CTR через отношение сумм за весь период теста, что нивелирует разовые всплески до 11–12 кликов за сессию на десктопах.  
#
# 7. **Гипотеза:** изменение расположения элементов на странице в пользу заметности платных блоков (расположение рекламных блоков выше бесплатных элементов) за 28 дней увеличит среднюю выручку на пользователя (revenue) минимум на 3.5% без ухудшения глубины просмотра.    
#
#    - **Нулевая гипотеза Н0:** Через 28 дней теста средняя выручка на пользователя в тестовой группе не изменится относительно контрольной  
#    - **Альтернативная гипотеза Н1:** Через 28 дней теста средняя выручка на пользователя в тестовой группе статистически значимо изменится относительно контрольной (при α = 0.05).  
#
# 8. **Решение:** раскатываем изменение на 100% пользователей, если по CUPED-преобразованной метрике revenue получен статистически значимый рост (p-value < 0.05), а по барьерной метрике pages_viewed нет статистически значимого падения.  


# %% [markdown]
# ## 4. Анализ результатов эксперимента

# %%
# Загрузка и проверка корректности данных эксперимента
exp_data = load_csv("experiment_data.csv")
exp_data.columns = exp_data.columns.str.strip().str.lower()

# Вычисляем CTR, если колонки отсутствуют
if "ctr_paid" not in exp_data.columns:
    exp_data["ctr_paid"] = np.where(
        exp_data["pages_viewed"] > 0, exp_data["paid_clicks"] / exp_data["pages_viewed"], 0
    )
if "ctr_organic" not in exp_data.columns:
    exp_data["ctr_organic"] = np.where(
        exp_data["pages_viewed"] > 0, exp_data["organic_clicks"] / exp_data["pages_viewed"], 0
    )

# Приводим названия групп к единому формату (A — контроль, B — тест)
exp_data["group"] = exp_data["group"].astype(str).str.strip()
group_map = {"control": "A", "test": "B", "a": "A", "b": "B", "0": "A", "1": "B"}
exp_data["group"] = exp_data["group"].str.lower().map(group_map).fillna(exp_data["group"])

print("\nИнформация о датасете эксперимента:")
exp_data.info()
print("\nПропуски по колонкам:")
print(exp_data.isnull().sum())
print(f"\nДубликатов по user_id: {exp_data['user_id'].duplicated().sum()}")

# Проверка пересечения пользователей между группами
users_a = set(exp_data.loc[exp_data["group"] == "A", "user_id"])
users_b = set(exp_data.loc[exp_data["group"] == "B", "user_id"])
print(f"Пересечение пользователей между группами A и B: {len(users_a & users_b)}")

# Проверка Sample Ratio Mismatch (SRM) с помощью критерия хи-квадрат
group_counts = exp_data["group"].value_counts().sort_index()
_, p_srm = stats.chisquare(group_counts.values)
print("\nРазмеры групп:")
print(group_counts.to_string())
print(
    f"SRM-тест (хи-квадрат): p-value = {p_srm:.4f} -> "
    f"{'Сплит корректен' if p_srm >= 0.05 else 'Внимание: обнаружен SRM!'}"
)

# Проверка баланса устройств между группами
if "device" in exp_data.columns:
    device_ct = pd.crosstab(exp_data["group"], exp_data["device"])
    _, p_dev, _, _ = stats.chi2_contingency(device_ct)
    print(f"\nБаланс устройств между группами (chi2 p-value = {p_dev:.4f}):")
    print(device_ct.assign(mobile_share=(device_ct.get("mobile", 0) / device_ct.sum(axis=1)).round(4)))

# %%
def run_welch_ab_analysis(df: pd.DataFrame, metrics_list: list, alpha: float = 0.05) -> pd.DataFrame:
    """Сравнивает группы A и B с помощью критерия Уэлча и строит 95% ДИ относительного эффекта."""
    group_a_df = df[df["group"] == "A"]
    group_b_df = df[df["group"] == "B"]
    rows = []

    for m in metrics_list:
        a = group_a_df[m].dropna().to_numpy()
        b = group_b_df[m].dropna().to_numpy()

        mean_a, mean_b = np.mean(a), np.mean(b)
        diff = mean_b - mean_a
        rel_effect = (diff / mean_a) * 100 if mean_a != 0 else np.nan

        # Двусторонний t-тест Уэлча
        _, p_val = stats.ttest_ind(b, a, equal_var=False)

        # 95% доверительный интервал для относительного изменения (%)
        var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
        n_a, n_b = len(a), len(b)
        se_diff = np.sqrt(var_a / n_a + var_b / n_b)
        df_welch = (var_a / n_a + var_b / n_b) ** 2 / (
            (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        )
        t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
        ci_low_pct = ((diff - t_crit * se_diff) / mean_a) * 100
        ci_high_pct = ((diff + t_crit * se_diff) / mean_a) * 100

        rows.append(
            {
                "Метрика": m,
                "Среднее (контрольная A)": round(mean_a, 6),
                "Среднее (тестовая B)": round(mean_b, 6),
                "Изменение": round(diff, 6),
                "Эффект, %": round(rel_effect, 2),
                "95% ДИ эффекта, %": f"[{ci_low_pct:+.2f}%; {ci_high_pct:+.2f}%]",
                "p-value": round(p_val, 4),
                "Значимо (<0.05)?": "ДА" if p_val < alpha else "НЕТ",
            }
        )

    return pd.DataFrame(rows)


ab_results = run_welch_ab_analysis(exp_data, METRICS, alpha=ALPHA)
print("\nИтоговая таблица результатов A/B-теста (критерий Уэлча):")
print(ab_results.to_string(index=False))

# %%
# 1. Визуализация относительного эффекта по метрикам
plot_df = ab_results.sort_values("Эффект, %", ascending=True).copy()
colors = [
    "#0F4C5C" if (sig == "ДА" and eff > 0) else ("#B83B2A" if (sig == "ДА" and eff < 0) else "#9AA3AD")
    for eff, sig in zip(plot_df["Эффект, %"], plot_df["Значимо (<0.05)?"])
]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(plot_df["Метрика"], plot_df["Эффект, %"], color=colors, height=0.55)
ax.axvline(0, color="#181C20", linewidth=1.2)

for bar, (_, row) in zip(bars, plot_df.iterrows()):
    eff = row["Эффект, %"]
    p_val = row["p-value"]
    sig_label = "Значимо" if row["Значимо (<0.05)?"] == "ДА" else "Не значимо"
    offset = 0.8 if eff >= 0 else -0.8
    ha = "left" if eff >= 0 else "right"
    ax.text(
        eff + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{eff:+.2f}% ({sig_label})",
        va="center",
        ha=ha,
        fontsize=10
    )

ax.set_xlim(min(plot_df["Эффект, %"].min() - 14, -35), max(plot_df["Эффект, %"].max() + 16, 28))
ax.set_xlabel("Относительное изменение (B vs A), %", fontsize=11)
ax.set_title(
    "Результаты A/B-теста: относительный эффект по метрикам (критерий Уэлча)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.grid(axis="x", linestyle="--", alpha=0.4)
sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.show()

# 2. Сравнение распределений метрик в группах A и B
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Распределение метрик по группам A (контроль) и B (тест)", fontsize=14, fontweight="bold", y=1.02)

for ax, m in zip(axes.flatten(), METRICS):
    sns.boxplot(
        data=exp_data,
        x="group",
        y=m,
        order=["A", "B"],
        hue="group",
        hue_order=["A", "B"],
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Метрика: {m}", fontsize=12)
    ax.set_xlabel("Группа")
    ax.set_ylabel(m)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %%
# Дополнительный сегментный анализ по типам устройств (mobile vs desktop)
if "device" in exp_data.columns:
    for dev in sorted(exp_data["device"].dropna().unique()):
        sub_df = exp_data[exp_data["device"] == dev]
        n_a_dev = (sub_df["group"] == "A").sum()
        n_b_dev = (sub_df["group"] == "B").sum()
        print(f"\n=== Сегмент: {dev} (n_A={n_a_dev}, n_B={n_b_dev}) ===")
        print(run_welch_ab_analysis(sub_df, METRICS, alpha=ALPHA).to_string(index=False))

# %% [markdown]
# ## 5. Общий вывод и бизнес-рекомендации
#
# - **Решение: НЕ ВНЕДРЯТЬ** новое расположение рекламных блоков на всю аудиторию.
# - **Итоги:**
#   - Средняя выручка на пользователя (`revenue`) изменилась на **+0.42%** ($p = 0.9198$) — статистически не значимо.
#   - Кликабельность платной рекламы (`ctr_paid`) выросла на **+15.88%** ($p < 0.0001$), но органический CTR (`ctr_organic`) упал на **-24.74%** ($p < 0.0001$).
#   - Глубина просмотра (`pages_viewed`) осталась стабильной (**+0.65%**, $p = 0.8372$).
# - **Рекомендации:**
#   1. Провести дополнительный анализ монетизации платного трафика (показы, CPC, RPM, случайные клики, постклик-конверсия).
#   2. Исследовать падение `ctr_organic` в разрезе устройств и типов страниц.
#   3. Разработать менее агрессивный вариант размещения рекламных блоков без вытеснения органического контента.
#   4. Заранее определить допустимые пороги для барьерных метрик (тест на не меньшую эффективность / эквивалентность).
#   5. Провести повторный A/B-тест с доработанной вёрсткой.
# %%

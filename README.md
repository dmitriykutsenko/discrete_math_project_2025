# Discrete Math Project 2025

Работа выполнена Куценко Дмитрием и Шатурным Алексеем.

## Оглавление

- [Описание проекта](#описание-проекта)  
- [Структура проекта](#структура-проекта)  
- [src/](#src)  
  - [__init__.py](#srcinitpy)  
  - [features.py](#srcfeaturespy)  
  - [graph_builders.py](#srcgraph_builderspy)  
  - [simulation.py](#srcsimulationpy)  
  - [utils.py](#srcutilspy)  
- [notebooks/](#notebookspy)  
  - [part_1_distribution_exploration.ipynb](#part_1_distribution_explorationipynb)  
  - [part_2_experiment.ipynb](#part_2_experimentipynb)  
- [tests/](#tests)  

---

## Описание проекта

Этот проект посвящён исследованию статистических критериев на графах, построенных по выборкам. Основные шаги:
1. Генерация выборок из различных непрерывных распределений (нормального, Лапласа, Парето, экспоненциального).  
2. Построение по выборке неориентированных графов двух типов:
   - **k-Nearest Neighbors (kNN)** – вершины соединяются, если взаимно входят в k ближайших друг к другу.  
   - **Distance graph** – вершины соединяются, если их расстояние ≤ порогового значения.  
3. Вычисление структурных характеристик (features) полученного графа (максимальная степень, число компонент связности, число треугольников и др.).  
4. Многократное моделирование для оценки мощности статистического критерия на основе сравнения распределений характеристик графов при двух гипотезах H₀ и H₁.  

---

## src/

### __init__.py  
Пустой файл, превращающий папку `src/` в Python-пакет.

---

### features.py  
Модуль для вычисления структурных характеристик (features) графа.
- **SUPPORTED_FEATURES** – кортеж поддерживаемых характеристик:
  - `max_degree`, `min_degree`, `num_components`, `triangle_count`
  - `max_independent_set`, `chromatic_number`, `clique_number`
  - `domination_number`, `clique_cover_number`, `articulation_points`
- **compute_feature(G, feature_name)** → `float`  
  Принимает `networkx.Graph` и имя характеристики, возвращает её значение.  
  Выбрасывает `ValueError`, если имя неизвестно.

---

### graph_builders.py  
Инструменты для построения неориентированных графов по одномерным данным.
- **build_knn_graph(samples: np.ndarray, k: int)** → `networkx.Graph`  
  Симметричный kNN-граф: вершины 0…n−1 соединяются ребром, если каждая входит в k ближайших соседей другой.  
  Проверяет `0 < k < n`; при `k = n−1` возвращает полный граф.
- **build_distance_graph(samples: np.ndarray, d: float)** → `networkx.Graph`  
  Граф расстояний: соединяет пару вершин `(i,j)`, если `|samples[i] − samples[j]| ≤ d`.  
  При `d = ∞` сразу строит полный граф. Проверяет `d ≥ 0`.

---

### simulation.py  
Генерация выборок из стандартных распределений.
- **simulate_sample(n: int, dist: str, params: dict)** → `np.ndarray`  
  Параметры:
  - `n` – размер выборки (целое ≥ 0)
  - `dist` – строка: `"normal"`, `"laplace"`, `"pareto"`, `"exponential"`
  - `params` – словарь параметров:
    - `"normal"`: `{"mu", "sigma"}`
    - `"laplace"`: `{"mu", "beta"}`
    - `"pareto"`: `{"alpha"}`
    - `"exponential"`: `{"lam"}`
  Проверяет корректность `n` и типов параметров.  
  Выбрасывает `ValueError`, если имя распределения неизвестно.

---

### utils.py  
Высокоуровневая логика однопроходного и полного эксперимента.
- **single_run(dist, n, graph_type, param, feature_name)** → `float`  
  1. Генерирует выборку `n` из распределения `dist`  
  2. Строит граф по типу `graph_type` (`"knn"` или `"distance"`) с параметром `param`  
  3. Вычисляет характеристику `feature_name`  
- **run_experiment(dist0, dist1, sample_sizes, params, feature_name, n_sim=500, graph_type="knn", alpha=0.055, seed=None)** → `pd.DataFrame`  
  Многократное моделирование для двух распределений H₀=dist0, H₁=dist1.  
  Возвращает таблицу с колонками:
  - `n`, `param`
  - `mean_H0`, `var_H0`
  - `mean_H1`, `var_H1`
  - `threshold` (по квантили H₀), `power` (оценка мощности при H₁)

---

## notebooks/

### part_1_distribution_exploration.ipynb  
- Исследование распределений:  
  - Реализация и проверка `simulate_sample` для нормального, Лапласа, Парето, экспоненциального.  
  - Визуализация плотностей, гистограмм, QQ-плотов.  
  - Оценка выборочных моментов (среднего, дисперсии) и их зависимость от параметров.

### part_2_experiment.ipynb  
- Проведение графовых экспериментов:  
  - Генерация выборок, построение kNN- и distance-графов.  
  - Вычисление различных характеристик графов (`compute_feature`).  
  - Построение кривых мощности статистического критерия в зависимости от размера выборки и параметра графа.  
  - Сравнение результатов для разных распределений (H₀ vs H₁).

---

## tests/  

Тесты для ключевых функций модуля `src/`, проверяющие:
- Корректность генерации выборок (`simulate_sample`)
- Построение графов (`build_knn_graph`, `build_distance_graph`)
- Вычисление характеристик (`compute_feature`)
- Логика функций `single_run` и `run_experiment`

---

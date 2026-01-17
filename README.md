# 📚 Гибридная Рекомендательная Система Книг

> Система предсказания рейтингов книг на основе ML моделей, embeddings и ensemble методов

**Результат:** RMSE 0.7178 (+11.44% vs baseline)

---

## 🎯 Что внутри

Полный ML pipeline от сырых данных до финальной модели:

1. **EDA** → Быстрый EDA.ipynb
2. **Preprocessing** → preprocessing_new.ipynb
3. **Feature Engineering** → FEATURE-ENGINEERING.ipynb
4. **Baseline Models** → baseline_models_updated.ipynb
5. **ML Models** → advanced_ml_models_FINAL.ipynb
6. **Ensemble & Hybrid** → ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb
7. **🏆 Final Neural Network** → final_neural_network_system.ipynb

---

## 📂 Структура репозитория

### 📓 Jupyter Notebooks (основная работа)

**Анализ и подготовка:**
- `Быстрый EDA.ipynb` — исследовательский анализ данных
- `preprocessing_new.ipynb` — очистка, обработка текстов, создание embeddings
- `FEATURE-ENGINEERING.ipynb` — создание 17 признаков для моделей

**Моделирование:**
- `baseline_models_updated.ipynb` — простые baseline модели (RMSE 0.81)
- `advanced_ml_models_FINAL.ipynb` — CatBoost, LightGBM, KNN (RMSE 0.77)
- `ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb` — Stacking и ансамбли
- **`final_neural_network_system.ipynb`** — 🏆 **ГЛАВНЫЙ ФАЙЛ** — MLP + Stacking (RMSE 0.718)

### 📊 Данные (CSV)

**Исходные данные:**
- `ratings.csv` (982K оценок)
- `books.csv` (10K книг)
- `tags.csv` (34K тегов)
- `book_tags.csv` (1M связей)
- `to_read.csv` (9MB, списки "хочу прочитать")

**Обработанные данные:**
- `complete_dataset.csv` (1.1MB) — объединенный датасет
- `train_dataset.csv` (80MB) — обучающая выборка
- `test_dataset.csv` (10MB) — тестовая выборка
- `train_features_full.csv` (169MB) — train с признаками
- `test_features_full.csv` (21MB) — test с признаками

**Feature tables:**
- `user_features.csv` (2.6MB) — признаки пользователей
- `book_features.csv` (384KB) — признаки книг

### 🧠 Модели и Артефакты (PKL)

**Embeddings:**
- `book_embeddings.pkl` (61MB) — векторы книг (768-dim)
- `train_embeddings.pkl` (7.9GB) — векторы для train
- `test_embeddings.pkl` (968MB) — векторы для test
- `user_embeddings.pkl` (537MB) — профили пользователей

**Обученные модели:**
- `baseline_artifacts.pkl` (746KB) — baseline модели
- `ml_models_full_features.pkl` (247MB) — CatBoost, LightGBM, KNN
- `mlp_model.pkl` (2.1MB) — MLP без embeddings
- `mlp_model_786.pkl` (2.6MB) — MLP с embeddings
- `meta_model.pkl` (1KB) — Stacking Ridge

**Предсказания:**
- `baseline_predictions_test.npy` (839KB)
- `ml_predictions_full.npy` (4.2MB)
- `mlp_predictions_786.npy` (2.7MB)
- `ensemble_predictions.npy` (4.6MB)
- `final_predictions.npy` (4.2MB)

**Метаданные:**
- `preprocessing_metadata.pkl` (1KB)
- `features_metadata.pkl` (1KB)
- `holdout_dict.pkl` (558KB)

### 📈 Результаты (Excel/NPY)

- `baseline_results.xlsx` (2KB) — результаты baseline
- `ml_models_results_full.xlsx` (1KB)
- `ensemble_results.xlsx` (1KB)
- `final_results.xlsx` (1KB)
- `final_results_fixed.csv` (1KB)

### 🖼️ Визуализации (PNG/JPG)

- `baseline_comparison.png` (275KB)
- `baseline_error_distributions.png` (469KB)
- `ml_models_full_features.jpg` (294KB)
- `feature_importance_full.jpg` (219KB)
- `feature_importance_comparison_updated.png` (202KB)
- `ensemble_hybrid_evaluation.jpg` (135KB)
- `final_system_evaluation.png` (114KB)

### 📄 Отчеты

- `отчет.docx` (29KB) — финальный отчет
- `report_final_structured.txt` — структурированная версия

### 📁 Прочее

- `catboost_info/` — логи CatBoost

---

## 🚀 Как запустить

### 1. Клонировать репозиторий

```bash
git clone <repo-url>
cd book-recommendation-system
```

### 2. Установить зависимости

```bash
pip install pandas numpy scikit-learn
pip install catboost lightgbm
pip install sentence-transformers torch
pip install jupyter matplotlib seaborn
```

### 3. Скачать данные

**Важно:** Репозиторий содержит только код и легкие файлы.

Тяжелые файлы (embeddings, модели) нужно скачать отдельно:
- `train_embeddings.pkl` (7.9GB)
- `ml_models_full_features.pkl` (247MB)

Или запустить pipeline с нуля (см. ниже).

### 4. Запустить полный pipeline

**Вариант A: Использовать готовые модели**

```python
# Открыть final_neural_network_system.ipynb
# Запустить все ячейки
# Получить предсказания
```

**Вариант B: Обучить с нуля (4-5 часов)**

```python
# 1. Preprocessing (30 мин)
jupyter notebook preprocessing_new.ipynb

# 2. Feature Engineering (1 час)
jupyter notebook FEATURE-ENGINEERING.ipynb

# 3. Baseline Models (10 мин)
jupyter notebook baseline_models_updated.ipynb

# 4. ML Models (2 часа)
jupyter notebook advanced_ml_models_FINAL.ipynb

# 5. Ensemble (30 мин)
jupyter notebook "ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb"

# 6. Final Neural Network (30 мин)
jupyter notebook final_neural_network_system.ipynb
```

---

## 📊 Результаты

| Модель | RMSE | Улучшение | Файл |
|--------|------|-----------|------|
| Baseline (User+Book) | 0.8104 | - | baseline_models_updated.ipynb |
| CatBoost | 0.7731 | +4.60% | advanced_ml_models_FINAL.ipynb |
| Stacking (3 models) | 0.7217 | +10.94% | ENSEMBLE & HYBRID.ipynb |
| **Stacking + MLP** | **0.7178** | **+11.44%** | **final_neural_network_system.ipynb** |

---

## 🔑 Ключевые файлы

### Для запуска inference:

**Минимальный набор:**
1. `final_neural_network_system.ipynb` — главный notebook
2. `ml_models_full_features.pkl` (247MB) — обученные модели
3. `mlp_model_786.pkl` (2.6MB) — MLP модель
4. `test_features_full.csv` (21MB) — данные для предсказания

**Загрузить и запустить → получить предсказания**

### Для полного воспроизведения:

**Исходные данные:**
- `ratings.csv`, `books.csv`, `tags.csv`, `book_tags.csv`

**Запустить notebooks в порядке:**
1. `preprocessing_new.ipynb`
2. `FEATURE-ENGINEERING.ipynb`
3. `baseline_models_updated.ipynb`
4. `advanced_ml_models_FINAL.ipynb`
5. `ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb`
6. `final_neural_network_system.ipynb`

---

## 📚 Структура Pipeline

```
Исходные данные (CSV)
    ↓
[1] Preprocessing (очистка, embeddings)
    → complete_dataset.csv
    → train/test split
    → book/user embeddings
    ↓
[2] Feature Engineering (17 признаков)
    → interaction features (6)
    → user features (4)
    → book features (3)
    → preprocessing features (4)
    ↓
[3] Baseline Models (простые подходы)
    → User Average: RMSE 0.91
    → User+Book Bias: RMSE 0.81
    ↓
[4] ML Models (gradient boosting)
    → CatBoost: RMSE 0.77 🥇
    → LightGBM: RMSE 0.79
    → KNN: RMSE 0.78
    ↓
[5] Neural Network (MLP)
    → MLP (786 features): RMSE 0.83
    → Переобучение, но полезна для ensemble
    ↓
[6] Ensemble & Stacking
    → Weighted Ensemble: RMSE 0.76
    → Stacking (3 models): RMSE 0.72
    → Stacking + MLP: RMSE 0.718 🏆
```

---

## 💡 Что внутри каждого файла

### Notebooks:

**`preprocessing_new.ipynb`**
- Очистка ratings, books, tags
- Создание embeddings (Sentence Transformers)
- Train/test split
- Output: complete_dataset.csv, embeddings.pkl

**`FEATURE-ENGINEERING.ipynb`**
- Создание 17 числовых признаков
- User/Book/Interaction features
- Output: train_features_full.csv, test_features_full.csv

**`baseline_models_updated.ipynb`**
- Random, Global Average, User/Book Average
- Ridge с 17 признаками
- Output: baseline_artifacts.pkl, RMSE 0.81

**`advanced_ml_models_FINAL.ipynb`**
- CatBoost, LightGBM, Random Forest, KNN
- Feature importance analysis
- Output: ml_models_full_features.pkl, RMSE 0.77

**`ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb`**
- Weighted Ensemble
- Stacking Ridge (meta-learner)
- Hybrid подходы
- Output: meta_model.pkl, RMSE 0.72

**`final_neural_network_system.ipynb`** 🏆
- MLP с embeddings (786 features)
- Интеграция в Stacking
- Финальная система
- Output: mlp_model_786.pkl, final_predictions.npy, RMSE 0.718

---

## 🛠️ Технологии

**ML:**
- CatBoost, LightGBM (gradient boosting)
- KNN (distance-based)
- Ridge (linear regression)
- MLPRegressor (neural network)
- Stacking ensemble

**NLP:**
- Sentence Transformers (all-MiniLM-L6-v2)
- 768-dim embeddings для книг
- 384-dim embeddings для пользователей

**Data:**
- Pandas, NumPy
- Scikit-learn
- CUDA (для embeddings и CatBoost)

---

## 📝 Основные выводы

1. **CatBoost** — лучшая одиночная модель (RMSE 0.77)
2. **Embeddings** критичны: +8% для CatBoost
3. **Stacking** эффективнее простого усреднения: +4%
4. **MLP** слабая индивидуально, но улучшает ensemble (diversity)
5. **Gradient Boosting > Neural Networks** на табличных данных

---

## 📧 Контакты

Проект выполнен в рамках курса по рекомендательным системам.

Финальный результат: **RMSE 0.7178** (улучшение **+11.44%** vs baseline)

---

## 📌 Важные замечания

### Размеры файлов:

**Легкие (можно грузить в Git):**
- Notebooks (.ipynb) — 20-300KB
- CSV до 10MB
- PKL до 10MB
- Images

**Тяжелые (нужен Git LFS или облако):**
- `train_embeddings.pkl` — 7.9GB ⚠️
- `ml_models_full_features.pkl` — 247MB
- `train_features_full.csv` — 169MB
- `test_embeddings.pkl` — 968MB

### Git LFS setup:

```bash
git lfs install
git lfs track "*.pkl"
git lfs track "train_embeddings.pkl"
git lfs track "test_embeddings.pkl"
git add .gitattributes
```

Или использовать Google Drive / Яндекс.Диск для тяжелых файлов.

---

## 🎯 Быстрый старт (5 минут)

**Хочу просто посмотреть результаты:**

1. Открыть `final_neural_network_system.ipynb`
2. Посмотреть ячейку с финальными результатами
3. Готово! Все метрики и визуализации там

**Хочу запустить inference:**

1. Скачать `mlp_model_786.pkl` и `ml_models_full_features.pkl`
2. Открыть `final_neural_network_system.ipynb`
3. Запустить секцию "Prediction Pipeline"
4. Получить предсказания для новых пользователей

**Хочу обучить с нуля:**

1. Запустить notebooks по порядку (1→6)
2. Ждать ~4-5 часов
3. Получить все артефакты

---

**Проект готов к использованию и воспроизведению!** 🚀

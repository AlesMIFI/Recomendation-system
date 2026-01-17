# Book Recommendation System

Комплексная гибридная система рекомендаций книг с использованием классических и нейросетевых подходов.

## 📂 Структура проекта

```
Recomendation-system/
│
├── notebooks/                              # Jupyter ноутбуки
│   ├── preprocessing_new.ipynb            # Предобработка данных
│   ├── FEATURE ENGINEERING.ipynb          # Создание признаков
│   ├── Быстрый EDA.ipynb                  # Exploratory Data Analysis
│   ├── baseline_models_updated.ipynb      # Базовые модели
│   ├── advanced_ml_models_FINAL.ipynb     # ML модели
│   ├── final_neural_network_system.ipynb  # ⭐ Нейросетевая система
│   └── ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb  # ⭐ Гибридная система
│
├── data/
│   ├── raw/                               # Исходные данные
│   │   ├── books.csv
│   │   ├── ratings.csv
│   │   ├── tags.csv
│   │   ├── book_tags.csv
│   │   └── to_read.csv
│   │
│   └── processed/                         # Обработанные данные
│       ├── train_dataset.csv
│       ├── test_dataset.csv
│       ├── book_features.csv
│       ├── user_features.csv
│       └── test_features_full.csv
│
├── models/                                # Сохранённые модели
│   ├── mlp_model*.pkl
│   ├── stacking*.pkl
│   ├── baseline_artifacts.pkl
│   └── book_embeddings.pkl                # 61MB
│
├── results/
│   ├── predictions/                       # Предсказания (.npy)
│   ├── metrics/                           # Метрики качества (.csv)
│   └── visualizations/                    # Графики (.png)
│
├── metadata/                              # Вспомогательные метаданные
│   ├── preprocessing_metadata.pkl
│   ├── features_metadata.pkl
│   └── holdout_dict.pkl
│
├── docs/                                  # Документация
│   └── отчет.docx
│
├── .gitignore
└── README.md
```

## 🚀 Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/AlesMIFI/Recomendation-system.git
cd Recomendation-system
```

### 2. Установить зависимости

```bash
pip install pandas numpy scikit-learn lightgbm catboost tensorflow keras matplotlib seaborn
```

### 3. Запустить ноутбуки

```bash
jupyter notebook
```

## 📋 Последовательность запуска

1. `notebooks/preprocessing_new.ipynb` — Предобработка данных
2. `notebooks/FEATURE ENGINEERING.ipynb` — Создание признаков
3. `notebooks/baseline_models_updated.ipynb` — Базовые модели
4. `notebooks/advanced_ml_models_FINAL.ipynb` — ML модели
5. `notebooks/final_neural_network_system.ipynb` ⭐ — Нейросетевая система
6. `notebooks/ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb` ⭐ — Гибридная система

## 📊 Результаты

Все метрики и визуализации доступны в папке `results/`:

- **Метрики**: `results/metrics/`
- **Визуализации**: `results/visualizations/`
- **Предсказания**: `results/predictions/`

## 🎯 Особенности

- ✅ Гибридная архитектура (collaborative + content-based)
- ✅ Нейросетевые модели (MLP)
- ✅ Ансамблевые методы (Stacking)
- ✅ Feature Engineering с embeddings
- ✅ Полная воспроизводимость

## 📝 Воспроизводимость

Большие файлы (>100MB) не включены в репозиторий, но генерируются автоматически при запуске ноутбуков:

- `train_embeddings.pkl` — генерируется в FEATURE ENGINEERING
- `test_embeddings.pkl` — генерируется в FEATURE ENGINEERING
- `user_embeddings.pkl` — генерируется в FEATURE ENGINEERING

## 🛠️ Технологии

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- LightGBM, CatBoost
- TensorFlow/Keras
- Matplotlib, Seaborn

## 👨‍💻 Автор

[AlesMIFI](https://github.com/AlesMIFI)

## 📄 Лицензия

Проект создан в образовательных целях.

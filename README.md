# Book Recommendation System

Комплексная гибридная система рекомендаций книг с использованием классических и нейросетевых подходов.

## 📂 Структура проекта

\\\
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
│   │   ├── books.csv                      # Информация о книгах
│   │   ├── ratings.csv                    # Оценки пользователей
│   │   ├── tags.csv                       # Теги
│   │   ├── book_tags.csv                  # Связь книг и тегов
│   │   └── to_read.csv                    # Списки "к прочтению"
│   │
│   └── processed/                         # Обработанные данные
│       ├── train_dataset.csv              # Тренировочный набор
│       ├── test_dataset.csv               # Тестовый набор
│       ├── book_features.csv              # Признаки книг
│       ├── user_features.csv              # Признаки пользователей
│       └── test_features_full.csv         # Полный набор признаков
│
├── models/                                # Сохранённые модели
│   ├── mlp_model*.pkl                     # MLP модели
│   ├── stacking*.pkl                      # Stacking ансамбли
│   ├── baseline_artifacts.pkl             # Базовые модели
│   └── book_embeddings.pkl                # Embeddings книг (61MB)
│
├── results/
│   ├── predictions/                       # Предсказания моделей (.npy)
│   ├── metrics/                           # Метрики качества (.csv)
│   └── visualizations/                    # Графики и визуализации (.png)
│
├── metadata/                              # Вспомогательные метаданные
│   ├── preprocessing_metadata.pkl
│   ├── features_metadata.pkl
│   └── holdout_dict.pkl
│
├── docs/                                  # Документация
│   └── отчет.docx                         # Отчёт по проекту
│
├── .gitignore
└── README.md
\\\

## 🚀 Быстрый старт

### 1. Клонировать репозиторий
\\\ash
git clone https://github.com/AlesMIFI/Recomendation-system.git
cd Recomendation-system
\\\

### 2. Установить зависимости
\\\ash
pip install pandas numpy scikit-learn lightgbm catboost tensorflow keras matplotlib seaborn
\\\

### 3. Запустить ноутбуки по порядку
\\\ash
jupyter notebook
\\\

**Последовательность запуска:**
1. \
otebooks/preprocessing_new.ipynb\ — Предобработка данных
2. \
otebooks/FEATURE ENGINEERING.ipynb\ — Создание признаков
3. \
otebooks/baseline_models_updated.ipynb\ — Базовые модели
4. \
otebooks/advanced_ml_models_FINAL.ipynb\ — ML модели
5. \
otebooks/final_neural_network_system.ipynb\ ⭐ — Нейросетевая система
6. \
otebooks/ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb\ ⭐ — Гибридная система

## 📊 Результаты

Все метрики и визуализации доступны в папке \esults/\:
- **Метрики**: \esults/metrics/\
- **Визуализации**: \esults/visualizations/\
- **Предсказания**: \esults/predictions/\

## 🎯 Особенности

- ✅ Гибридная архитектура (collaborative + content-based)
- ✅ Нейросетевые модели (MLP)
- ✅ Ансамблевые методы (Stacking)
- ✅ Feature Engineering с embeddings
- ✅ Полная воспроизводимость

## 📝 Воспроизводимость

Большие файлы (>100MB) не включены в репозиторий, но генерируются автоматически:
- \	rain_embeddings.pkl\ — генерируется в FEATURE ENGINEERING
- \	est_embeddings.pkl\ — генерируется в FEATURE ENGINEERING
- \user_embeddings.pkl\ — генерируется в FEATURE ENGINEERING

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

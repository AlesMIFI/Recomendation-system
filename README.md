# Book Recommendation System

Комплексная гибридная система рекомендаций книг с использованием классических и нейросетевых подходов.

## 📂 Структура проекта

Recomendation-system/
│
├── notebooks/ # Jupyter ноутбуки
│ ├── preprocessing_new.ipynb
│ ├── FEATURE ENGINEERING.ipynb
│ ├── Быстрый EDA.ipynb
│ ├── baseline_models_updated.ipynb
│ ├── advanced_ml_models_FINAL.ipynb
│ ├── final_neural_network_system.ipynb ⭐
│ └── ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb ⭐
│
├── data/
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные
│
├── models/ # Сохранённые модели (61MB)
├── results/
│ ├── predictions/ # Предсказания (.npy)
│ ├── metrics/ # Метрики (.csv)
│ └── visualizations/ # Графики (.png)
│
├── metadata/ # Вспомогательные файлы
├── docs/ # Документация
└── README.md

text

## 🚀 Быстрый старт

```bash
git clone https://github.com/AlesMIFI/Recomendation-system.git
cd Recomendation-system
pip install pandas numpy scikit-learn lightgbm catboost tensorflow keras matplotlib seaborn
jupyter notebook
📋 Последовательность запуска
preprocessing_new.ipynb — Предобработка

FEATURE ENGINEERING.ipynb — Создание признаков

baseline_models_updated.ipynb — Базовые модели

advanced_ml_models_FINAL.ipynb — ML модели

final_neural_network_system.ipynb ⭐ — Нейросети

ENSEMBLE & HYBRID RECOMMENDATION SYSTEM.ipynb ⭐ — Гибридная система

🎯 Особенности
✅ Гибридная архитектура (collaborative + content-based)

✅ Нейросетевые модели (MLP)

✅ Ансамблевые методы (Stacking)

✅ Feature Engineering с embeddings


# DA-3-30 — Комплексный EDA-отчёт: данные, визуализации, выводы  

## Описание  
Программа **`eda_report_generator.py`** создаёт **полноценный исследовательский отчёт (EDA)** по данным, включая анализ пропусков, выбросов и распределений, автоматическое создание новых признаков, выполнение статистических тестов и построение визуализаций.  
Итог сохраняется в **HTML-отчёт**, при наличии `pdfkit` и `wkhtmltopdf` — также в **PDF**.

---

### Основные этапы работы программы:
1. **Выбор датасета**  
   - `iris` — классификация (по умолчанию)  
   - `boston` — регрессия (если поддерживается в вашей версии `scikit-learn`)

2. **Базовый анализ данных (EDA)**  
   - Проверка пропусков  
   - Анализ выбросов (по IQR)  
   - Распределения, скошенность, эксцесс  

3. **Генерация признаков (Feature Engineering)**  
   Создаются **5 и более** новых признаков:
   - Нормированные (`_z`)  
   - Отношения длин/ширин  
   - Площади (`petal_area`, `sepal_area`)  
   - Взаимодействия и квадраты признаков  
   - Лог-преобразования и z-индекс (для регрессии)

4. **Статистические тесты (3 и более):**  
   - Shapiro–Wilk — нормальность распределений  
   - ANOVA / Kruskal–Wallis — различия между классами (iris)  
   - Пирсон / Левен — корреляции и равенство дисперсий (boston)

5. **Визуализации (5+):**  
   - Гистограммы  
   - Boxplots  
   - Тепловая карта корреляций  
   - Scatter matrix  
   - Violin-plot (для классификации) или Scatter-plot (для регрессии)

6. **Отчёт (HTML/PDF)**  
   - Сводка данных  
   - Таблицы с пропусками, выбросами, распределениями  
   - Результаты тестов в JSON-виде  
   - Визуализации (встроены Base64)  
   - Краткие выводы  

---

## Пример запуска  
**HTML-отчёт (по умолчанию):**
```bash
python eda_report_generator.py
```

Явный выбор датасета и формата:
```bash
python eda_report_generator.py --dataset iris --out eda_report_iris.html
```

С генерацией PDF:
```bash
python eda_report_generator.py --dataset iris --out eda_report_iris.html --pdf
```

Для Boston:
```bash
python eda_report_generator.py --dataset boston --out eda_report_boston.html --pdf
```
---

## Установка и запуск 
1. Клонировать репозиторий:
```
  git clone https://github.com/AlexandraAgapova/AI-development.git
  cd AI-development/DA-3-40
```
3. Создать виртуальное окружение:
```
  python -m venv venv
  venv\Scripts\activate  # Windows
  source venv/bin/activate  # Linux/Mac
```
5. Установить зависимости:
```bash
  pip install -r requirements.txt
```
6. Для экспорта PDF установите также системную утилиту:
```bash
  sudo apt install wkhtmltopdf  # Linux
  choco install wkhtmltopdf     # Windows (через Chocolatey)
```
7. Запустить проект:
```bash
  python eda_report_generator.py
```

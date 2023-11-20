# balance_metabolic_syndrome
Проект, прогнозирующий наличие метаболического синдрома на основе информации о пациенте, с использованием фреймворка [Streamlit](https://streamlit.io).
Данные, используемые в этом репозитории, являются набором данных [Metabolic Syndrome](https://www.kaggle.com/datasets/antimoni/metabolic-syndrome) от Kaggle.

* Файлы
   * app.py: файл приложения streamlit
   * model.py: скрипт для генерации модели классификатора Bagging
   * data.csv: файл данных 
   * model_weights.mw: предварительно обученная модель
   * requirements.txt: файлы с требованиями к пакету

## Запустите демонстрацию локально

### Оболочка
Для прямого запуска streamlit локально в корневой папке репозитория следующим образом:

> python -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> streamlit run app.py

Откройте http://localhost:8501, чтобы просмотреть приложение.

## Hазвертывание в облаке

Пройдите по прямой ссылке https://metabolicsyndrome.streamlit.app.

Разработка системы анализа эмоциональной окраски музыкальных текстов: определение настроения и эмоций в песнях.
===========
[Данные, нужные для запуска кода](#title1)


[Запуск кода](#title2)  

[Пример текста песни](#title3)

[Как работает код](#title4)

[Результаты программы](#title5)

## <a id="title1">Данные, нужные для запуска кода</a>
1. Для использования программы нужно скачать файл RedditData.csv либо в GitHub, либо по [ссылке](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
   ![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/5cc51daf-e9c0-42e7-8e63-7f1993bba0a8)
   ![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/62228688-bf83-41e2-8aaa-f2026196600c)


3. Лабораторная_работа_по_компьютерной_лингвистике_и_обработке_естественного_языка.ipynb
## <a id="title2">Запуск кода</a>
Загрузить RedditData.csv в GoogleColab:   
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/4b8cf61e-7b59-4fff-860c-3639764f057a)  
Или в Kaggle:    
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/ff42c0a7-29b3-4562-990a-06b84b98efbd)

Запустить код по порядку до данного фрагмента кода:  
```python
# @title Текст заголовка по умолчанию
text_of_song = "" # @param {type:"string"}
predict(text_of_song)
```
На этом этапе нужно будет вставить текст песни:  
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/abc5ad7b-48a3-4566-85db-206baee9de3a)

## <a id="title3">Пример текста песни</a>
1. How long, must this feeling go on How long, must I stand the pain and How long, must this feeling go on
2. Huh (Because I'm happy) Clap along if you feel like a room without a roof (Because I'm happy) Clap along if you feel like happiness is the truth(Because I'm happy)

## <a id="title4">Как работает код</a>
Этот код позволяет проанализировать и предсказать настроение текста с использованием нейронной сети на основе данных, которые содержат текстовые комментарии и их категории. Загружаются необходимые библиотеки для обработки данных и построения модели нейронной сети с использованием библиотеки TensorFlow и Keras. Подсчитывается количество элементов в каждой из категорий (-1, 0, 1), и результат визуализируется в виде столбчатой диаграммы. Создается словарь, в котором ключами являются текстовые комментарии, а значениями их категории. Вычисляется средняя длина слов в категориях Negative, Neutral и Positive, а также строится график изменения длины слов для каждой категории. Производится токенизация текстовых данных и их подготовка для обучения нейронной сети. Строится и компилируется модель нейронной сети с использованием различных слоев (Embedding, LSTM, Bidirectional, Dense) и обучается на предварительно подготовленных данных. Визуализируются графики точности и функции потерь на тренировочном и валидационном наборах данных. Реализуется функция predict, которая принимает текстовую строку и использует обученную модель для предсказания настроения текста (Positive, Negative, Neutral). Вводится текстовая строка для предсказания настроения с использованием ранее обученной модели. 


## <a id="title5">Результаты программы</a>
Выполненная программа на основе текстов выше:  
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/51ba50b0-76e1-46e9-8732-26c77724c469)
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/15a5c4b2-3367-4ced-a2f5-869f8480eb9b)



  

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
Загружаем необходимые библиотеки:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,Bidirectional,LSTM,Dense,Dropout
from tensorflow.keras.utils import to_categorical
```
Добавляем ссылку на RedditData.csv 
```python
df=pd.read_csv('ваша_ссылка')
df.head(5)
```
```python
# Получение списка категорий из столбца "category" фрейма данных
dist = list(df.category)

# Инициализация списка для подсчета количества элементов в каждой категории
pp = [0, 0, 0]

# Проход по списку категорий и увеличение счетчиков соответствующих категорий
for i in dist:
    if i == -1:
        pp[0] += 1  # Увеличение счетчика для категории -1
    elif i == 0:
        pp[1] += 1  # Увеличение счетчика для категории 0
    else:
        pp[2] += 1  # Увеличение счетчика для остальных категорий

# Вывод результата подсчета
print(pp)
```
```python
labels=['Negative','Neutral','Positive']
sns.barplot(x=labels,y=pp)
plt.show()
```
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/51da169c-3888-488b-b895-c55ca9b43666)

```python
comment=list(df.clean_comment.astype(str))
sentiment=list(df.category)
reddit_dict=dict(zip(comment,sentiment))
```
```python
print(list(reddit_dict.items())[:5])
```
Распределение по категориям, где Негативные:-1, Позитивные:1, Нейтральные:0.
```python
Neg_list=[]
Pos_list=[]
Neutral_list=[]
for i,j in reddit_dict.items():
    if j==-1:
        Neg_list.append(i)
    elif j==0:
        Neutral_list.append(i)
    else:
        Pos_list.append(i)
```
```python
print(Neg_list[:2],'\n',Neutral_list[:2],'\n',Pos_list[:2])
```
```python
# Создаем пустой список для хранения длин строк из списка Pos_list
pos_len = []

# Итерируемся по элементам списка Pos_list
for i in Pos_list:
    # Добавляем длину текущей строки (i) в список pos_len
    pos_len.append(len(i))
```
```python
# Создаем пустой список для хранения длин строк из списка Neg_list
neg_len = []

# Итерируемся по элементам списка Neg_list
for i in Neg_list:
    # Добавляем длину текущей строки (i) в список neg_len
    neg_len.append(len(i))

```
```python
# Создаем пустой список для хранения длин строк из списка Neutral_list
Neutral_len = []

# Итерируемся по элементам списка Neutral_list
for i in Neutral_list:
    # Добавляем длину текущей строки (i) в список Neutral_len
    Neutral_len.append(len(i))

```
```python
import matplotlib.pyplot as plt

plt.subplots(figsize=(20, 8))

plt.title("Word Length Variation")

# Построение линий для длины слов для каждой из категорий
# синим цветом для Neutral, красным для Negative и зеленым для Positive
plt.plot(Neutral_len[:250], c='b', label='Neutral')
plt.plot(neg_len[:250], c='r', label='Negative')
plt.plot(pos_len[:250], c='g', label='Positive')

plt.legend(loc='upper left')

plt.show()
```
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/343a9a11-6fa5-42da-91aa-c947d0b3175e)

```python
# Вычисление среднего значения длины слов для категории Positive
pos_mean = sum(pos_len) // len(pos_len)

# Вычисление среднего значения длины слов для категории Negative
neg_mean = sum(neg_len) // len(neg_len)

# Вычисление среднего значения длины слов для категории Neutral
neutral_mean = sum(Neutral_len) // len(Neutral_len)

# Вычисление общего среднего значения длины слов
combined_mean = (sum(pos_len) + sum(neg_len) + sum(Neutral_len)) // (len(pos_len) + len(neg_len) + len(Neutral_len))
```
```python
plt.title("Average Word Length")
sns.barplot(x=['Negative','Neutral','Positive','Combined'],y=[neg_mean,neutral_mean,pos_mean,combined_mean])
plt.show()
```
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/b98d7333-6193-4711-9c43-8f4a76aceee8)

```python
# Извлечение столбца 'clean_comment' из DataFrame и преобразование его в строки
X = df['clean_comment'].astype('str')

# Вывод первых 5 элементов полученной Series
print(X[:5])

```
```python
lp=""
for i in X:
    lp+=i+" "
print(lp[:100])
```
```python
st=lp.split(' ')
dict_len=len(set(st))
```
```python
dict_len,len(st)
```
```python
# Создание объекта класса Tokenizer с указанием максимального количества слов (num_words),
# преобразование всех слов в нижний регистр (lower=True) и токена для неизвестных слов (oov_token)
tokenizer = Tokenizer(num_words=dict_len, lower=True, oov_token="OOV")

# Обучение токенизатора на текстовых данных X
tokenizer.fit_on_texts(X)

```
```python
len(tokenizer.word_index)
```
```python
# Преобразование текстовых данных в последовательности токенов
X_train = tokenizer.texts_to_sequences(X)

# Дополнительная обработка: заполнение и обрезка последовательностей токенов
# до фиксированной длины (maxlen=175), заполнение производится после текста (padding='post'),
# обрезка производится после текста (truncating='post')
X_train_padded = pad_sequences(X_train, maxlen=175, padding='post', truncating='post')

```
```python
X_train[:2]
```
```python
df['category']=df['category'].replace({-1:2})
```
```python
mp={0:"Neutral",1:"Positve",2:"Negative"}
```
```python
Y=df['category'].values
```
```python
# Преобразование вектора меток классов Y в матрицу категорий (one-hot encoding)
Y_hot = to_categorical(Y)
```
```python
print(Y_hot[:3])
```
```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense

# Создание пустой последовательной модели
model = Sequential()

# Добавление слоя Embedding для преобразования индексов слов в векторы фиксированной размерности (64)
# input_length=175 указывает максимальную длину входных последовательностей
model.add(Embedding(dict_len, 64, input_length=175))

# Добавление слоя Dropout для регуляризации и предотвращения переобучения (отсев 30% нейронов)
model.add(Dropout(0.3))

# Добавление слоя Bidirectional LSTM (175 нейронов), который обрабатывает последовательности в обоих направлениях
# return_sequences=True указывает, что этот слой возвращает последовательности для следующего слоя
model.add(Bidirectional(LSTM(175, return_sequences=True)))

# Добавление слоя Dropout для регуляризации
model.add(Dropout(0.3))

# Добавление еще одного слоя Bidirectional LSTM (350 нейронов) с возвращением последовательностей
model.add(Bidirectional(LSTM(350, return_sequences=True)))

# Добавление слоя Dropout для регуляризации
model.add(Dropout(0.3))

# Добавление еще одного слоя Bidirectional LSTM (700 нейронов) без возвращения последовательностей
model.add(Bidirectional(LSTM(700)))

# Добавление полносвязного слоя с 3 нейронами и функцией активации softmax для классификации на 3 класса
model.add(Dense(3, activation='softmax'))

# Вывод краткой информации о модели
print(model.summary())
```
```python
# Компиляция модели с оптимизатором 'adam', функцией потерь 'categorical_crossentropy'
# и метрикой 'accuracy' для оценки производительности модели в процессе обучения.
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
```
```python
hist=model.fit(X_train_padded,Y_hot,epochs=5,validation_split=0.2)
```
```python
import matplotlib.pyplot as plt

# Построение графика точности на тренировочном наборе данных (синий цвет)
plt.plot(hist.history['accuracy'], c='b', label='Training')

# Построение графика точности на валидационном наборе данных (красный цвет)
plt.plot(hist.history['val_accuracy'], c='r', label='Validation')

# Добавление легенды в правый нижний угол графика
plt.legend(loc='lower right')

# Отображение графика
plt.show()

```
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/11f8adec-0aea-49d6-a48d-63ebcdd056fc)

```python
import matplotlib.pyplot as plt

# Построение графика функции потерь на тренировочном наборе данных (синий цвет)
plt.plot(hist.history['loss'], c='b', label='Training')

# Построение графика функции потерь на валидационном наборе данных (красный цвет)
plt.plot(hist.history['val_loss'], c='r', label='Validation')

# Добавление легенды в правый верхний угол графика
plt.legend(loc='upper right')

# Отображение графика
plt.show()

```
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/30aed73e-7be0-4d78-a3c6-765bdd19a08c)

```python
def predict(s):
    # Создаем список, содержащий текстовую строку s
    X_tes = []
    X_tes.append(s)
    
    # Преобразование текстовой строки в последовательность токенов
    X_test = tokenizer.texts_to_sequences(X_tes)
    
    # Дополнение и обрезка последовательности токенов
    X_test_padded = pad_sequences(X_test, maxlen=175, padding='post', truncating='post')
    
    # Предсказание класса с использованием обученной модели
    sent = int(model.predict_classes(X_test_padded))
    
    # Вывод предсказанного настроения
    print("The Predicted Sentiment is", mp[sent])

```
```python
# @title Текст заголовка по умолчанию
text_of_song = "" # @param {type:"string"}
predict(text_of_song)
```

## <a id="title5">Результаты программы</a>
Выполненная программа на основе текстов выше:  
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/51ba50b0-76e1-46e9-8732-26c77724c469)
![image](https://github.com/kurrosan/ComputerLanguage/assets/120035199/15a5c4b2-3367-4ced-a2f5-869f8480eb9b)



  

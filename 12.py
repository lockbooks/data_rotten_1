# импортируем все зависимости
from collections import Counter
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# загружаем датасет в переменную
df = pl.read_csv('../rotten_tomatoes_movie_reviews.csv')
# проверяем, что всё загрузилось: выводим первые 5 значений
print(df.head())

# задаём SQL-запрос для выборки определённых столбцов
# из датасета и выводим обновлённые данные
query = """
SELECT id, reviewText, scoreSentiment, criticName, isTopCritic, reviewState
FROM self
"""
# применяем запрос
df = df.sql(query)
# проверяем результат
print(df.head())


# определяем функцию для очистки текстовых данных в столбце reviewText:
# удаляем ссылки, спецсимволы, числа и приводим текст к нижнему регистру.
# pl.DataFrame — объект двумерного массива из строк и столбцов
def clean_data(df_cleaning: pl.DataFrame):
    # метод with_columns добавляет столбцы
    df_cleaning = df_cleaning.with_columns(
        # очищаем все значения в столбце reviewText методом строки replace_all:
        # заменяем ненужные символы на пробелы или более удобные для анализа символы
        pl.col('reviewText')
        .str.to_lowercase()
        .str.replace_all("'s", ' is')
        .str.replace_all("n't", ' not')
        .str.replace_all("'m", ' am')
        .str.replace_all('((www\.[^\s]+)|(https?://[^\s]+))', '')
        .str.replace_all('@[^\s]+', '')
        .str.replace_all(r"\d+", "")
        .str.replace_all("\W", " ")
        .str.replace_all('https?://\S+|www\.\S+', '')
        .str.replace_all('<.*?>+', '')
        .str.replace_all('\n', '')
        .str.replace_all(r'#([^\s]+)', r'\1')
        .str.replace_all('\.', ' ')
        .str.replace_all('\"', ' ')
        .str.replace_all('\w*\d\w*', '')
        .str.replace_all('\!', ' ')
        .str.replace_all('\-', ' ')
        .str.replace_all('\:', ' ')
        .str.replace_all('\)', ' ')
        .str.replace_all('\,', ' ')
        .str.replace_all('[\s]+', ' ')
        .str.strip_chars('\'"')
    )
    return df_cleaning


# применяем функцию очистки данных к датасету
df = clean_data(df)

# пишем SQL-запрос для фильтра: оставляем только те строки,
# где значение в столбце reviewText не пустое - значит, рецензия есть
query = """
SELECT *
FROM self
WHERE reviewText <> ''
"""
# применяем запрос
df = df.sql(query)
# проверяем: выводим первые 10 фильмов с рецензиями
print(df.head(10))

# 1 - круговая диаграмма
# строим круговую диаграмму для отображения распределения рецензий по scoreSentiment
unique, count = np.unique(df['scoreSentiment'], return_counts=True)
# передаём в библиотеку matplotlib массив  значений
# posititve-negative и их количество на каждый фильм
plt.pie(x=count, labels=unique, autopct='%.0f%%')
# необязательный параметр
# plt.xlabel('Подпись для оси')
# пишем название диаграммы
plt.title('Общая оценка распределения положительных и негативных рецензий')
# выводим на экран
plt.show()


# 2 - столбчатая горизонтальная диаграмма для критиков
# отбираем рецензии только топ-критиков
topCriticsQuery = """
SELECT *
FROM self
WHERE isTopCritic=1
"""
topCriticDf = df.sql(topCriticsQuery)

# считаем количество рецензий каждого из топ-критиков
# и сортируем их в порядке убывания
query = """
SELECT criticName, COUNT(*) AS reviewCount
FROM self
GROUP BY criticName
ORDER BY reviewCount DESC
"""
topCriticCount = topCriticDf.sql(query)

# строим горизонтальный график для первых 30 критиков с наибольшим количеством рецензий
# устанавливаем оси: количества рецензий по x, имя критика по y
sns.barplot(x=topCriticCount['reviewCount'].head(30),
            y=topCriticCount['criticName'].head(30),
            # устанавливаем горизонтальную ориентацию графика
            orient='h',
            # устанавливаем цветовую маркировку для каждого столбца
            hue=topCriticCount['criticName'].head(30)
            )
# задаём название для диаграммы
plt.title('30 Топ-критиков с наибольшим количеством рецензий')
# увеличиваем отступ слева
plt.subplots_adjust(left=0.3)
# рисуем график
plt.show()

# 3 - столбчатая горизонтальная диаграмма для фильмов по количеству отзывов
# задаём SQL-запрос для получения списка фильмов
# с наибольшим количеством отрицательных отзывов
query = """
SELECT id, COUNT(*) as negReviewCount
FROM self
WHERE scoreSentiment='NEGATIVE'
GROUP BY id
ORDER BY negReviewCount DESC
LIMIT 50
"""
# применяем запрос
topNegMov = df.sql(query)
# строим столбчатый график одного цвета
# задаём размер графика
plt.figure(figsize=(10, 8))
# устанавливаем оси X и Y, ориентацию графика, цвет столбцов
sns.barplot(x=topNegMov['negReviewCount'],
            y=topNegMov['id'],
            orient='h',
            color='red'
            )
# устанавливаем метку по X
plt.xlabel('Количество негативных отзывов')
# устанавливаем метку по Y
plt.ylabel('Название фильма')
# задаём название для диаграммы
plt.title('50 фильмов с наибольшим количеством отрицательных рецензий')
# увеличиваем отступ слева
plt.subplots_adjust(left=0.3)
# рисуем график
plt.show()


# 4 - строим wordcloud по количеству слов в положительных и отрицательных рецензиях
# загружаем зависимости для стоп-сллов
nltk.download('punkt')
nltk.download('stopwords')

# составляеем список стоп-слов: добавляем готовые из английского словаря
stop_words_all = set(stopwords.words('english'))
# записываем свои
custom_stop_words = {'movie', 'film', 'one', 'would', 'like', 'quite', 'enough',
                     'character', 'comedy', 'films', 'make', 'time', 'see', 'way',
                     'year', 'even', 'good', 'much', 'work', 'take', 'characters',
                     'still', 'review', 'feel', 'seem', 'scene', 'though', 'funny',
                     'scene', 'find', 'day', 'actor', 'play', 'know', 'bit', 'set',
                     'making', 'turn', 'want', 'feature', 'filmmaker', 'around',
                     'hour', 'think', 'show'}
# добавляем свои слова в набор стоп-слов
stop_words_all.update(custom_stop_words)


# пишем функцию для очистки текста от стоп-слов
def clean_text(text, stop_words):
    # токенизируем и приводим к нижнему регистру
    tokens = word_tokenize(text.lower())
    # удаляем стоп-слова и небуквенные символы
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # возвращаем очищенный текст
    return ' '.join(filtered_tokens)


# пишем функцию для нахождения самых частых слов в рецензиях
def get_most_common_words(text, num_words=100):
    # разбиваем текст на слова
    words = text.split()
    # создаём коллекцию вида: слово - количество повторений
    counter = Counter(words)
    # возвращаем первые 100 слов из всех переданных в функцию рецензий
    return counter.most_common(num_words)


# пишем запрос для выделения всех положительных рецензий
pos_query = """
SELECT reviewText
FROM self
WHERE scoreSentiment = 'POSITIVE'
"""

# пишем запрос для выделения всех отрицательных рецензий
neg_query = """
SELECT reviewText
FROM self
WHERE scoreSentiment = 'NEGATIVE'
"""

# выделяем все позитивные рецензии через запрос
pos_df = df.sql(pos_query)['reviewText']
# объединяем все слова из рецензий
all_pos_words = ' '.join(pos_df.to_list())
# очищаем текст от стоп-слов
cleaned_pos_text = clean_text(all_pos_words, stop_words_all)
# выделяем самые часто встречающиеся слова
pos_common_words = get_most_common_words(cleaned_pos_text)

# выделяем все негативные рецензии через запрос
neg_df = df.sql(neg_query)['reviewText']
# объединяем все слова из рецензий
all_neg_words = ' '.join(neg_df.to_list())
# очищаем текст от стоп-слов
cleaned_neg_text = clean_text(all_neg_words, stop_words_all)
# выделяем самые часто встречающиеся слова
neg_common_words = get_most_common_words(cleaned_neg_text)

# преобразуем самые частые слова в множества
# позитивных и негативных часто встречающихся слов
top_pos_words = {word for word, _ in pos_common_words}
top_neg_words = {word for word, _ in neg_common_words}

# объединяем слова из обоих наборов
top_words = top_pos_words | top_neg_words

# обновляем стоп-слова
stop_words_all.update(top_words)


# создаём облако слов для положительных отзывов
all_word = ' '.join(pos_df.to_list())

cleaned_text = clean_text(all_word, stop_words_all)

wordcloud = WordCloud(width=800, height=500,
                      colormap='Blues',
                      max_font_size=110,
                      collocations=False).generate(cleaned_text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Самые частые слова из положительных рецензий")
plt.show()


# создаём облако слов для отрицательных отзывов
all_word = ' '.join(neg_df.to_list())

cleaned_text = clean_text(all_word, stop_words_all)

wordcloud = WordCloud(width=800, height=500,
                      colormap='Oranges_r',
                      max_font_size=110,
                      collocations=False).generate(cleaned_text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Самые частые слова из отрицательных рецензий")
plt.show()

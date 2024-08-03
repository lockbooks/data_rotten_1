# 1 - импорт всего
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# 2 - проверем датасет
df = pl.read_csv('../rotten_tomatoes_movie_reviews.csv')
print(df.head())

# 3 - Выполняем SQL-запрос для выборки определенных столбцов из DataFrame и выводим обновленные данные.
query = """
SELECT id, reviewText, scoreSentiment, criticName, isTopCritic, reviewState
FROM self
"""
df = df.sql(query)
print(df.head())


# 4 - Определяем функцию для очистки текстовых данных в столбце reviewText,
# удаляя ссылки, спецсимволы, числа и приводя текст к нижнему регистру.
def clean_data(df: pl.DataFrame):
    df = df.with_columns(
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
    return df


# 5 - Применяем функцию очистки данных к DataFrame и выполняем SQL-запрос
# для фильтрации строк, где reviewText не пустой, затем выводим результат.
df = clean_data(df)

query = """
SELECT *
FROM self
WHERE reviewText <> ''
"""
df = df.sql(query)
print(df.head())

# 6 - Строим круговую диаграмму для отображения распределения классов отзывов по scoreSentiment.
unique, count = np.unique(df['scoreSentiment'], return_counts=True)
plt.pie(x=count, labels=unique, autopct='%.0f%%')
plt.xlabel('Target')
plt.title('scoreSentiment Class Distribution')
plt.show()

# 7 - Отбираем отзывы топ-критиков и строим горизонтальный
# барплот для 30 критиков с наибольшим количеством отзывов.
topCriticsQuery = """
SELECT *
FROM self
WHERE isTopCritic=1
"""
topCriticDf = df.sql(topCriticsQuery)

query = """
SELECT criticName, COUNT(*) AS reviewCount
FROM self
GROUP BY criticName
ORDER BY reviewCount DESC
"""
topCriticCount = topCriticDf.sql(query)

sns.barplot(x=topCriticCount['reviewCount'].head(30),
            y=topCriticCount['criticName'].head(30),
            orient='h',
            hue=topCriticCount['criticName'].head(30))
plt.title('Top 30 Critic with Highest Review Counts')
plt.legend('')
plt.show()

# 9 - Выполняем SQL-запрос для получения списка фильмов
# с наибольшим количеством отрицательных отзывов и выводим результат.
query = """
SELECT id, COUNT(*) as negReviewCount
FROM self
WHERE scoreSentiment='NEGATIVE'
GROUP BY id
ORDER BY negReviewCount DESC
"""
topNegMov = df.sql(query)
print(topNegMov.head())

# 9.1

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Добавьте любые другие слова, которые вы хотите исключить
custom_stop_words = {'movie', 'film', 'one', 'would', 'like'}
stop_words.update(custom_stop_words)


def clean_text(text, stop_words):
    tokens = word_tokenize(text.lower())  # Токенизация и приведение к нижнему регистру
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Удаление стоп-слов и небуквенных символов
    return ' '.join(filtered_tokens)


# 10 - Создаем облако слов для положительных отзывов, объединив текст
# всех положительных отзывов и визуализировав его с помощью WordCloud.
pos_query = """
SELECT reviewText
FROM self
WHERE scoreSentiment = 'POSITIVE'
"""

pos_df = df.sql(pos_query)['reviewText']

all_word = ' '.join(pos_df.to_list())
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_word)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Review Wordcloud")
plt.show()

# 11 - Создаем облако слов для отрицательных отзывов, объединив текст
# всех отрицательных отзывов и визуализировав его с помощью WordCloud.
neg_query = """
SELECT reviewText
FROM self
WHERE scoreSentiment = 'NEGATIVE'
"""

neg_df = df.sql(neg_query)['reviewText']

all_word = ' '.join(neg_df.to_list())
wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_word)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Negative Review Wordcloud")
plt.show()

# 12 тут надо будет вычесть одно из другого

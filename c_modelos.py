import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objs as go
from mlxtend.preprocessing import TransactionEncoder

# Conectar a la base de datos
conn = sql.connect('data/db_movies2')

# Cargar la tabla 'full_ratings'
full_ratings = pd.read_sql('SELECT * FROM full_ratings', conn)

# Extraer el año de la columna 'title'
full_ratings['year'] = full_ratings['title'].str.extract(r'\((\d{4})\)').astype(float)

# Normalizar la columna 'year'
scaler = MinMaxScaler()
full_ratings[['year_normalized']] = scaler.fit_transform(full_ratings[['year']])

# Separar los géneros en columnas individuales
genres = full_ratings['genres'].str.split('|')
te = TransactionEncoder()
genres_matrix = te.fit_transform(genres)
genres_df = pd.DataFrame(genres_matrix, columns=te.columns_)

# Unir las nuevas columnas de géneros con el DataFrame original
full_ratings = pd.concat([full_ratings, genres_df], axis=1)

# Eliminar columnas innecesarias para los modelos
general = full_ratings.drop(columns=['user_id', 'movieId', 'rating', 'genres', 'title'])
final2 = pd.get_dummies(general)

final2

###############################################################
# 1. Recomendación basada en popularidad
# Seleccionar las películas más calificadas
top_rated_movies = pd.read_sql('''
    SELECT title, COUNT(*) AS num_ratings, AVG(rating) AS avg_rating
    FROM full_ratings
    GROUP BY title
    HAVING num_ratings >= 10
    ORDER BY avg_rating DESC, num_ratings DESC
    LIMIT 10
''', conn)

# Mostrar las 10 películas más calificadas
fig_popularity = px.bar(top_rated_movies, x='title', y='avg_rating', title='Top 10 Películas por Calificación Promedio')
fig_popularity.show()

###################################################

# Calcular la calificación promedio por película y año de lanzamiento
top_movies_by_year = full_ratings.groupby(['year', 'title'])['rating'].mean().reset_index()

# Ordenar las películas por año y calificación promedio
top_movies_by_year = top_movies_by_year.sort_values(by=['year', 'rating'], ascending=[True, False])

# Obtener las películas mejor calificadas por año (una por cada año)
best_movies_per_year = top_movies_by_year.groupby('year').head(1)

# Gráfico de barras con las películas mejor calificadas por año
fig = px.bar(best_movies_per_year, x='year', y='rating', color='title',
             title='Películas Mejor Calificadas por Año de Lanzamiento',
             labels={'rating': 'Calificación Promedio', 'year': 'Año de Lanzamiento', 'title': 'Título de la Película'},
             hover_data=['title'])

# Mostrar el gráfico
fig.show()
#####################################################


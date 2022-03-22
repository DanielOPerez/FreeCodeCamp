# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# *Note:  You are  currently  reading this  using Google  Colaboratory
# *which  is a  cloud-hosted version  of Jupyter  Notebook. This  is a
# *document containing both text  cells for documentation and runnable
# *code cells. If you are unfamiliar with Jupyter Notebook, watch this
# *3-minute    introduction    before   starting    this    challenge:
# *https://www.youtube.com/watch?v=inN8seMm7UI*

# ---

# In this challenge,  you will create a  book recommendation algorithm
# using **K-Nearest Neighbors**.

# You          will          use          the          [Book-Crossings
# dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/). This
# dataset  contains 1.1  million ratings  (scale of  1-10) of  270,000
# books by 90,000 users.

# After importing  and cleaning the data,  use `NearestNeighbors` from
# `sklearn.neighbors` to  develop a  model that  shows books  that are
# similar to  a given book.  The Nearest Neighbors  algorithm measures
# distance to determine the “closeness” of instances.

# Create a  function named  `get_recommends` that  takes a  book title
# (from the  dataset) as an argument  and returns a list  of 5 similar
# books with their distances from the book argument.

# This code:

# `get_recommends("The  Queen   of  the  Damned   (Vampire  Chronicles
# (Paperback))")`

# should return:

# ```
# [
#   'The Queen of the Damned (Vampire Chronicles (Paperback))',
#   [
#     ['Catch 22', 0.793983519077301], 
#     ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
#     ['Interview with the Vampire', 0.7345068454742432],
#     ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
#     ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
#   ]
# ]
# ```

# Notice that the data returned from `get_recommends()` is a list. The
# first  element in  the  list is  the  book title  passed  in to  the
# function. The  second element  in the  list is a  list of  five more
# lists. Each  of the five lists  contains a recommended book  and the
# distance from  the recommended  book to  the book  passed in  to the
# function.

# If you graph the dataset (optional), you will notice that most books
# are not rated frequently. To ensure statistical significance, remove
# from the  dataset users with  less than  200 ratings and  books with
# less than 100 ratings.

# The first three cells import libraries  you may need and the data to
# use. The final  cell is for testing. Write all  your code in between
# those cells.

# get data files
#wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

#unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'


# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author', 'year', 'publisher'],
    usecols=['isbn', 'title', 'author', 'year', 'publisher'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str', 'year': 'str', 'publisher': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})


# To ensure  statistical significance,  remove from the  dataset users
# with less than 200 ratings and books with less than 100 ratings.

# Search the isbns of the books that were rated more thant 100 times
book_count = df_ratings['isbn'].value_counts()
good_books = book_count[book_count >= 100].index
#create a list of good books 
good_books = df_books.loc[df_books["isbn"].isin(good_books)]

#Count the numer of ratings for each user
user_count = df_ratings['user'].value_counts()
#create a list of good users
good_users = user_count[user_count >= 200].index

#remove users and books from dataset
df = df_ratings.loc[df_ratings['user'].isin(good_users)]
df = df.loc[df['isbn'].isin(good_books['isbn'])]

#add titles to df dataset to work directly with titles instead of isbn 
df = pd.merge(df, good_books, on='isbn').reset_index()

#create a table, neds to be a pivot_table to avoid index problems
rating_books_pivot = df.pivot_table(index='title', columns='user', values='rating').fillna(0)

nbrs = NearestNeighbors(metric='cosine')
nbrs.fit(rating_books_pivot)

#function to return recommended books - this will be tested
def get_recommends(title = "",nrec=6):

    distance, indice = nbrs.kneighbors(rating_books_pivot.loc[[title]],
                                       n_neighbors=n_rec,
                                       return_distance=True)

    indice = indice[0][1:]
    distance = distance[0][1:]
    
    titles = [rating_books_pivot.index[i] for i in indice]
    recommended = [list(z) for z in zip(titles, distance)][::-1]
        
    return [title, recommended]
    

recommended_books=get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))",nrec=10)

print(recommended_books)

#Use    the    cell    below    to    test    your    function.    The
#test_book_recommendation() function will inform you if you passed the
#challenge or need to keep trying.

# books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
# print(books)

# def test_book_recommendation():
#   test_pass = True
#   recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
#   if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
#     test_pass = False
#   recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
#   recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
#   for i in range(2): 
#     if recommends[1][i][0] not in recommended_books:
#       test_pass = False
#     if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
#       test_pass = False
#   if test_pass:
#     print("You passed the challenge!")
#   else:
#     print("You haven't passed yet. Keep trying!")

# test_book_recommendation()

import pickle
import streamlit as st
import numpy as np

model = pickle.load(open('model/model.pkl','rb'))
book_names = pickle.load(open('model/book_names.pkl','rb'))
final_rating = pickle.load(open('model/final_rating.pkl','rb'))
book_pivot = pickle.load(open('model/book_pivot.pkl','rb'))


def get_url(suggestion):
    
    book_name = [book_pivot.index[id] for id in suggestion[0]]

    ids_index = [np.where(final_rating['title']==name)[0][0] for name in book_name ]

    poster_url = [final_rating.iloc[i]['image_url'] for i in ids_index]
    
    return poster_url


def recommend_book(book_name):

    book_id = np.where(book_pivot.index.isin([book_name]))[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
    

    poster_url = get_url(suggestion)

    # Assuming book_pivot.index contains the list of book names corresponding to the suggestion indices
    books_list = [j for i in range(len(suggestion)) for j in book_pivot.index[suggestion[i]]]

    return books_list, poster_url





def main():
    st.set_page_config(
        page_title='Book Predictor',
        page_icon=":closed_book:",
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    cols_1, cols_2= st.columns([1,4]) 

    with cols_2:
        st.header("Book Recommender System Using Machine Learning")
    
    
    cols1, cols2, cols3 = st.columns([1,3,1])
    with cols2:
       
       selected_books = st.selectbox(
       "Type or select a book from the dropdown",
        book_names)
       books, url = recommend_book(selected_books)
       col4, col5, col6, col7, col8 = st.columns(5)
       if st.button('Show Recomendations'):
        with col4 :
            st.write(books[1])
            st.image(url[1])

        with col5 :
            st.write(books[2])
            st.image(url[2])

        with col6 :
            st.write(books[3])
            st.image(url[3])


        with col7 :
            st.write(books[4])
            st.image(url[4])

        with col8 :
            st.write(books[5])
            st.image(url[5])

    with open("style.css") as f:
       st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    

if __name__ == '__main__':
    main()
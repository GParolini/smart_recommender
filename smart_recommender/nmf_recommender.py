import pandas as pd
import numpy as np
import json
from sklearn.decomposition import NMF
import pickle
import random

#Loading the wide matrix
nmf_matrix = pd.read_csv("./smart_recommender/r_wide_matrix.csv", index_col =0)

#Loading the saved model
with open('./smart_recommender/nmf_model.pkl', 'rb') as file:
    nmf_model = pickle.load(file)

#Loading the movies dictionary
movies_dict_file = open("./smart_recommender/movies_dict.json", "r")
movies_dict = json.load(movies_dict_file)


def nmf_recommendations(form_data):
    new_user_input= {}
    for item in form_data.keys():
        new_user_input.update({item:float(form_data[item])})


    new_user = pd.DataFrame(new_user_input, index=['new_user'], columns=nmf_matrix.columns)
    Q = nmf_model.components_
    user_P = nmf_model.transform(new_user.fillna(0))
    user_R = pd.DataFrame(np.dot(user_P, Q), columns=nmf_matrix.columns, index=['new_user'])
    recommendations = user_R.drop(['2018', '2467', '3260', '5304', '3451'], axis=1)# remove the movies on which the user gave an evaluation
    my_rec = recommendations.sort_values(axis=1, by='new_user', ascending=False)
    rec_lst = my_rec.columns[0:5].to_list()
    my_rec_sel = my_rec.columns[0:50].to_list()
    rec_lst = random.sample(my_rec_sel, 5)


    user_rec_lst = []
    for key in rec_lst:
        movie_title = movies_dict[key]
        user_rec_lst.append(movie_title)

    return user_rec_lst

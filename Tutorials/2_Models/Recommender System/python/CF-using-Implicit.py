

from __future__ import print_function, absolute_import

import argparse
import os
import time
import logging

import pandas
import numpy as np
import scipy
from scipy.sparse import coo_matrix
import implicit

def read_data(path):
    df = pandas.read_table(path, sep='\t')
    return df

def create_sparse(dataframe):
    dataframe['values'] = 1
    # create sorted profile
    shopping_profile_id_u = list(np.sort(dataframe.shopping_profile_id.unique()))
    # create sorted brand
    brand_id_u = list(np.sort(dataframe.brand_id.unique()))
    data = dataframe['values'].astype(float).tolist()
    # create rows for sparse matrix
    row = dataframe.brand_id.astype('category', categories= brand_id_u).cat.codes
    # create columns for sparse matrix
    col = dataframe.shopping_profile_id.astype('category', categories= shopping_profile_id_u).cat.codes
    sparse = coo_matrix((data, (row, col)), shape=(len(brand_id_u), len(shopping_profile_id_u)))

    return sparse

def fit_model(sparse):
    model = implicit.als.AlternatingLeastSquares(factors=100)
    print("Satrting to fit model ...")
    model.fit(sparse)
    print("Fitting Done !")
    return model


def predict_model(brand_name, dataframe, models):
    unique_brands = np.sort(dataframe.brand_id.unique())
    try:
        b_id = dataframe.at[dataframe[dataframe['name'] == brand_name.index[0], 'brand_id']]
    except:
        print("brand does not exist")
    arr_val = np.where(unique_brands == b_id) #get the array position in the sparse matrix of the brand
    related = models.similar_items(arr_val[0][0]) #Feed the value into the similarity calculation of the model
    similar = []
    scores = []
    #Convert the sparse matrix positions back to brand names
    for i in related:
        value = int(i[0])
        scores.append(str(i[1]))
        similar_id = unique_brands[value]
        similar.append(dataframe.loc[dataframe['brand_id'] == similar_id, 'name'].tolist()[0])
    return zip(similar, scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates related brands from brands_filtered.txt "
                                     "dataset brands_filtered.txt",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='Path of the brands dataset', required=True)
    parser.add_argument('--output', type=str, default='similar-brands.tsv',
                        dest='outputfile', help='output file name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    df = read_data(args.inputfile)
    sparse_matrix = create_sparse(df)
    model = fit_model(sparse_matrix)
    with open(args.outputfile, "w") as o:
        unique = df['name'].unique()
        list1 = []
        for name in unique[26]:
            list1.append(predict_model(name, model))
            o.write("%s\t%s\t\n" % (list1[0], list1[1]))
    predict_model('Kate Spade', model)
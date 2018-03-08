'''
一、使用开源库implicit
Install: pip install implicit
https://github.com/benfred/implicit

二、训练数据集movielens
DownLoad: https://grouplens.org/datasets/movielens/

三、脚本运行
python3 movielens_recommand.py --input /tmp/ml-20m --output /tmp/output

--input：表示输入的数据集
--output: 表示输出文件
'''

from __future__ import print_function, absolute_import
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

import argparse
import os
import time
import logging

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender
from implicit.nearest_neighbours import BM25Recommender
from implicit.nearest_neighbours import TFIDFRecommender
from implicit.nearest_neighbours import bm25_weight

def read_data(path, min_rating=4.0):
    '''
    :param path: 读取数据的路径
    :param min_rating: 用户评价分数的阈值
    :return:
    '''

    rating_datas = pd.read_csv(os.path.join(path, "ratings.csv"))
    #过滤大于评价阈值的数据
    positive_datas = rating_datas[rating_datas.rating >= min_rating]

    movies_datas = pd.read_csv(os.path.join(path, "movies.csv"))
    m = coo_matrix((positive_datas['rating'].astype(np.float32),
                    (positive_datas['movieId'], positive_datas['userId'])))

    m.data = np.ones(len(m.data))

    return rating_datas, movies_datas, m

def calculate_similar_movies(input_path, output_filename,
                             model_name="als", min_rating=4.0):
    """
    :param input_path: 训练数据集的路径
    :param output_filename: 输出的文件名称
    :param model_name: 采用的模型
    :param min_rating: 过滤所需的阈值大小
    :return:
    """

    logging.debug("reading data from %s", input_path)
    start = time.time()
    rating_data, movies_data, m = read_data(input_path, min_rating=min_rating)
    logging.debug("reading data in %s", time.time() - start)

    if model_name == "als":
        model = AlternatingLeastSquares()

        logging.debug("weighting matrix by bm25_weight")
        m = bm25_weight(m, B=0.9) * 5

    elif model_name == "tfidf":
        model = TFIDFRecommender()

    elif model_name == "cosine":
        model = CosineRecommender()

    elif model_name == "bm25":
        model = BM25Recommender()

    else:
        raise NotImplementedError("TODU: model %s" % model_name)


    m = m.tocsr()
    logging.debug("Training model :%s" % model_name)
    start = time.time()
    model.fit(m)
    logging.debug("trained model '%s' in %s", model_name, time.time() - start)
    logging.debug("calculating top movies")

    user_count = rating_data.groupby("movieId").size()
    movie_lookup = dict((i, m) for i,m in
                        zip(movies_data['movieId'], movies_data['title']))
    to_generate = sorted(list(movies_data['movieId']), key=lambda x: -user_count.get(x, 0))

    with open(output_filename, "w") as o:
        for movieid in to_generate:
            if(m.indptr[movieid] == m.indptr[movieid + 1]):
                continue

            movie = movie_lookup[movieid]

            for other, score in model.similar_items(movieid, 11):
                o.write("%s\t%s\t%s\n" % (movie, movie_lookup[other], score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates related movies from the MovieLens 20M "
                                     "dataset (https://grouplens.org/datasets/movielens/20m/)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str,
                        dest='inputfile', help='Path of the unzipped ml-20m dataset', required=True)
    parser.add_argument('--output', type=str, default='similar-movies.tsv',
                        dest='outputfile', help='output file name')
    parser.add_argument('--model', type=str, default='als',
                        dest='model', help='model to calculate (als/bm25/tfidf/cosine)')
    parser.add_argument('--min_rating', type=float, default=4.0, dest='min_rating',
                        help='Minimum rating to assume that a rating is positive')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_movies(args.inputfile, args.outputfile,
                             model_name=args.model,
                             min_rating=args.min_rating)





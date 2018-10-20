import pickle
from tweet2vec import Tweet2Vec
from minibatch import Batchfeeder
from utils import encode_test_data
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
	with open('./model.pkl', 'rb') as myFile:
		params = pickle.load(myFile)

	bf = Batchfeeder('./data/test_stackoverflow.txt', 'test')
	print(bf.labeldict)
	print(len(bf.labeldict))
	t2v = Tweet2Vec(len(bf.chardict.keys())+1, len(bf.labeldict.keys())+1)
	t2v.setParams(params)

	print('Start Predicting')

	for tweets, targets in bf:
		tweets_test = tweets[0]
		tweets_mask = tweets[1]
		output = t2v.predict(tweets_test, tweets_mask)
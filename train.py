from minibatch import Batchfeeder
from tweet2vec import Tweet2Vec
import numpy as np
import pickle

EPOCH = 20

if __name__ == "__main__":
	bf = Batchfeeder('./data/train.txt')
	t2v = Tweet2Vec(len(bf.chardict.keys())+1, len(bf.labeldict.keys())+1)

	for i in range(EPOCH):
		for tweets, targets in bf:
			tweets_train = tweets[0]
			tweets_mask = tweets[1]
			cost = t2v.train(tweets_train, tweets_mask, targets)
			print(cost)

	params = t2v.getParams()
	
	with open('model.pkl', 'wb') as myFile:
		pickle.dump(params, myFile)

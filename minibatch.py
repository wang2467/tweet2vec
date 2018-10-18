from utils import encode_data, buildCharDictionary, buildLabelDictionary
import numpy as np
import theano

N_BATCH = 64

class Batchfeeder:
	def __init__(self, fname):
		self.load_data(fname)
		self.chardict, self.charcount = buildCharDictionary(self.tweets)
		self.labeldict, self.labelcount = buildLabelDictionary(self.targets)

		self.batch_size = N_BATCH
		self.curr_pos = 0

	def load_data(self, fname):
		with open(fname, 'r') as myFile:
			lines = myFile.read().splitlines()

		self.tweets, self.targets = [], []

		for i in lines:
			self.tweets.append(i.split('---')[1])
			self.targets.append(i.split('---')[0])

	def __next__(self):
		if self.curr_pos < len(self.tweets):
			tweets_res = self.tweets[self.curr_pos:min(self.curr_pos+self.batch_size, len(self.tweets))]
			targets_res = self.targets[self.curr_pos:min(self.curr_pos+self.batch_size, len(self.targets))]
			self.curr_pos += self.batch_size
			tweets_res = encode_data(tweets_res, self.chardict)
			targets_res = [self.labeldict[i] for i in targets_res]
			return tweets_res, np.array(targets_res)
		else:
			self.curr_pos = 0
			raise StopIteration

	def __iter__(self):
		return self

if __name__ == "__main__":
	b = Batchfeeder('./data/train.txt')
	for x, y in b:
		# print(x[0])
		# print(x[1])
		print(type(x[0]))
		print(len(x[0][3]))
		print(len(x[0][2]))
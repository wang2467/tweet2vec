from collections import OrderedDict
import numpy as np

def buildCharDictionary(tweets):
	charcount = OrderedDict()
	for tweet in tweets:
		for i in list(tweet):
			if i not in charcount:
				charcount[i] = 0
			charcount[i] += 1

	# encode the character in reverse order of its frequency
	charrank = sorted(charcount.items(), key=lambda x: x[1], reverse=True)

	chardict = OrderedDict()
	for idx, char in enumerate(charrank):
		chardict[char[0]] = idx + 1

	return chardict, charcount

def buildLabelDictionary(labels):
	labelcount = OrderedDict()
	for label in labels:
		if label not in labelcount:
			labelcount[label] = 0
		labelcount[label] += 1

	# encode the label in reverse order of its frequency
	labelrank = sorted(labelcount.items(), key=lambda x: x[1], reverse=True)

	labeldict = OrderedDict()
	for idx, label in enumerate(labelrank):
		labeldict[label[0]] = idx + 1

	return labeldict, labelcount

def encode_data(tweets, chardict):
	res = []
	max_length = 200
	for tweet in tweets:
		res.append([chardict[c] if chardict[c] <= 1000 else 0 for c in list(tweet)])

	x = np.zeros((len(tweets), max_length)).astype('int32')
	mask = np.zeros((len(tweets), max_length)).astype('int32')
	for idx, twts in enumerate(res):
		x[idx, :len(twts)] = twts
		mask[idx, :len(twts)] = 1

	return np.expand_dims(x, axis=2), mask

if __name__ == "__main__":
	tweets = ['I am a dog', 'Yao is a cat']
	chardict, charcount = buildCharDictionary(tweets)
	twts, mask = encode_data(tweets, chardict)
	print(twts)
	print(mask)
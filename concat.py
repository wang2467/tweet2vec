if __name__ == "__main__":
	with open('./data/label_StackOverflow.txt') as myFile:
		labels = myFile.read().splitlines()
	with open('./data/title_StackOverflow.txt') as myFile:
		titles = myFile.read().splitlines()

	res = [str(labels[i])+'---'+str(c) for i, c in enumerate(titles)]

	for i in range(0, len(res), 100):
		print(res[i])

	train_data_len = int(len(res) * 4 / 5)

	with open('./data/train_stackoverflow.txt', 'w') as myFile:
		for i in range(train_data_len):
			myFile.write('{}\n'.format(res[i]))

	with open('./data/test_stackoverflow.txt', 'w') as myFile:
		for i in range(train_data_len, len(res)):
			myFile.write('{}\n'.format(res[i]))



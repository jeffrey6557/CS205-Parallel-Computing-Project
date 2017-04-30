## split data into train+validation and test data
# fname = 'price_inputs_GS2016'
fname = 'second_level_inputs_GS2016'
f = open('../data/{}.csv'.format(fname),'r')
f_train = open('../data/{}_train.csv'.format(fname), 'w')
f_test = open('../data/{}_test.csv'.format(fname), 'w')

lines = f.readlines()
f.close()
n = len(lines)
n_train = n * 0.8
for x in xrange(n):
	if x < n_train:
		f_train.write(lines[x])
	else:
		f_test.write(lines[x])
f_test.close()
f_train.close()
fname = 'second_level_inputs_GS2016_train.csv'
fin = open('../data/{}'.format(fname), 'r')
fout = open('../data/expanded_{}'.format(fname), 'w')

lines = fin.readlines()
fin.close()
n = len(lines) #including the header
k = 10 #how many time points per row
nn = (n-1)//k
for x in xrange(nn):
	out_line = ''
	if x == 0: #header
		for i in xrange(k):
			out_line += ','.join(lines[0].strip().split(',')[1:])
			fout.write(lines[0].strip()+','+out_line)
		fout.write('\n')
	else:
		for i in xrange(k):
			out_line += ','.join(lines[x*k+i+1].strip().split(',')[1:])
			fout.write(lines[x*k+i+1].strip()+','+out_line)
		fout.write('\n')
fout.close()
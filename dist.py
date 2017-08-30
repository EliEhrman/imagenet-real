"""
The purpose of this module is to take the csv files created by daphna from activations of the vgg on the imagenet db
and to distribute them, by label into a file per class
"""
import os
import sys
import glob

files_dir = '../../data/imagenet/real'
labels_dir = '../../data/imglabels'

# c_files_str = 's*_data.csv'
c_files_str = 'validation*_data.csv'

for ifn, fn in enumerate(glob.glob(os.path.join(files_dir, c_files_str))):
	fnl = fn.replace('_data', '_labels')
	print fn, fnl
	fhl = open(fnl, "r")
	if fhl.mode == 'r':
		labels_raw = fhl.read()
		labels = labels_raw.split('\n')
	fhd = open(fn, "r")
	if fhd.mode == 'r':
		data_raw = fhd.read()
		data = data_raw.split('\n')
	for il, label in enumerate(labels):
		if label == '-1':
			label = '999'
		elif label == '':
			continue
		fnout = os.path.join(labels_dir, label)
		fhout = open(fnout, "a+")
		fhout.write(data[il])
		fhout.write('\n')
		fhout.close()
		print fnout
	print labels


print 'done'
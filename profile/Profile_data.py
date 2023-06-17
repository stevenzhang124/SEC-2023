from os import listdir
from os.path import isfile, join

def processing(fr1):
	layer_name = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']

	lines = fr1.readlines()
	# profile output size of each layer line 5 7 9 11 13
	# line_index = [5, 7, 9, 11, 13]
	# for index in line_index:
	# 	print(lines[index-1])

	layer_1_feed_time = 0
	layer_1_feed_time_count = 0

	layer_2_feed_time = 0
	layer_2_feed_time_count = 0

	layer_3_feed_time = 0
	layer_3_feed_time_count = 0

	layer_4_feed_time = 0
	layer_4_feed_time_count = 0

	layer_5_feed_time = 0
	layer_5_feed_time_count = 0



	last_5_layer_back_time = 0
	last_5_layer_back_time_count = 0

	last_4_layer_back_time = 0
	last_4_layer_back_time_count = 0

	last_3_layer_back_time = 0
	last_3_layer_back_time_count = 0

	last_2_layer_back_time = 0
	last_2_layer_back_time_count = 0

	last_1_layer_back_time = 0
	last_1_layer_back_time_count = 0


	full_local_batch = 0
	full_local_batch_count = 0

	for i, line in enumerate(lines):
		# profile feed forward execution time of each layer
		
		if "forward time Conv1" in line:
			layer_1_feed_time += float(line.split()[-1])
			layer_1_feed_time_count += 1

		
		if "forward time Conv2" in line:
			layer_2_feed_time += float(line.split()[-1])
			layer_2_feed_time_count += 1

		
		if "forward time FC1" in line:
			layer_3_feed_time += float(line.split()[-1])
			layer_3_feed_time_count += 1

		
		if "forward time FC2" in line:
			layer_4_feed_time += float(line.split()[-1])
			layer_4_feed_time_count += 1

		if "forward time FC3" in line:
			layer_5_feed_time += float(line.split()[-1])
			layer_5_feed_time_count += 1

		# profile backward execution time of each layer

		if "Backward time for layers conv1 conv2 fc1 fc2 fc3" in line:
			last_5_layer_back_time += float(line.split()[-1])
			last_5_layer_back_time_count += 1

		if "Backward time for layers conv2 fc1 fc2 fc3" in line:
			last_4_layer_back_time += float(line.split()[-1])
			last_4_layer_back_time_count += 1

		
		if "Backward time for layers fc1 fc2 fc3" in line:
			last_3_layer_back_time += float(line.split()[-1])
			last_3_layer_back_time_count += 1

		
		if "Backward time for layers fc2 fc3" in line:
			last_2_layer_back_time += float(line.split()[-1])
			last_2_layer_back_time_count += 1

		if "Backward time for layers fc3" in line:
			last_1_layer_back_time += float(line.split()[-1])
			last_1_layer_back_time_count += 1

		if "consumes" in line:
			full_local_batch += float(line.split()[-1])
			full_local_batch_count += 1

	#Feed forward time
	# print("Average feed forward time for layer_1 is ", 
	# 	layer_1_feed_time / layer_1_feed_time_count)
	# print("Average feed forward time for layer_2 is ", 
	# 	layer_2_feed_time / layer_2_feed_time_count)
	# print("Average feed forward time for layer_3 is ", 
	# 	layer_3_feed_time / layer_3_feed_time_count)
	# print("Average feed forward time for layer_4 is ", 
	# 	layer_4_feed_time / layer_4_feed_time_count)
	# print("Average feed forward time for layer_5 is ", 
	# 	layer_5_feed_time / layer_5_feed_time_count)

	print("Average feed time are {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(layer_1_feed_time / layer_1_feed_time_count, 
		layer_2_feed_time / layer_2_feed_time_count, layer_3_feed_time / layer_3_feed_time_count, layer_4_feed_time / layer_4_feed_time_count, layer_5_feed_time / layer_5_feed_time_count))

	#Backward time
	# print("Average back forward time for last 5 layers is ", 
	# 	last_5_layer_back_time / last_5_layer_back_time_count)
	# print("Average back forward time for last 4 layers is ", 
	# 	last_4_layer_back_time / last_4_layer_back_time_count)
	# print("Average back forward time for last 3 layers is ", 
	# 	last_3_layer_back_time / last_3_layer_back_time_count)
	# print("Average back forward time for last 2 layers is ", 
	# 	last_2_layer_back_time / last_2_layer_back_time_count)
	# print("Average back forward time for last 1 layers is ", 
	# 	last_1_layer_back_time / last_1_layer_back_time_count)

	print("Average back time are {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}".format(last_5_layer_back_time / last_5_layer_back_time_count, 
		last_4_layer_back_time / last_4_layer_back_time_count, last_3_layer_back_time / last_3_layer_back_time_count, last_2_layer_back_time / last_2_layer_back_time_count, last_1_layer_back_time / last_1_layer_back_time_count))

	#Full local time
	print("Average full local time for a batch is {:.5f}".format(full_local_batch / full_local_batch_count))

mypath = '.'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in files:
	(name, ext) = file.split('.')
	if ext == 'txt':
		print("Profile ", name)
		fr1 = open(file, 'r')
		processing(fr1)


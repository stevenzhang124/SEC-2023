# import math
# model profile
from cvx import convex
model = 'LeNet'
batchsize = 128
batch_num = 469
partition_point = 4
model_output = [442368, 131072, 61440, 43008, 5120] # output size of each layer
model_weights = [624, 9664, 123360, 40656, 3400] # size of each layer

# network profile
# device feed forward time for a batch
# [2*value for value in [0.08045, 0.05937, 0.01371, 0.00471, 0.00139]]
raspberry_feed = [2*value for value in [0.08045, 0.05937, 0.01371, 0.00471, 0.00139]]
nano_feed = [0.03554, 0.01035, 0.00455, 0.00076, 0.00024]
nx_feed = [0.06322, 0.01111, 0.01566, 0.00519, 0.00022]
agx_feed = [0.5*value for value in [0.07276, 0.05302, 0.00040, 0.00019, 0.00006]]
server_feed = [0.00101, 0.00057, 0.00012, 0.00008, 0.00004]
feed_time = {'raspberry':raspberry_feed, 'nano':nano_feed, 'nx':nx_feed, 'agx':agx_feed}

# device backward time for a batch
# from full model to the last layer, reverse direction
raspberry_back = [2*value for value in [0.33554, 0.09960, 0.01299, 0.03013, 0.00242]]
nano_back = [0.07518, 0.03178, 0.00504, 0.00136, 0.00060]
nx_back = [0.05580, 0.02125, 0.00399, 0.00299, 0.00068]
agx_back = [0.5*value for value in [0.02796, 0.01136, 0.00212, 0.00186, 0.00191]]
server_back = [0.00349, 0.00132, 0.00036, 0.00022, 0.00013]
back_time = {'raspberry':raspberry_back, 'nano':nano_back, 'nx':nx_back, 'agx':agx_back}

# full model local execution time for a batch
raspberry_local = 2.7518*2
nano_local = 0.28246
nx_local = 0.21598
agx_local = 0.23931*0.5
server_local = 0.00367
local_time = {'raspberry':raspberry_local, 'nano':nano_local, 'nx':nx_local, 'agx':agx_local}

#bandwidth profile
bandwidth_num = 5              #Mbps
bandwidth = 1024*1024/8*bandwidth_num  #MB/s

# device num
device_num = 20
raspberry_num = int(device_num*0.25)
nano_num = int(device_num*0.25)
nx_num = int(device_num*0.25)
agx_num = int(device_num*0.25)

def vanilla_FL():
	# all computation locally, edge devices equally share the bandwidth

	raspberry_time = raspberry_local*batch_num + device_num*sum(model_weights) / bandwidth * 2
	nano_time = nano_local*batch_num + device_num*sum(model_weights) / bandwidth * 2
	nx_time = nx_local*batch_num + device_num*sum(model_weights) / bandwidth * 2
	agx_time = agx_local*batch_num + device_num*sum(model_weights) / bandwidth * 2

	train_time = max(raspberry_time, nano_time, nx_time, agx_time)

	return train_time

def variant_FL():
	# adjust the bandwidth to reduce the gap
	A = []
	for i in range(raspberry_num):
		A.append(raspberry_local*batch_num)
	for i in range(nano_num):
		A.append(nano_local*batch_num)
	for i in range(nx_num):
		A.append(nx_local*batch_num)
	for i in range(agx_num):
		A.append(agx_local*batch_num)

	C = [sum(model_weights)*2]*device_num

	train_time = convex(A, C, bandwidth)

	return train_time


def heur_partition(device_name, raspberry_time):
	# decide partition point for heuristic method
	gaps = []
	for i in range(partition_point):
		execution_time = batch_num*(sum(feed_time[device_name][0:i+1]) + sum(server_feed[i+1:]) + server_back[i+1] + back_time[device_name][0] - back_time[device_name][i+1])
		gap = abs(execution_time - raspberry_time)
		gaps.append(gap)

	partition_layer = gaps.index(min(gaps)) + 1
	return partition_layer

def heuristic():
	# heuristic algorithm for 
	#weak_device_feed = min(raspberry_feed, nano_feed, nx_feed, agx_feed)
	
	# For weak device, assume partition point is 1, calculate the local and server execution
	raspberry_time = batch_num*(raspberry_feed[0] + sum(server_feed[1:]) + server_back[1] + raspberry_back[0] - raspberry_back[1])
	# now assume it is 4
	# weak_device_time = batch_num*(sum(raspberry_feed[0:3]) + sum(server_feed[3:]) + server_back[3] + raspberry_back[0] - raspberry_back[3])

	# decide partition point for nano
	partition_nano = heur_partition('nano', raspberry_time)
	comp_time_nano = batch_num*(sum(nano_feed[0:partition_nano]) + sum(server_feed[partition_nano:]) + server_back[partition_nano] + nano_back[0] - nano_back[partition_nano])

	# decide partition point for nx
	partition_nx = heur_partition('nx', raspberry_time)
	comp_time_nx = batch_num*(sum(nx_feed[0:partition_nx]) + sum(server_feed[partition_nx:]) + server_back[partition_nx] + nx_back[0] - nx_back[partition_nx])

	# decide partition point for agx 
	partition_agx = heur_partition('agx', raspberry_time)
	comp_time_agx = batch_num*(sum(agx_feed[0:partition_agx]) + sum(server_feed[partition_agx:]) + server_back[partition_agx] + agx_back[0] - agx_back[partition_agx])


	# then decide the bandwidth allocation, the bandwidth allocation should be inverse of the amount of data to send
	# data to send = activations * batch + partial weights
	send_data_raspberry = batch_num*model_output[0]*2 + model_weights[0]*2
	send_data_nano = batch_num*model_output[partition_nano-1]*2 + sum(model_weights[0:partition_nano])*2
	send_data_nx = batch_num*model_output[partition_nx-1]*2 + sum(model_weights[0:partition_nx])*2
	send_data_agx = batch_num*model_output[partition_agx-1]*2 + sum(model_weights[0:partition_agx])*2

	# bandwidth_raspberry = bandwidth * send_data_raspberry / (send_data_raspberry + send_data_nano + send_data_nx + send_data_agx)
	# bandwidth_nano = bandwidth * send_data_nano / (send_data_raspberry + send_data_nano + send_data_nx + send_data_agx)
	# bandwidth_nx = bandwidth * send_data_nx / (send_data_raspberry + send_data_nano + send_data_nx + send_data_agx)
	# bandwidth_agx = bandwidth * send_data_agx / (send_data_raspberry + send_data_nano + send_data_nx + send_data_agx)

	trans_time = (send_data_raspberry*raspberry_num + send_data_nano*nano_num + send_data_nx*nx_num + send_data_agx*agx_num) / bandwidth

	train_time = max(raspberry_time, comp_time_nano, comp_time_nx, comp_time_agx) + trans_time


	return train_time

def proposed_partition(device_name, B_i):
	
	execution_time = []
	A_i_list = []
	C_j_list = []
	for i in range(partition_point+1):
		if i == partition_point: # all local
			A_i_tmp = batch_num*local_time[device_name]
			C_j_tmp = sum(model_weights) * 2
		else:
			A_i_tmp = batch_num*(sum(feed_time[device_name][0:i+1]) + sum(server_feed[i+1:]) + server_back[i+1] + back_time[device_name][0] - back_time[device_name][i+1])
			C_j_tmp = batch_num*model_output[i]*2 + sum(model_weights[0:i+1])*2
		
		tmp_time = A_i_tmp + C_j_tmp / B_i

		A_i_list.append(A_i_tmp)
		C_j_list.append(C_j_tmp)
		execution_time.append(tmp_time)
	
	partition_layer = execution_time.index(min(execution_time))

	A_i = A_i_list[partition_layer]
	C_j = C_j_list[partition_layer]


	return A_i, C_j, partition_layer

def proposed_solution():

	# decide Xij first, then decide bandwidth allocation use convex optimization
	# initial bandwidth allocation 
	# (equal share first)
	B_i = bandwidth / device_num
	# for all kinds of devices calculate the partition point
	raspberry_result = proposed_partition('raspberry', B_i)
	nano_result = proposed_partition('nano', B_i)
	nx_result = proposed_partition('nx', B_i)
	agx_result = proposed_partition('agx', B_i)
	print("raspberry's partition layer is ", raspberry_result[2])
	print("nano's partition layer is ", nano_result[2])
	print("nx's partition layer is ", nx_result[2])
	print("agx's partition layer is ", agx_result[2])

	A = []
	C = []
	for i in range(raspberry_num):
		A.append(raspberry_result[0])
		C.append(raspberry_result[1])
	for i in range(nano_num):
		A.append(nano_result[0])
		C.append(nano_result[1])
	for i in range(nx_num):
		A.append(nx_result[0])
		C.append(nx_result[1])
	for i in range(agx_num):
		A.append(agx_result[0])
		C.append(agx_result[1])


	train_time = convex(A, C, bandwidth)

	return train_time

if __name__ == '__main__':
	print("Train time for vanilla_FL is ", vanilla_FL())
	print("Train time for heuristic is ", heuristic())
	print("Train time for variant_FL is ", variant_FL())
	print("Train time for proposed method is ", proposed_solution())

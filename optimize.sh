#!/bin/bash
# the following shell script manages the training and optimization of all of the models over this data. Testing results are saved in the directory under results and
# model states are saved under the specific model name in neural_network.

train ()
{
	python neural_network/TrainNetwork.py BaselineModel data/train/Raw data/train/balanced.csv 0.001 results/BaselineModel/errorlog.txt 25

}

test ()
{	# tests the performance of the previously trained model
	python neural_network/TestNetwork.py data/dev/Raw BaselineModel results/BaselineModel/hypothesis data/dev/table.csv results/BaselineModel
}

for ((i=0;i<50;i++));
	do
	train
	test
	echo "round "$i;
	done

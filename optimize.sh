#!/bin/bash
# the following shell script manages the training and optimization of all of the models over this data. Testing results are saved in the directory under results and
# model states are saved under the specific model name in neural_network.

train ()
{
	# trains the neural network where its name is specified by the first argument. A second argument specifies the number of runs to make.
	for((i=0;i<$2;++i));
		do
		python neural_network/TrainNetwork.py $1 data/train/Raw data/train/balanced.csv 0.001 results/$1/errorlog.txt 25 -a data/train/{Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
		done
}

test ()
{
		# tests the performance of the previously trained model first argument is the name of the model, the second specifies the testing batch
		python neural_network/TestNetwork.py data/$2/Raw $1 results/$1/hypothesis data/$2/table.csv results/$1 data/$2/{Dur,F{0,1,2,3,4,5}norm,Intensitynorm}
}


buildpraat ()
{
	# the praat model has a specific requirement that all of the data that currently lives in separate directories needs to now be combined into one.
	# this step can also be used to add or omit certain features in order to compare overall model performance.
	for batch in {train,test,dev};
		do
		rm -r data/$batch/Praat
		mkdir data/$batch/Praat
		python preprocessing/Praat.py data/$batch/table.csv data/$batch/Praat data/$batch/{Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
		done
}
trainpraat ()
{
	for ((i=0;i<$1;++i));
		do
		python neural_network/TrainNetwork.py PraatModel data/train/Praat data/train/balanced.csv 0.001 results/PraatModel/errorlog.txt 25

		python neural_network/TestNetwork.py data/dev/Praat  PraatModel results/PraatModel/hypothesis data/dev/table.csv results/PraatModel;
		done
	echo -e '\a'
}

reset ()
{
	# helper code that resets the training session for a model so that it can be retrained

	rm -r results/$1/previous_session
	mkdir results/$1/previous_session
	mv results/$1/{errorlog,hypothesis_anaylsis,statistics}.txt results/$1/previous_session

}

#reset Wav2vec+PraatModel

for((j=0;j<50;++j));
	do
	train Raw+PraatModel 1
	test Raw+PraatModel dev
	echo round $j;
	done
echo -e '\a'


#buildpraat
#reset PraatModel
#trainpraat 100

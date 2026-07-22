#! /bin/bash

# the following script manages the ablation study over Praat model. Individual features are removed sequentially and the model is trained over the resulting dataset.
# the perfomance of the model is logged at the end of each session, and the final results are used to analyze the importance of individual features.

buildpraat ()
{
	# the praat model has a specific requirement that all of the data that currently lives in separate directories needs to now be combined into one.
	# this step can also be used to add or omit certain features in order to compare overall model performance.
	for batch in {train,test,dev};
		do
		rm -r data/$batch/Praat
		mkdir data/$batch/Praat
		python ablation/Ablation.py data/$batch/table.csv data/$batch/Praat data/$batch/ data/$batch/$1;
		done
}
resetmodel ()
{
	# runs the reset training round to initialize the model
	python neural_network/TrainNetwork.py PraatModel data/train/Praat data/train/balanced.csv 99 results/PraatModel/errorlog.txt 1 -r
	python neural_network/TestNetwork.py data/dev/Praat  PraatModel results/PraatModel/hypothesis data/dev/table.csv results/PraatModel
}

trainpraat ()
{
	for ((i=0;i<$1;++i));
		do
		python neural_network/TrainNetwork.py PraatModel data/train/Praat data/train/balanced.csv 0.001 results/PraatModel/errorlog.txt 25

		python neural_network/TestNetwork.py data/dev/Praat  PraatModel results/PraatModel/hypothesis data/dev/table.csv results/PraatModel;
		done
}


summarize ()
{
	python results/txt2csv.py results/PraatModel/statistics.txt results/PraatModel/statistics.csv
	python ablation/Summarize.py ablation/summary.txt results/PraatModel/statistics.csv
}

resetresults ()
{
	# helper code that resets the training session for a model so that it can be retrained

	rm -r results/$1/previous_session
	mkdir results/$1/previous_session
	mv results/$1/{errorlog,hypothesis_anaylsis,statistics}.txt results/$1/previous_session

}

ablation ()
{
	# implements the study. each loop is a sequence of building the omitted dataset, training the model on it, testing the results, summarizing the results, and finally
	# reseting the model for the next round. The first study uses all features. The result of this study is deterministic with one caveat: when a model is reset, the parameters
	# are initiallized randomly (from the same distribution). Though an identical distribution is used, the actual value of the initial parameters cannot right now be predicted.
	# as of right now, this is assumed to have no serious consequence on the performance of each study.

	#first the study that uses all features
	buildpraat ""
	resetmodel
	trainpraat 100
	summarize
	resetresults "PraatModel"

	#next the ablation studies
	for feature in {Dur,F{0,1,2,3,4,5}norm,Intensitynorm};
		do
		buildpraat $feature
		resetmodel
		trainpraat 100
		summarize
		resetresults "PraatModel";
		done
}

ablation

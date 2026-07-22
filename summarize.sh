#! /bin/bash
# the following shell script implements the results summary process. first it translates all of the results generated to tables, then generates
# a summary of the statistics on those tables

summarize ()
{
	for model in {Raw+PraatModel,Wav2vec+PraatModel};
		do
		python results/txt2csv.py results/$model/statistics.txt results/$model/statistics.csv;
		done

	python results/Summarize.py results/summary.txt results/{Wav2vec+PraatModel,Raw+PraatModel}/statistics.csv
}

summarize

#! /bin/bash
# the following shell script implements the results summary process. first it translates all of the results generated to tables, then generates
# a summary of the statistics on those tables

summarize ()
{
	python results/txt2csv.py results/PraatModel/statistics.txt results/PraatModel/statistics.csv

	python results/Summarize.py results/summary.txt results/PraatModel/statistics.csv
}

summarize

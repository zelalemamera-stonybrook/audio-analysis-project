# this script syllabifies all of the aligned audio generated from the previous step of the pipeline.

# the input directory contains all of the data to be syllabified. First, textgrids from the previous alignment step are collected and syllable boundary information is obtained
# from the index dictionary generated alongside the alignment. Then the audio file concerned will be broken up into these time intervals, and saved at the target directory. 
# because each sound file is assumed to have at least two such boundaries, the resulting directory is expected to be at least twice as large as the input audio directory.

form: "input directory information"
	text: "textgrids", "path to textgrids"
	text: "syllableindex", "path to syllable index"
	text: "sounds", "sound file path"
	text: "target", "output path"
endform

#this will read first the textgrid directory, then the wavefile directory, and finally the table which contains the index information
#the actual object to be stored here are strings with the name of the files in the directory, the provided address is used to read the 
#real object from disk. The unicode sorting of the two string lists ensures that string i in textgrids is generated from waveform i (assuming 
#that the aligner also preserves this identification during the generation of each textgrid file. The actual names of the audio files are generated from the source table and 
#the audio dataset. 
 
gridlist$# = fileNames$#: textgrids$ + "*.TextGrid"
wavlist$# = fileNames$#: sounds$ + "*.wav"
Read Table from tab-separated file: syllableindex$

for i to size (wavlist$#)
	
	writeInfoLine: gridlist$# [i] + " " +  wavlist$# [i]

	row = Search column: "id", gridlist$# [i]
	class = Get value: row, "class"
	Read from file: textgrids$ + gridlist$# [i]
	Read from file: sounds$ + wavlist$# [i]
	iD$ = selected$ ("Sound")	

	left = 1
	for j to class
		selectObject ("Table syllable_index")
		right = Get value: row, string$ (j)
		right = right + 1
		
		selectObject ("TextGrid " + iD$)
		start_time = Get end time of interval: 2, left
		end_time = Get end time of interval: 2, right
		Insert interval tier: 3, "syllable tier"
		Insert boundary: 3, start_time
		Insert boundary: 3, end_time
		Set interval text: 3, 2, "current"
		
		writeInfoLine: "generating boundary " + string$ (j) + " " + string$ (start_time) + "s " + string$ (end_time) + "s"
		plusObject ("Sound " +  iD$)
		Extract intervals where: 3,"yes", "is equal to", "current"
		Save as WAV file: target$ + iD$ + "_" + string$ (j) +  ".wav" 
		Remove
		
		selectObject ("TextGrid "+  iD$)
		Remove tier: 3
		left = right
	endfor
	selectObject ("Table syllable_index")
endfor
Remove

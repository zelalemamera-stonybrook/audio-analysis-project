# this script adds a syllable tier to each textgrid and adds syllable boundaries. this is for syllable length 4

form: "input directory information"
	text: "textgrids", "path to textgrids"
	text: "syllableindex", "path to syllable index"
	text: "target", "output path"
endform

Create Strings as file list: "filelist", textgrids$ + "*.TextGrid"

number_of_files = Get number of strings

Read Table from tab-separated file: syllableindex$
writeInfoLine: " generating new tier for " + textgrids$ + " to " + target$
for i from 1 to number_of_files
	selectObject: "Strings filelist"
	filename$ = Get string: i
	selectObject: "Table syllable_index"
	row = Search column: "filename", filename$
	writeInfoLine: " adding boundaries in new tier for " + filename$
	syll1 = Get value: row, "syll1"
	syll2 = Get value: row, "syll2"
	syll3 = Get value: row, "syll3"

	path$ = textgrids$ + filename$
	Read from file: path$

	start_timestep = Get end time of interval: 2, 1
	syll1_timestamp = Get end time of interval: 2, syll1 + 1
	syll2_timestamp = Get end time of interval: 2, syll2 + 1
	syll3_timestamp = Get end time of interval: 2, syll3 + 1
	end = Get number of intervals: 2
	end_timestep = Get start time of interval: 2, end

	Insert interval tier: 3, "syllables"
	Insert boundary: 3, start_timestep
	Insert boundary: 3, syll1_timestamp
	Insert boundary: 3, syll2_timestamp
	Insert boundary: 3, syll3_timestamp
	Insert boundary: 3, end_timestep

	Set interval text: 3, 2, "1"
	Set interval text: 3, 3, "2"
	Set interval text: 3, 4, "3"
	Set interval text: 3, 5, "4"
	Save as text file: target$ + filename$
	Remove
endfor
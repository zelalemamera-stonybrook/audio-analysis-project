# the following script generates melspectrgram data for each audio file in the input directory and saves the result as a csv file in the target directory, with the same name.

form: "Information"
	text: "source_directory", "input_wav\"
	text: "target_directory", "pitch.csv"
endform

Create Strings as file list: "fileList", source_directory$ + "*.wav"

n = Get number of strings

for i from 1 to n
	wavname$ = Get string: i
	Read from file: source_directory$ + wavname$
	name$ = selected$ ("Sound", 1)
	writeInfoLine: source_directory$ + wavname$
	
	To MelSpectrogram: 0.015, 0.005, 100.0, 100.0, 0.0
	To Matrix: "yes"
	To TableOfReal
	To Table: "none"
	
	Save as comma-separated file: target_directory$ + name$ + ".csv"
	writeInfoLine: target_directory$ + name$ + ".csv"
	selectObject: "Sound " + name$
	Remove
	selectObject: "Strings fileList"
endfor
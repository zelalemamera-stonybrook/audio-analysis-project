# the following praat script uses the third tier of each textgrid to break down the audio into syllables. This is only for syllable length 3
form: "input directory paths"
	text: "textgrids", "text grid file path"
	text: "sounds", "sound file path"
	text: "target", "new audio files path"
endform

Create Strings as file list: "textlist", textgrids$ + "*.TextGrid"
Create Strings as file list: "wavlist", sounds$ + "*.wav"

n = Get number of strings
writeInfoLine: "syllabifying batch " + sounds$ + " to " + target$
for i from 1 to n
	selectObject: "Strings wavlist"
	wavname$ = Get string: i
	selectObject: "Strings textlist"
	textname$ = Get string: i

	Read from file: sounds$ + wavname$
	wav$ = selected$ ("Sound", 1)
	writeInfoLine: "reading sound file from " + textgrids$ + textname$
	Read from file: textgrids$ + textname$
	text$ = selected$ ("TextGrid", 1)
	
	writeInfoLine: " breaking up " + wav$ + " using " + text$
	
	plusObject: "Sound " + wav$
	Extract intervals where: 3,"yes", "is equal to", "1"
	Save as WAV file: target$ + wav$ + "_syll1.wav"
	Remove

	selectObject: "TextGrid " + text$
	plusObject: "Sound " + wav$

	Extract intervals where: 3,"yes", "is equal to", "2"
	Save as WAV file: target$ + wav$ + "_syll2.wav"
	Remove
	
	selectObject: "TextGrid " + text$
	plusObject: "Sound " + wav$

	Extract intervals where: 3,"yes", "is equal to", "3"
	Save as WAV file: target$ + wav$ + "_syll3.wav"
	Remove

	selectObject: "Sound " + wav$
	Remove
	selectObject: "TextGrid " + text$
	Remove
endfor
# the following shell script automates and implements the entire data pipeline.
# there are environmental variables and applications that are assumed to exist on the machine before running this script
# PRAAT should be set to the application location to praat. conda should be installed and have an aligner environment available for mfa. Please look at the set up
# instructions for mfa for how to add this specfically from https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html


align ()
{
	echo "aligning data..." ;
	python "preprocessing/PrepareAlignments.py" ;
	mkdir "data/alignment/source" ;
	cp -r "data/alignment/text/." "data/alignment/source" ;
	cp -r "data/alignment/audio/." "data/alignment/source" ;
	rm -r "data/alignment/textgrid" ;
	mkdir "data/alignment/textgrid" ;
	echo "running aligner..." ;
	conda run -n aligner mfa align --clean "data/alignment/source" "data/arabic_mfa.dict" "english_mfa" "data/alignment/textgrid" ;
	rm -r "data/alignment/source" ;
}

syllabify ()
{
	echo "syllabifying data..." ;
	python "preprocessing/GenerateIndex.py" ;

	rm -r "data/alignment/syllabified_audio" ;
	mkdir "data/alignment/syllabified_audio" ;

	$PRAAT --run "preprocessing/Syllabify.praat" "../data/alignment/textgrid/" "../data/syllable_index.txt" "../data/alignment/audio/" "../data/alignment/syllabified_audio/" ;
}

align
syllabify


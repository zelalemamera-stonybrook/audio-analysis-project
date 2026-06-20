#!/bin/zsh
# the following program first generates the syllable indices for the data and then subsequently makes several calls to the praat program to extract syllable information and break audio files by syllable boundaries

python preprocessing/GenerateIndex.py

rm -r data/data_2/train/alignment/sylltextgrid
mkdir data/data_2/train/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_2.praat ../data/data_2/train/alignment/textgrid/ ../data/data_2/syllable_index.txt ../data/data_2/train/alignment/sylltextgrid/

rm -r data/data_2/test/alignment/sylltextgrid
mkdir data/data_2/test/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_2.praat ../data/data_2/test/alignment/textgrid/ ../data/data_2/syllable_index.txt ../data/data_2/test/alignment/sylltextgrid/

rm -r data/data_2/dev/alignment/sylltextgrid
mkdir data/data_2/dev/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_2.praat ../data/data_2/dev/alignment/textgrid/ ../data/data_2/syllable_index.txt ../data/data_2/dev/alignment/sylltextgrid/

rm -r data/data_3/train/alignment/sylltextgrid
mkdir data/data_3/train/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_3.praat ../data/data_3/train/alignment/textgrid/ ../data/data_3/syllable_index.txt ../data/data_3/train/alignment/sylltextgrid/

rm -r data/data_3/test/alignment/sylltextgrid
mkdir data/data_3/test/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_3.praat ../data/data_3/test/alignment/textgrid/ ../data/data_3/syllable_index.txt ../data/data_3/test/alignment/sylltextgrid/

rm -r data/data_3/dev/alignment/sylltextgrid
mkdir data/data_3/dev/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_3.praat ../data/data_3/dev/alignment/textgrid/ ../data/data_3/syllable_index.txt ../data/data_3/dev/alignment/sylltextgrid/

rm -r data/data_4/train/alignment/sylltextgrid
mkdir data/data_4/train/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_4.praat ../data/data_4/train/alignment/textgrid/ ../data/data_4/syllable_index.txt ../data/data_4/train/alignment/sylltextgrid/

rm -r data/data_4/test/alignment/sylltextgrid
mkdir data/data_4/test/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_4.praat ../data/data_4/test/alignment/textgrid/ ../data/data_4/syllable_index.txt ../data/data_4/test/alignment/sylltextgrid/

rm -r data/data_4/dev/alignment/sylltextgrid
mkdir data/data_4/dev/alignment/sylltextgrid
$PRAAT --run preprocessing/syllabify_textgrids_4.praat ../data/data_4/dev/alignment/textgrid/ ../data/data_4/syllable_index.txt ../data/data_4/dev/alignment/sylltextgrid/

rm -r data/data_2/train/alignment/syllaudio
mkdir data/data_2/train/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_2.praat ../data/data_2/train/alignment/sylltextgrid/ ../data/data_2/train/alignment/audio/ ../data/data_2/train/alignment/syllaudio/

rm -r data/data_2/test/alignment/syllaudio
mkdir data/data_2/test/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_2.praat ../data/data_2/test/alignment/sylltextgrid/ ../data/data_2/test/alignment/audio/ ../data/data_2/test/alignment/syllaudio/

rm -r data/data_2/dev/alignment/syllaudio
mkdir data/data_2/dev/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_2.praat ../data/data_2/dev/alignment/sylltextgrid/ ../data/data_2/dev/alignment/audio/ ../data/data_2/dev/alignment/syllaudio/

rm -r data/data_3/train/alignment/syllaudio
mkdir data/data_3/train/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_3.praat ../data/data_3/train/alignment/sylltextgrid/ ../data/data_3/train/alignment/audio/ ../data/data_3/train/alignment/syllaudio/

rm -r data/data_3/test/alignment/syllaudio
mkdir data/data_3/test/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_3.praat ../data/data_3/test/alignment/sylltextgrid/ ../data/data_3/test/alignment/audio/ ../data/data_3/test/alignment/syllaudio/

rm -r data/data_3/dev/alignment/syllaudio
mkdir data/data_3/dev/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_3.praat ../data/data_3/dev/alignment/sylltextgrid/ ../data/data_3/dev/alignment/audio/ ../data/data_3/dev/alignment/syllaudio/

rm -r data/data_4/train/alignment/syllaudio
mkdir data/data_4/train/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_4.praat ../data/data_4/train/alignment/sylltextgrid/ ../data/data_4/train/alignment/audio/ ../data/data_4/train/alignment/syllaudio/

rm -r data/data_4/test/alignment/syllaudio
mkdir data/data_4/test/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_4.praat ../data/data_4/test/alignment/sylltextgrid/ ../data/data_4/test/alignment/audio/ ../data/data_4/test/alignment/syllaudio/

rm -r data/data_4/dev/alignment/syllaudio
mkdir data/data_4/dev/alignment/syllaudio
$PRAAT --run preprocessing/syllabify_audio_4.praat ../data/data_4/dev/alignment/sylltextgrid/ ../data/data_4/dev/alignment/audio/ ../data/data_4/dev/alignment/syllaudio/



















#!/bin/zsh
# if the user wants to generate features from the audio data, they should run this script. It calls several scripts written by Feipeng Shao (cited in
# the documentation) to generate duration, pitch, intensity, and formant

rm -r data/data_2/train/features/formant
mkdir data/data_2/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/train/alignment/audio/ ../data/data_2/train/alignment/sylltextgrid/ 3 ../data/data_2/train/features/formant/dur_formant.txt

rm -r data/data_2/test/features/formant
mkdir data/data_2/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/test/alignment/audio/ ../data/data_2/test/alignment/sylltextgrid/ 3 ../data/data_2/test/features/formant/dur_formant.txt

rm -r data/data_2/dev/features/formant
mkdir data/data_2/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/dev/alignment/audio/ ../data/data_2/dev/alignment/sylltextgrid/ 3 ../data/data_2/dev/features/formant/dur_formant.txt

rm -r data/data_3/train/features/formant
mkdir data/data_3/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/train/alignment/audio/ ../data/data_3/train/alignment/sylltextgrid/ 3 ../data/data_3/train/features/formant/dur_formant.txt

rm -r data/data_3/test/features/formant
mkdir data/data_3/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/test/alignment/audio/ ../data/data_3/test/alignment/sylltextgrid/ 3 ../data/data_3/test/features/formant/dur_formant.txt

rm -r data/data_3/dev/features/formant
mkdir data/data_3/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/dev/alignment/audio/ ../data/data_3/dev/alignment/sylltextgrid/ 3 ../data/data_3/dev/features/formant/dur_formant.txt

rm -r data/data_4/train/features/formant
mkdir data/data_4/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/train/alignment/audio/ ../data/data_4/train/alignment/sylltextgrid/ 3 ../data/data_4/train/features/formant/dur_formant.txt

rm -r data/data_4/test/features/formant
mkdir data/data_4/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/test/alignment/audio/ ../data/data_4/test/alignment/sylltextgrid/ 3 ../data/data_4/test/features/formant/dur_formant.txt

rm -r data/data_4/dev/features/formant
mkdir data/data_4/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/dev/alignment/audio/ ../data/data_4/dev/alignment/sylltextgrid/ 3 ../data/data_4/dev/features/formant/dur_formant.txt

rm -r data/data_2/train/features/pitch
mkdir data/data_2/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/train/alignment/audio/ ../data/data_2/train/alignment/sylltextgrid/ 3 ../data/data_2/train/features/pitch/dur_pitch.txt

rm -r data/data_2/test/features/pitch
mkdir data/data_2/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/test/alignment/audio/ ../data/data_2/test/alignment/sylltextgrid/ 3 ../data/data_2/test/features/pitch/dur_pitch.txt

rm -r data/data_2/dev/features/pitch
mkdir data/data_2/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/dev/alignment/audio/ ../data/data_2/dev/alignment/sylltextgrid/ 3 ../data/data_2/dev/features/pitch/dur_pitch.txt

rm -r data/data_3/train/features/pitch
mkdir data/data_3/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/train/alignment/audio/ ../data/data_3/train/alignment/sylltextgrid/ 3 ../data/data_3/train/features/pitch/dur_pitch.txt

rm -r data/data_3/test/features/pitch
mkdir data/data_3/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/test/alignment/audio/ ../data/data_3/test/alignment/sylltextgrid/ 3 ../data/data_3/test/features/pitch/dur_pitch.txt

rm -r data/data_3/dev/features/pitch
mkdir data/data_3/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/dev/alignment/audio/ ../data/data_3/dev/alignment/sylltextgrid/ 3 ../data/data_3/dev/features/pitch/dur_pitch.txt

rm -r data/data_4/train/features/pitch
mkdir data/data_4/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/train/alignment/audio/ ../data/data_4/train/alignment/sylltextgrid/ 3 ../data/data_4/train/features/pitch/dur_pitch.txt

rm -r data/data_4/test/features/pitch
mkdir data/data_4/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/test/alignment/audio/ ../data/data_4/test/alignment/sylltextgrid/ 3 ../data/data_4/test/features/pitch/dur_pitch.txt

rm -r data/data_4/dev/features/pitch
mkdir data/data_4/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/dev/alignment/audio/ ../data/data_4/dev/alignment/sylltextgrid/ 3 ../data/data_4/dev/features/pitch/dur_pitch.txt

rm -r data/data_2/train/features/intensity
mkdir data/data_2/train/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_2/train/alignment/audio/ ../data/data_2/train/alignment/sylltextgrid/ 3 ../data/data_2/train/features/intensity/dur_intensity.txt

rm -r data/data_2/test/features/intensity
mkdir data/data_2/test/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_2/test/alignment/audio/ ../data/data_2/test/alignment/sylltextgrid/ 3 ../data/data_2/test/features/intensity/dur_intensity.txt

rm -r data/data_2/dev/features/intensity
mkdir data/data_2/dev/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_2/dev/alignment/audio/ ../data/data_2/dev/alignment/sylltextgrid/ 3 ../data/data_2/dev/features/intensity/dur_intensity.txt

rm -r data/data_3/train/features/intensity
mkdir data/data_3/train/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_3/train/alignment/audio/ ../data/data_3/train/alignment/sylltextgrid/ 3 ../data/data_3/train/features/intensity/dur_intensity.txt

rm -r data/data_3/test/features/intensity
mkdir data/data_3/test/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_3/test/alignment/audio/ ../data/data_3/test/alignment/sylltextgrid/ 3 ../data/data_3/test/features/intensity/dur_intensity.txt

rm -r data/data_3/dev/features/intensity
mkdir data/data_3/dev/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_3/dev/alignment/audio/ ../data/data_3/dev/alignment/sylltextgrid/ 3 ../data/data_3/dev/features/intensity/dur_intensity.txt

rm -r data/data_4/train/features/intensity
mkdir data/data_4/train/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_4/train/alignment/audio/ ../data/data_4/train/alignment/sylltextgrid/ 3 ../data/data_4/train/features/intensity/dur_intensity.txt

rm -r data/data_4/test/features/intensity
mkdir data/data_4/test/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_4/test/alignment/audio/ ../data/data_4/test/alignment/sylltextgrid/ 3 ../data/data_4/test/features/intensity/dur_intensity.txt

rm -r data/data_4/dev/features/intensity
mkdir data/data_4/dev/features/intensity
$PRAAT --run preprocessing/dur_intensity.praat ../data/data_4/dev/alignment/audio/ ../data/data_4/dev/alignment/sylltextgrid/ 3 ../data/data_4/dev/features/intensity/dur_intensity.txt

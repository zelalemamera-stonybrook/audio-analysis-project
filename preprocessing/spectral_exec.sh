#!/bin/zsh
# if the user wants to generate features from the audio data, they should run this script. It calls several scripts written by Feipeng Shao (cited in
# the documentation) to generate duration, pitch, intensity, and formant

rm -r data/data_2/train/features/formant
mkdir data/data_2/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/train/alignment/syllaudio/ ../data/data_2/train/features/formant/

rm -r data/data_2/test/features/formant
mkdir data/data_2/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/test/alignment/syllaudio/ ../data/data_2/test/features/formant/

rm -r data/data_2/dev/features/formant
mkdir data/data_2/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_2/dev/alignment/syllaudio/ ../data/data_2/dev/features/formant/

rm -r data/data_3/train/features/formant
mkdir data/data_3/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/train/alignment/syllaudio/ ../data/data_3/train/features/formant/

rm -r data/data_3/test/features/formant
mkdir data/data_3/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/test/alignment/syllaudio/ ../data/data_3/test/features/formant/

rm -r data/data_3/dev/features/formant
mkdir data/data_3/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_3/dev/alignment/syllaudio/ ../data/data_3/dev/features/formant/

rm -r data/data_4/train/features/formant
mkdir data/data_4/train/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/train/alignment/syllaudio/ ../data/data_4/train/features/formant/

rm -r data/data_4/test/features/formant
mkdir data/data_4/test/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/test/alignment/syllaudio/ ../data/data_4/test/features/formant/

rm -r data/data_4/dev/features/formant
mkdir data/data_4/dev/features/formant
$PRAAT --run preprocessing/dur_formant.praat ../data/data_4/dev/alignment/syllaudio/ ../data/data_4/dev/features/formant/

rm -r data/data_2/train/features/pitch
mkdir data/data_2/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/train/alignment/syllaudio/ ../data/data_2/train/features/pitch/

rm -r data/data_2/test/features/pitch
mkdir data/data_2/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/test/alignment/syllaudio/ ../data/data_2/test/features/pitch/

rm -r data/data_2/dev/features/pitch
mkdir data/data_2/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_2/dev/alignment/syllaudio/ ../data/data_2/dev/features/pitch/

rm -r data/data_3/train/features/pitch
mkdir data/data_3/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/train/alignment/syllaudio/ ../data/data_3/train/features/pitch/

rm -r data/data_3/test/features/pitch
mkdir data/data_3/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/test/alignment/syllaudio/ ../data/data_3/test/features/pitch/

rm -r data/data_3/dev/features/pitch
mkdir data/data_3/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_3/dev/alignment/syllaudio/ ../data/data_3/dev/features/pitch/

rm -r data/data_4/train/features/pitch
mkdir data/data_4/train/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/train/alignment/syllaudio/ ../data/data_4/train/features/pitch/

rm -r data/data_4/test/features/pitch
mkdir data/data_4/test/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/test/alignment/syllaudio/ ../data/data_4/test/features/pitch/

rm -r data/data_4/dev/features/pitch
mkdir data/data_4/dev/features/pitch
$PRAAT --run preprocessing/dur_pitch.praat ../data/data_4/dev/alignment/syllaudio/ ../data/data_4/dev/features/pitch/

rm -r data/data_2/train/features/mel
mkdir data/data_2/train/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_2/train/alignment/syllaudio/ ../data/data_2/train/features/mel/

rm -r data/data_2/test/features/mel
mkdir data/data_2/test/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_2/test/alignment/syllaudio/ ../data/data_2/test/features/mel/

rm -r data/data_2/dev/features/mel
mkdir data/data_2/dev/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_2/dev/alignment/syllaudio/ ../data/data_2/dev/features/mel/

rm -r data/data_3/train/features/mel
mkdir data/data_3/train/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_3/train/alignment/syllaudio/ ../data/data_3/train/features/mel/

rm -r data/data_3/test/features/mel
mkdir data/data_3/test/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_3/test/alignment/syllaudio/ ../data/data_3/test/features/mel/

rm -r data/data_3/dev/features/mel
mkdir data/data_3/dev/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_3/dev/alignment/syllaudio/ ../data/data_3/dev/features/mel/

rm -r data/data_4/train/features/mel
mkdir data/data_4/train/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_4/train/alignment/syllaudio/ ../data/data_4/train/features/mel/

rm -r data/data_4/test/features/mel
mkdir data/data_4/test/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_4/test/alignment/syllaudio/ ../data/data_4/test/features/mel/

rm -r data/data_4/dev/features/mel
mkdir data/data_4/dev/features/mel
$PRAAT --run preprocessing/mel.praat ../data/data_4/dev/alignment/syllaudio/ ../data/data_4/dev/features/mel/

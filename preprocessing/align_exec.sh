#!/bin/zsh

# this is a shell script that manually populates and writes all of the aligned textgrid files to the directory, it is assumed that you already have the montral forced aligner envirnoment up and running while you call this. 

mkdir data/data_2/train/alignment/source
cp -r data/data_2/train/alignment/text/. data/data_2/train/alignment/source
cp -r data/data_2/train/alignment/audio/. data/data_2/train/alignment/source
rm -r data/data_2/train/alignment/textgrid
mkdir data/data_2/train/alignment/textgrid
mfa align --clean data/data_2/train/alignment/source data/data_2/arabic_mfa.dict english_mfa data/data_2/train/alignment/textgrid
rm -r data/data_2/train/alignment/source

mkdir data/data_2/test/alignment/source
cp -r data/data_2/test/alignment/text/. data/data_2/test/alignment/source
cp -r data/data_2/test/alignment/audio/. data/data_2/test/alignment/source
rm -r data/data_2/test/alignment/textgrid
mkdir data/data_2/test/alignment/textgrid
mfa align --clean data/data_2/test/alignment/source data/data_2/arabic_mfa.dict english_mfa data/data_2/test/alignment/textgrid
rm -r data/data_2/test/alignment/source

mkdir data/data_2/dev/alignment/source
cp -r data/data_2/dev/alignment/text/. data/data_2/dev/alignment/source
cp -r data/data_2/dev/alignment/audio/. data/data_2/dev/alignment/source
rm -r data/data_2/dev/alignment/textgrid
mkdir data/data_2/dev/alignment/textgrid
mfa align --clean data/data_2/dev/alignment/source data/data_2/arabic_mfa.dict english_mfa data/data_2/dev/alignment/textgrid
rm -r data/data_2/dev/alignment/source

mkdir data/data_3/train/alignment/source
cp -r data/data_3/train/alignment/text/. data/data_3/train/alignment/source
cp -r data/data_3/train/alignment/audio/. data/data_3/train/alignment/source
rm -r data/data_3/train/alignment/textgrid
mkdir data/data_3/train/alignment/textgrid
mfa align --clean data/data_3/train/alignment/source data/data_3/arabic_mfa.dict english_mfa data/data_3/train/alignment/textgrid
rm -r data/data_3/train/alignment/source

mkdir data/data_3/test/alignment/source
cp -r data/data_3/test/alignment/text/. data/data_3/test/alignment/source
cp -r data/data_3/test/alignment/audio/. data/data_3/test/alignment/source
rm -r data/data_3/test/alignment/textgrid
mkdir data/data_3/test/alignment/textgrid
mfa align --clean data/data_3/test/alignment/source data/data_3/arabic_mfa.dict english_mfa data/data_3/test/alignment/textgrid
rm -r data/data_3/test/alignment/source

mkdir data/data_3/dev/alignment/source
cp -r data/data_3/dev/alignment/text/. data/data_3/dev/alignment/source
cp -r data/data_3/dev/alignment/audio/. data/data_3/dev/alignment/source
rm -r data/data_3/dev/alignment/textgrid
mkdir data/data_3/dev/alignment/textgrid
mfa align --clean data/data_3/dev/alignment/source data/data_3/arabic_mfa.dict english_mfa data/data_3/dev/alignment/textgrid
rm -r data/data_3/dev/alignment/source

mkdir data/data_4/train/alignment/source
cp -r data/data_4/train/alignment/text/. data/data_4/train/alignment/source
cp -r data/data_4/train/alignment/audio/. data/data_4/train/alignment/source
rm -r data/data_4/train/alignment/textgrid
mkdir data/data_4/train/alignment/textgrid
mfa align --clean data/data_4/train/alignment/source data/data_4/arabic_mfa.dict english_mfa data/data_4/train/alignment/textgrid
rm -r data/data_4/train/alignment/source

mkdir data/data_4/test/alignment/source
cp -r data/data_4/test/alignment/text/. data/data_4/test/alignment/source
cp -r data/data_4/test/alignment/audio/. data/data_4/test/alignment/source
rm -r data/data_4/test/alignment/textgrid
mkdir data/data_4/test/alignment/textgrid
mfa align --clean data/data_4/test/alignment/source data/data_4/arabic_mfa.dict english_mfa data/data_4/test/alignment/textgrid
rm -r data/data_4/test/alignment/source

mkdir data/data_4/dev/alignment/source
cp -r data/data_4/dev/alignment/text/. data/data_4/dev/alignment/source
cp -r data/data_4/dev/alignment/audio/. data/data_4/dev/alignment/source
rm -r data/data_4/dev/alignment/textgrid
mkdir data/data_4/dev/alignment/textgrid
mfa align --clean data/data_4/dev/alignment/source data/data_4/arabic_mfa.dict english_mfa data/data_4/dev/alignment/textgrid
rm -r data/data_4/dev/alignment/source








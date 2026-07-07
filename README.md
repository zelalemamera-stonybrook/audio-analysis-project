This repository holds all of the relevant files for our stress analysis project. You may clone this repository
to replicate all of the steps that were involved during the development of the model. The pipeline begins with raw audio and a stress table.
These were obtained from our data source and were modified slightly for ease and convenience. In particular, we chose to remove one syllable
data from the original dataset, because we make the assumption that monosyllabic words in this language (Jordanian Arabic) are all stressed
and hence do not contribute additional information. Additionally, we generate syllable information from the table which we use
to syllabify the raw audio. 

All code is assumed to be running from the root of this directory unless any subprocesses are called within subdirectories, which will always be done automatically and will not require any user input. 

The entire pipeline has been implemented as a shell script as pipline.sh

In order to sucessfully replicate the steps of this pipeline, you should install some necessary programs. One is Python which is used for most of the data processing (via pandas tables) and for the machine
learning steps (via Pytorch). You should also install torchaudio and torchcodec. Two additional audio analysis software are used: Praat and MFA(montreal forced aligner).
 
MFA is the aligner. We use it to map the Arabic text provided in the table to raw audio input. Aligning raw Arabic audio is a non trivial task since there is no pretrained model to do this. We generate a custom dictionary and
repurpose an english acoustic model. You may be able to obtain forced alignments in another way, for example a neural network, we only make the assumption that the method you use has the ability to generate textgrids, 
which are the objects that Praat operates on. 

We use Praat for syllabifiying audio, and for generating high quality acoustic and spectral features.

Once all of the above requirements are met, you may perform the pipeline via

bash pipeline.sh 

or simply
./pipeline.sh

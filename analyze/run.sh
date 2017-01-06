#!/bin/bash

python3 analyze.py \
    studydata/ \
    ../filedict.pickle \
    ../simil.npy \
    ../titles.pickle \
    studydata/postsurvey.tsv

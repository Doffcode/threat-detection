#!/bin/bash
cd /home/shiva/Desktop/projects/akriti/weapon-detection/work/downloads/openimages
M=$(cat mids.txt)
for u in v6/oidv6-train-annotations-bbox.csv v5/validation-annotations-bbox.csv v5/test-annotations-bbox.csv; do
  f=$(basename $u); curl -sS https://storage.googleapis.com/openimages/$u | grep -E ",($M)," > filt_$f; echo done $f $(wc -l < filt_$f)
done
for u in v7/oidv7-train-annotations-human-imagelabels.csv v7/oidv7-val-annotations-human-imagelabels.csv v7/oidv7-test-annotations-human-imagelabels.csv; do
  f=$(basename $u); curl -sS https://storage.googleapis.com/openimages/$u | grep -E ",($M),1" > filt_$f; echo done $f $(wc -l < filt_$f)
done
echo ALLDONE

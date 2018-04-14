#!/bin/bash

OUTPUT_FOLDER=$1"_ARFF/"

if [ ! -d "$OUTPUT_FOLDER" ]; then
  mkdir $OUTPUT_FOLDER
fi

for filename in $1"/"*".wav"; do
    filename_wo_ext=$(echo $filename | cut -d'/' -f6 | cut -d'.' -f1)
    ./openSMILE-2.2rc1/bin/linux_x64_standalone_static/SMILExtract -C ./openSMILE-2.2rc1/config/IS09_emotion.conf -I $filename -O $OUTPUT_FOLDER$filename_wo_ext.arff
done

# Automatic emotion recognition in speech

## Introduction

The goal of this project was to reproduce the results in the papers [4] and [5].
Using corpuses AIBO [1] and EMO-DB [2] as training and test datasets.
Using OpenSmile[3] tool to extract acoustic features.

## To execute the classifier:
> Should need to download corpuses and OpenSmile (then run it on all corpuses files) and put everything in the right folders.

```
python3 emotions_classifier.py NB_CLASSES [-show-plot]
```

- NB_CLASSES: [2, 5] 
  * Gives what type of classification to do: 2 or 5 classes.

- -show-plot: (optional)
  * Allows to show the confusion matrix plot.

## Credits

[1] http://www5.cs.fau.de/de/mitarbeiter/steidl-stefan/fau-aibo-emotion-corpus/

[2] http://emodb.bilderbar.info/docu/

[3] http://audeering.com/technology/opensmile/

[4] Björn Schuller and Stefan Steidl and Anton Batliner, “The INTERSPEECH 2009 Emotion Challenge,”
In proc. of Interspeech, Brighton, U.K., 2009.

[5] George Trigeorgis et al., “Adieu Features? End-to-end speech emotion recognition using a Deep Convo-
lutional Recurrent Network,” In proc. of ICASSP, Shanghai, China, 2016.

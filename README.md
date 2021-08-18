# Seismic classifier
 Practical earthquake event classifier, which accurately classifies earthquake events into natural earthquakes, explosion, collapse, etc.

# Source code description:
-------------------------------------------------------------------------
For specific function descriptions, please follow the detailed comments in the code.

independent_test.py
Independent test file
train.py
Training file
model.py
Function definitions necessary for training the model
model_definition.py
Model definition file

----------------------------------------------------------------------------

# Sample data description
-------------------------------
The sample data contains the waveform data and markers of 50989 seismic events, of which 36,625 are real seismic events, and the other 14,364 seismic events are derived from data enhancement.

The data shape is: 50989 * 18006

18000 data bits before the data segment
Data segment 18001 flag bit

Flag description:
18001 is the data type, corresponding to natural earthquake 0, blasting 1, and collapse 2 respectively.

Data bit description:
Data bits are the first 18000 bits
0-6000: vertical waveform Z
6001-12000: EW waveform of horizontal item
12001-18000: is the horizontal item NS waveform

Each record contains samples of three channels
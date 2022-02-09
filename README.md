# lstm-classical-piano
Making AI compose piano music

This repo is to generate unique classical piano melodies using LSTM models with a dataset of piano works by well-known composers.
Dataset contains MIDI files.
Repo is dependent on various python libraries namely Music21, Pandas, scikit-learn.

 midi_io.py        -> Converts MIDI to a combinational data using a data representation approach (Memis and Yalim Keles, 2021). It also contains modules for converting back to MIDI format.
 process_train.py  -> Applies some pre-processing and performs the training. 

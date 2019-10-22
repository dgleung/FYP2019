from music21 import midi

mf = midi.MidiFile()
mf.open('./testmidi.MID')
mf.read()
mf.close()

events = mf.tracks[0].events

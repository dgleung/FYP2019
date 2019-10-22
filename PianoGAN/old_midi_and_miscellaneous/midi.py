# Thursday Week 4 22 August
# Created by David Leung
# Attempting to use music21 library to manipulate midi files

import music21.corpus as corpus
import string
from music21 import harmony, converter, corpus, instrument, midi, note, chord, pitch, stream
import numpy as np
import pandas as pd     # data processing, CSV file I/O
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from progress.bar import Bar

# bar = Bar('Processing', max=20)

def open_midi(midi_path):
    mf = midi.MidiFile()
    mf.open(midi_path,attrib='rb')
    mf.read()
    mf.close()
    return mf
    #return midi.translate.midiFileToStream(mf)

def midi2keystrikes(filename,tracknum=0):
    """ Reads a midifile, returns list of key hits"""

    mf = midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    events = mf.tracks[tracknum].events
    result = []
    t=0

    for data in events:
        if data.isDeltaTime() and (data.time is not None):

            t += data.time
        if data.isNoteOn() or data.isNoteOff():
            result.append(data.pitch)
            result.append(data.velocity)

    return result

def list_instruments(midi):
    partStream = midi.parts.stream()
    print("List of instruments found on MIDI file:")
    for p in partStream:
        print (p.partName)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    for nt in midi_part.flat.notes:
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
        elif isinstance(nt, chord.Chord):
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)

    return ret, parent_element


def print_parts_countour(midi):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minPitch = pitch.Pitch('C10').ps
    maxPitch = 0
    xMax = 0

    # Drawing notes.
    for i in range(len(midi.parts)):
        top = midi.parts[i].flat.notes
        y, parent_element = extract_notes(top)
        if (len(y) < 1): continue

        x = [n.offset for n in parent_element]
        ax.scatter(x, y, alpha=0.6, s=7)

        aux = min(y)
        if (aux < minPitch): minPitch = aux

        aux = max(y)
        if (aux > maxPitch): maxPitch = aux

        aux = max(x)
        if (aux > xMax): xMax = aux

    for i in range(1, 10):
        linePitch = pitch.Pitch('C{0}'.format(i)).ps
        if (linePitch > minPitch and linePitch < maxPitch):
            ax.add_line(mlines.Line2D([0, xMax], [linePitch, linePitch], color='red', alpha=0.1))

    plt.ylabel("Note index (each octave has 12 notes)")
    plt.xlabel("Number of quarter notes (beats)")
    plt.title('Voices motion approximation, each color is a different instrument, red lines show each octave')
    plt.show()

filepath = './AbdelmoulaJeanSelimSMF/AbdelmoulaJS04.MID'

midiraw = open_midi(filepath)
base_midi = midi2keystrikes(filepath)
#list_instruments(base_midi)

# base_midi.show('text')

# Focusing only on 6 first measures to make it easier to understand.
# print_parts_countour(base_midi.measures(0, 50))

# flatRH = base_midi.parts[0].notesAndRests
# flatLH = base_midi.parts[1].flat.notesAndRests

def listStream(streamIn):
    for data in streamIn:
        print(data.offset, data)

#
# chorded = base_midi.chordify()
# chorded = harmony.realizeChordSymbolDurations(chorded)
# newStream = stream.Stream(chorded)
# newStream.pop(len(newStream)-1)

file = open('out.txt', "w")
base_midi = tuple(base_midi)
new = '\n'.join(str(v) for v in base_midi)
writein = str(new)
file.write(writein)
file.close()

fileraw = open('raw.txt', "w")
writeraw = str(midiraw)
fileraw.write(writeraw)
fileraw.close()

fileread = open('out.txt', "r")
str1 = fileread.readline()
print(str1)
fileread.close()



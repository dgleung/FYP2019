from music21 import midi, defaults
import pickle
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def noteWrite(miditrack, on, pitch, velocity, channel=1):
    me = midi.MidiEvent(miditrack)
    if bool(on) == True:
        me.type = 'NOTE_ON'
    elif bool(on) == False:
        me.type = 'NOTE_OFF'
    me.channel = channel
    me.pitch = pitch
    me.velocity = velocity
    miditrack.events.append(me)


def eventWrite(miditrack, type, data):
    me = midi.MidiEvent(miditrack)
    me.type = type
    me.data = data
    miditrack.events.append(me)


def deltaWrite(miditrack, time):
    dt = midi.DeltaTime(miditrack)
    dt.time = time
    miditrack.events.append(dt)


def requiredHeader(tempo=117):
    mt1 = midi.MidiTrack(1)
    '''
    FROM EVENT: midi.getNumber('<str data>', 3)[0] 
    TEMPO TO EVENT: int(round(60000000.0/bpm)
                    me.data = midi.putNumber(<result>, 3)
    '''
    tempo = tempo
    tempo = int(round(60000000.0 / tempo))
    tempo = midi.putNumber(tempo, 3)

    deltaWrite(mt1, 0)
    eventWrite(mt1, 'SEQUENCE_TRACK_NAME', 'Piano')
    deltaWrite(mt1, 0)
    eventWrite(mt1, 'SET_TEMPO', tempo)
    deltaWrite(mt1, 0)
    eventWrite(mt1, 'TIME_SIGNATURE', '\x04\x02\x18\x08')
    return mt1


def requiredFooter(midiTrack):
    deltaWrite(midiTrack, 0)
    eventWrite(midiTrack,'END_OF_TRACK','')


def decrypt(intensor):
    on = intensor[:,0:128]
    v = intensor[:,128:160]
    off = intensor[:,160:288]
    t = intensor[:,288:388]

    on = np.expand_dims(np.argmax(on.numpy(),axis=1),axis=1)
    v = np.expand_dims(np.argmax(v.numpy(),axis=1),axis=1)
    off = np.expand_dims(np.argmax(off.numpy(),axis=1),axis=1)
    t = np.expand_dims(np.argmax(t.numpy(),axis=1),axis=1)

    return np.block([on, v, off, t])


def develocity(velocity_quantized):
    bins = np.linspace(1, 121, 31)
    bins -= 2
    bins[0] = 0
    bins = np.append(bins, 123)

    return int(bins[velocity_quantized])


def detime(time_quantized):
    bins = np.linspace(1, 785, 99)
    bins -= 4
    bins[0] = 0
    bins = np.append(bins, 789)

    return int(bins[time_quantized])


'''

# WRITING EVENT DATA TO MIDI (MUST HAVE DELTATIME IN BETWEEN EACH EVENT)

mt1 = requiredHeader(tempo=117)
deltaWrite(mt1, (5+1+1+1+191+191+191+191+191+191+191+191))
noteWrite(mt1, 1, 60, 100)
deltaWrite(mt1, 0)
noteWrite(mt1, 1, 63, 100)
deltaWrite(mt1, 26610)
noteWrite(mt1, 0, 60, 0)
deltaWrite(mt1, 0)
noteWrite(mt1, 0, 63, 0)
requiredFooter(mt1)


mf = midi.MidiFile()
mf.ticksPerQuarterNote = defaults.ticksPerQuarter
mf.tracks.append(mt1)
mf.open('testmidi.MID','wb')
mf.write()
mf.close()
'''


'''
DECRPYTING ONE-HOT VECTOR REPRESENTATION
'''
with open('./picklesave/singleexample5.data', 'rb') as filehandle:
    testtensor = pickle.load(filehandle)

result = decrypt(testtensor)

# Initialisations
mt2 = requiredHeader()

time_store = 0
pitchOn_store = 0
pitchOff_store = 0
velocity_store = 0


def init_pitch_vol(on, off, vel, time):
    on = 0
    off = 0
    vel = 0
    time = 0
    return on, off, vel, time


for i in range(result.shape[0]):
    if not np.count_nonzero(result[i,:]):   # continue to next shape[1,4] vector if empty (i.e. [0,0,0,0])
        continue
    else:
        nonzero_index = result[i, :].nonzero()[0][0]
        index_data = result[i][nonzero_index]

        if nonzero_index == 3:      # time case
            time_store += detime(index_data)    # dequantizing time required
        elif nonzero_index == 0:    # note on case
            pitchOn_store = index_data
        elif nonzero_index == 1:    # velocity case
            velocity_store = develocity(index_data) # dequantizing velocity required
        elif nonzero_index == 2:    # note off case
            pitchOff_store = index_data

        if not not ((pitchOn_store or pitchOff_store) and velocity_store):
            if pitchOn_store != 0:
                deltaWrite(mt2, time_store)                         # midi track must have DeltaTime event between any other event
                noteWrite(mt2,1,pitchOn_store,velocity_store)       # write note on event
            elif pitchOff_store != 0:
                deltaWrite(mt2, time_store)                         # midi track must have DeltaTime event between any other event
                noteWrite(mt2,0,pitchOff_store,velocity_store)      # write note off event
            pitchOn_store, pitchOff_store, velocity_store, time_store = init_pitch_vol(pitchOn_store, pitchOff_store, velocity_store, time_store)   # re-initialise pitch, velocity and time


requiredFooter(mt2)
mf = midi.MidiFile()
mf.ticksPerQuarterNote = 492 # Used empirical results to make 30 seconds long...  int(defaults.ticksPerQuarter / 2) come close
mf.tracks.append(mt2)
mf.open('testmidi5.MID','wb')
mf.write()
mf.close()


with open('./picklesave/filepathlist.data', 'rb') as filehandle:
    filepathlist = pickle.load(filehandle)



# decode.py
# Contains functions for decoding & reconstruction to midi
# David Leung
# Wednesday 25th September - Week 9

# Import libraries
from numpy import expand_dims, argmax, block, linspace, append, count_nonzero, zeros
from music21 import midi
from lib.functional import vectors_to_samples
from datetime import datetime

# Decrpyt a tensor sample from one-hot encoding to integer representation
def decrypt_real(np_shaped):
    on = np_shaped[:,0:128]
    v = np_shaped[:,128:160]
    off = np_shaped[:,160:288]
    t = np_shaped[:,288:388]

    on = expand_dims(argmax(on,axis=1),axis=1)
    v = expand_dims(argmax(v,axis=1),axis=1)
    off = expand_dims(argmax(off,axis=1),axis=1)
    t = expand_dims(argmax(t,axis=1),axis=1)

    return block([on, v, off, t])


# Decrypt a generated tensor (1D format) directly from generator
def decrypt_generated(intensor):
    tensor_shaped = vectors_to_samples(intensor.cpu())
    tensor_shaped = tensor_shaped.squeeze()
    index = tensor_shaped.numpy().argmax(axis=1)
    onehot_array = zeros((tensor_shaped.shape[0],tensor_shaped.shape[1]))

    for row in range(onehot_array.shape[0]):
        onehot_array[row,index[row]] = 1

    return decrypt_real(onehot_array)


# Write a note to midi track
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


# Write event to midi track
def eventWrite(miditrack, type, data):
    me = midi.MidiEvent(miditrack)
    me.type = type
    me.data = data
    miditrack.events.append(me)


# Write deltatime event to midi track
def deltaWrite(miditrack, time):
    dt = midi.DeltaTime(miditrack)
    dt.time = time
    miditrack.events.append(dt)


# Create empty midi track with required header data
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


# Write required footer data to midi track (EOT)
def requiredFooter(midiTrack):
    deltaWrite(midiTrack, 0)
    eventWrite(midiTrack,'END_OF_TRACK','')


# Convert quantized velocity into real value
def develocity(velocity_quantized):
    bins = linspace(1, 121, 31)
    bins -= 2
    bins[0] = 0
    bins = append(bins, 123)

    return int(bins[velocity_quantized])


# Convert quantized time into real value
def detime(time_quantized):
    bins = linspace(1, 785, 99)
    bins -= 4
    bins[0] = 0
    bins = append(bins, 789)

    return int(bins[time_quantized])


# Initialise note, velocity and time variables
def init_pitch_vol(on, off, vel, time):
    on = 0
    off = 0
    vel = 0
    time = 0
    return on, off, vel, time


# Convert [on, vel, off, time] sequence into midi file
def write_midi(result, savepath):
    # Initialisations
    mt = requiredHeader()
    time_store = 0
    pitchOn_store = 0
    pitchOff_store = 0
    velocity_store = 0

    # Loop to write midi track data to file
    for i in range(result.shape[0]):
        if not count_nonzero(result[i,:]):   # continue to next shape[1,4] vector if empty (i.e. [0,0,0,0])
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
                    deltaWrite(mt, time_store)                         # midi track must have DeltaTime event between any other event
                    noteWrite(mt,1,pitchOn_store,velocity_store)       # write note on event
                elif pitchOff_store != 0:
                    deltaWrite(mt, time_store)                         # midi track must have DeltaTime event between any other event
                    noteWrite(mt,0,pitchOff_store,velocity_store)      # write note off event
                pitchOn_store, pitchOff_store, velocity_store, time_store = init_pitch_vol(pitchOn_store, pitchOff_store, velocity_store, time_store)   # re-initialise pitch, velocity and time

    # Signal end of track and save
    requiredFooter(mt)
    mf = midi.MidiFile()
    mf.ticksPerQuarterNote = 492 # Used empirical results to make 30 seconds long...  int(defaults.ticksPerQuarter / 2) come close
    mf.tracks.append(mt)
    mf.open(savepath,'wb')
    mf.write()
    mf.close()

    print('\nsaved: '+savepath+'\n')


# Directly from generator to midifile
def gen2midi(tensor):
    dt = datetime.now()
    savepath = './generated/'+'generated'+str(dt.year)+'-'+str(dt.month)+'-'+str(dt.day)+'_'+str(dt.hour)+'-'+str(dt.minute)+'-'+str(dt.second)+'.mid'
    result = decrypt_generated(tensor)
    write_midi(result, savepath)

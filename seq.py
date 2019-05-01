# BEGIN imports

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import random
import pretty_midi
import os
import time
import pdb

device = torch.device('cuda:0')
torch.cuda.empty_cache()
# size of input/output vector
vec_size = 128

# END imports

# BEGIN model definition

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, dropout=.5)
        
    def forward(self, input):
        output, (hidden, cell) = self.gru(input)
        return hidden, cell
    
    #def initHidden(self):
    #    return torch.zeros(2, 102, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, num_layers=2, dropout=.5)
        self.out = nn.Linear(hidden_size, output_size)
        # some use softmax here?
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input, hidden, cell):
        output = F.relu(input)
        output, (hidden, cell) = self.gru(output, torch.stack((hidden, cell), dim=0))
        output = self.sigmoid(self.out(output[0]))
        return output, hidden, cell
    
    #def initHidden(self):
    #    return torch.zeros(2, 102, self.hidden_size, device = device)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        #initialize output vector
        length = round(src.numel() / vec_size)
        src = src.view(-1, vec_size)
        trg = trg.view(-1, vec_size)
        outputs = torch.zeros((length, vec_size)).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src.view(length,1,vec_size).to(self.device))
        
        decode_input = trg[0,:] #SOS token
            
        for t in range(0, length):
            output, hidden, cell = self.decoder(decode_input.view(1,1,vec_size).to(self.device), hidden.to(self.device), cell.to(self.device))
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            decode_input = (trg[t] if teacher_force else output)
        
        return outputs

# END model definition

# BEGIN data preparation

# load midi file a single track piano roll.
def load_piano_roll(filename):
    midi_data = pretty_midi.PrettyMIDI(filename)
    piano_roll = midi_data.get_piano_roll()
    # Binarize the pressed notes
    piano_roll[piano_roll > 0] = 1
    return piano_roll

# pad (or crop) the piano roll with 0 to make it length
def pad_piano_roll(roll, length):
    roll_length = roll.shape[1]
    if roll_length < length:
        return np.concatenate((roll, np.zeros((roll.shape[0], length-roll_length))), axis=1)
    return roll[:, :length]

# add sos and eos tokens to the piano roll
def add_tokens(roll):
    sos = np.zeros((roll.shape[0], 1))
    eos = np.ones((roll.shape[0], 1))
    return np.concatenate((sos, roll, eos), axis=1)



class MIDIDataSet(data.Dataset):
    def __init__(self, dir_name, longest=1500):
        filenames = os.listdir(dir_name)
        self.full_filenames = list(map(lambda f: os.path.join(dir_name, f), filenames))
        self.longest = longest
        #read_all()

    # build a giant piano roll of data; add SOS and EOS tokens in between
    def read_all(self):
        for filename in self.full_filenames:
            roll = load_piano_roll(filename)
            roll = pad_piano_roll(roll, self.longest)
            roll = add_tokens(roll)
            if not self.data:
                self.data = roll
            else:
                self.data = np.concatenate((self.data, roll), axis=1)

    def __len__(self):
        return len(self.full_filenames)
        #return len(self.data)

    def __getitem__(self, idx):
        roll = load_piano_roll(self.full_filenames[idx])
        roll = pad_piano_roll(roll, self.longest)
        return np.transpose(roll[:, :-1]), np.transpose(roll[:, 1:])
        #return self.data[idx,:]

# END data preparation


# BEGIN training
dataset = MIDIDataSet('./midi/pop', longest=500)
train_size = round(len(dataset)*0.8)
valid_size = len(dataset) - train_size
dataset_split = torch.utils.data.random_split(dataset, (train_size, valid_size))
print('Training data loaded')
train_loader = data.DataLoader(dataset_split[0], batch_size=1, shuffle=True)
valid_loader = data.DataLoader(dataset_split[1], batch_size=1, shuffle=True)

encoder = EncoderRNN(128, 512).to(device)
decoder = DecoderRNN(512, 128).to(device)

s2s = seq2seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(s2s.parameters())

criterion = nn.BCELoss()

num_epochs = 5

tr_loss_list = []
tt_loss_list = []

for epoch in range(num_epochs):
    epoch_training_loss = 0
    # train
    for X, Y in train_loader:
        X = X.to(device=device, dtype=torch.float32)
        Y = Y.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        output = s2s(X, Y)
        t = time.time()
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        print('Loss calc:', time.time() - t)

        epoch_training_loss += loss.item()
        
        
    # validate
    #s2s.eval()
    with torch.no_grad():
        epoch_validation_loss = 0
        for X, Y in valid_loader:
            X = X.to(device=device, dtype=torch.float32)
            Y = Y.to(device=device, dtype=torch.float32)
            output = s2s(X, Y, 0)
            loss = criterion(output, Y)
            epoch_validation_loss += loss.item()
        
        
    tr_loss_list.append(epoch_training_loss/train_size)
    tt_loss_list.append(epoch_validation_loss/valid_size)

    print('We are on epoch ', epoch)
    print('The current training loss is ', epoch_training_loss)
    print('The current validation loss is ', epoch_validation_loss)
    print()
    
    state = {
            'epoch': epoch,
            'state_dict': s2s.state_dict(),
            'optimizer': optimizer.state_dict(),
    }
    
    torch.save(state, 'modelstate.pth')

# END training

# BEGIN generation

def generate(model_filename, output_filename, length):
    encoder = EncoderRNN(vec_size, 512).to(device)
    decoder = DecoderRNN(512, vec_size).to(device)
    s2s = seq2seq(encoder, decoder, device).to(device)
    
    state = torch.load('modelstate.pth')
    s2s.load_state_dict(state['state_dict'])

    beginning_sequence = np.zeros((128, 1))
    beginning_sequence[np.random.choice(a=128, size=5, replace=False)] = 1
    outputs = torch.zeros((length, vec_size)).to(device)

    sequence = torch.tensor(beginning_sequence).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        for t in range(0, length):
            outputs[t] = s2s(sequence, sequence, 0)
            sequence = torch.tensor(outputs[t]).to(device=device, dtype=torch.float32)
        
    outputs = outputs.cpu().numpy()
    notes = np.where(outputs > 0.5, 1, 0)
    midi = piano_roll_to_pretty_midi(np.transpose(notes), fs=5)
    midi.write(output_filename)

    return notes


# END generation


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

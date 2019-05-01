# direct copy from https://github.com/warmspringwinds/pytorch-rnn-sequence-generation-classification/blob/master/notebooks/music_generation_training_nottingham.ipynb

import pretty_midi
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import os
#from utils import midiread, midiwrite

#sys.path.append("./Piano-midi.de/")

print('BEGIN EXECUTION')


device = torch.device('cuda:0')
print(device)
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
#     torch.cuda.device(device)
# else:
#     device = torch.device('cpu')
    
def midi_filename_to_piano_roll(midi_filename):
    midi_data = pretty_midi.PrettyMIDI(midi_filename)
    #midi_data = midiread(midi_filename, dt=0.3)
    # why transpose?
    piano_roll = midi_data.get_piano_roll()#.transpose()
    # Binarize the pressed notes
    piano_roll[piano_roll > 0] = 1
    return piano_roll


def pad_piano_roll(piano_roll, max_length=132333, pad_value=0):
    # We hardcode 88 -- because we will always use only 128 pitches
    # max length?

    # print('input piano roll size', piano_roll.shape)
    roll_length = piano_roll.shape[1]
    if roll_length < max_length:
        rows_to_concat = max_length - roll_length
        return np.concatenate((piano_roll[20:108, :], np.zeros((88,rows_to_concat))), axis=1)
    
    # padded_piano_roll = np.zeros((128, max_length))
    # padded_piano_roll[:] = pad_value
    
    # padded_piano_roll[:, :original_piano_roll_length] = piano_roll
    # return padded_piano_roll

    # we actually need to crop
    return piano_roll[20:108, :max_length]

class NotesGenerationDataset(data.Dataset):
    
    def __init__(self, midi_folder_path, longest_sequence_length=5000):
        self.midi_folder_path = midi_folder_path
        midi_filenames = os.listdir(midi_folder_path)
        self.longest_sequence_length = longest_sequence_length
        midi_full_filenames = list(map(lambda filename: os.path.join(midi_folder_path, filename), midi_filenames))
        self.midi_full_filenames = midi_full_filenames
        if longest_sequence_length is None:
            self.update_the_max_length()
    
    def update_the_max_length(self):
        """Recomputes the longest sequence constant of the dataset.

        Reads all the midi files from the midi folder and finds the max
        length.
        """
        
        sequences_lengths = list(map(lambda filename: midi_filename_to_piano_roll(filename).shape[1], self.midi_full_filenames))
        max_length = max(sequences_lengths)
        self.longest_sequence_length = max_length
    
    def __len__(self):
        return len(self.midi_full_filenames)
    
    def __getitem__(self, index):
        midi_full_filename = self.midi_full_filenames[index]
        piano_roll = midi_filename_to_piano_roll(midi_full_filename)
        # -1 because we will shift it
        sequence_length = piano_roll.shape[1] - 1
        # Shifted by one time step
        input_sequence = piano_roll[:, :-1]
        ground_truth_sequence = piano_roll[:, 1:]
                
        # pad sequence so that all of them have the same lenght
        # Otherwise the batching won't work
        input_sequence_padded = pad_piano_roll(input_sequence, max_length=self.longest_sequence_length)
        ground_truth_sequence_padded = pad_piano_roll(ground_truth_sequence,
                                                     max_length=self.longest_sequence_length,
                                                     pad_value=-100)
                
        input_sequence_padded = input_sequence_padded.transpose()
        ground_truth_sequence_padded = ground_truth_sequence_padded.transpose()
        
        return (torch.FloatTensor(input_sequence_padded),
                torch.LongTensor(ground_truth_sequence_padded),
                torch.LongTensor([input_sequence_padded.shape[1]]) )

def post_process_sequence_batch(batch_tuple):
    
    input_sequences, output_sequences, lengths = batch_tuple
    
    splitted_input_sequence_batch = input_sequences.split(split_size=1)
    splitted_output_sequence_batch = output_sequences.split(split_size=1)
    splitted_lengths_batch = lengths.split(split_size=1)

    training_data_tuples = zip(splitted_input_sequence_batch,
                               splitted_output_sequence_batch,
                               splitted_lengths_batch)

    training_data_tuples_sorted = sorted(training_data_tuples,
                                         key=lambda p: int(p[2]),
                                         reverse=True)

    splitted_input_sequence_batch, splitted_output_sequence_batch, splitted_lengths_batch = zip(*training_data_tuples_sorted)

    input_sequence_batch_sorted = torch.cat(splitted_input_sequence_batch)
    output_sequence_batch_sorted = torch.cat(splitted_output_sequence_batch)
    lengths_batch_sorted = torch.cat(splitted_lengths_batch)
    
    # Here we trim overall data matrix using the size of the longest sequence
    input_sequence_batch_sorted = input_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]
    output_sequence_batch_sorted = output_sequence_batch_sorted[:, :lengths_batch_sorted[0, 0], :]
    
    input_sequence_batch_transposed = input_sequence_batch_sorted.transpose(0, 1)
    
    # pytorch's api for rnns wants lenghts to be list of ints
    lengths_batch_sorted_list = list(lengths_batch_sorted)
    lengths_batch_sorted_list = map(lambda x: int(x), lengths_batch_sorted_list)
    
    return input_sequence_batch_transposed, output_sequence_batch_sorted, lengths_batch_sorted_list



class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes, n_layers=4):
        
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.logits_fc = nn.Linear(hidden_size, num_classes)
    
    
    def forward(self, input_sequences, input_sequences_lengths, hidden=None):
        
        batch_size = input_sequences.shape[1]
        notes_encoded = self.notes_encoder(input_sequences)
        
        # Here we run rnns only on non-padded regions of the batch
        #packed = torch.nn.utils.rnn.pack_padded_sequence(notes_encoded, input_sequences_lengths)
        outputs, hidden = self.lstm(notes_encoded, hidden)
        #outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        
        logits = self.logits_fc(outputs)
        # actually change the memory layout, not just change tensor metadata which is what transpose() does
        logits = logits.transpose(0, 1).contiguous()
        
        # switch 0 and 1?
        neg_logits = (1 - logits)
        
        # Since the BCE loss doesn't support masking, we use the crossentropy
        # doc mentions that we're using binary cross entropy. what?
        binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
        
        logits_flatten = binary_logits.view(-1, 2)
        
        return logits_flatten, hidden

def validate():

    full_val_loss = 0.0
    overall_sequence_length = 0.0

    for batch in valset_loader:

        post_processed_batch_tuple = post_process_sequence_batch(batch)

        input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple

        output_sequences_batch_var =  Variable( output_sequences_batch.contiguous().view(-1).to(device) )

        input_sequences_batch_var = Variable( input_sequences_batch.to(device) )

        logits, _ = rnn(input_sequences_batch_var, sequences_lengths)

        loss = criterion_val(logits, output_sequences_batch_var)

        full_val_loss += loss.item()
        #full_val_loss += loss.data[0]
        overall_sequence_length += sum(sequences_lengths)

    return full_val_loss / (overall_sequence_length * 88)



print('Trainset load')
trainset = NotesGenerationDataset('./pop-music-collection/train/')

trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, drop_last=True)

print('Valset load')
valset = NotesGenerationDataset('./pop-music-collection/valid/', longest_sequence_length=None)

valset_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False, drop_last=False)

# define model and optimizer
# train on 88 keys -- the range of piano. MIDI has 127 keys, so we'll have to pack/unpack ??
# what about rests?

rnn = RNN(input_size=88, hidden_size=512, num_classes=88)
rnn = rnn.to(device)
#rnn.load_state_dict(torch.load('music_rnn3.pth'))


"""
Instead, we will treat each element of the output vector as a binary variable ($1$ -- pressing, $0$ -- not pressing a key). 
We will define a separate loss for each element of the output vector to be binary cross-entropy.
And our final loss will be an averaged sum of these binary cross-entropies. 
"""
criterion = nn.CrossEntropyLoss().to(device)
criterion_val = nn.CrossEntropyLoss(size_average=False).to(device)

learning_rate = 0.001
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)


# now train!
clip = 1.0
epochs_number = 160
sample_history = []
best_val_loss = float("inf")

for epoch_number in range(epochs_number):
    print('Epoch', epoch_number)
    for batch in trainset_loader:
        post_processed_batch_tuple = post_process_sequence_batch(batch)
        input_sequences_batch, output_sequences_batch, sequences_lengths = post_processed_batch_tuple
        output_sequences_batch_var =  Variable( output_sequences_batch.contiguous().view(-1).to(device) )
        input_sequences_batch_var = Variable( input_sequences_batch.to(device) )
        
        optimizer.zero_grad()
        logits, _ = rnn(input_sequences_batch_var, sequences_lengths)
        loss = criterion(logits, output_sequences_batch_var)
        #loss_list.append( loss.data[0] )
        loss.backward()
        
        torch.nn.utils.clip_grad_norm(rnn.parameters(), clip)
        optimizer.step()
    
    
    current_val_loss = validate()
    print(current_val_loss)
    #val_list.append(current_val_loss)
    
    if current_val_loss < best_val_loss:
        print('Create checkpoint')
        torch.save(rnn.state_dict(), 'music_rnn4.pth')
        best_val_loss = current_val_loss

"""
Music Encoding/Decoding System

This module provides classes for encoding music from MIDI to text and vice versa.
The encoding process follows these steps:
1. MIDI file -> music21.Stream
2. Stream -> numpy chord array (timestep x instrument x noterange)
3. numpy array -> List[Timestep][NoteEnc] -> text representation

The decoding process follows the reverse steps.
"""

import music21
import numpy as np
import os
from itertools import groupby

# Constants
BPB = 4  # beats per bar
TIMESIG = f'{BPB}/4'  # default time signature
PIANO_RANGE = (21, 108)  # standard piano range (A0 to C8)
VALTSEP = -1  # separator value for numpy encoding
VALTCONT = -2  # numpy value for TCONT - needed for compressing chord array
DEFAULT_VELOCITY = 64
VELOCITY_RANGE = (0, 127)
SAMPLE_FREQ = 4
NOTE_SIZE = 128
DUR_SIZE = (10*BPB*SAMPLE_FREQ)+1  # Max length - 10 bars
MAX_NOTE_DUR = (8*BPB*SAMPLE_FREQ)


class Encoder:
    """
    Class for encoding MIDI files to numpy arrays and text representations.
    """
    
    def __init__(self, 
                 note_size=NOTE_SIZE,
                 sample_freq=SAMPLE_FREQ,
                 max_note_dur=MAX_NOTE_DUR,
                 piano_range=PIANO_RANGE):
        """
        Initialize the encoder with configurable parameters.
        
        Args:
            note_size: Size of the note range
            sample_freq: Sampling frequency for the encoding
            max_note_dur: Maximum note duration
            piano_range: Range of piano notes to include
        """
        self.note_size = note_size
        self.sample_freq = sample_freq
        self.max_note_dur = max_note_dur
        self.piano_range = piano_range
    
    def process_midi_directory(self, midi_directory, text_directory):
        """
        Process all MIDI files in a directory and save their text representations.
        
        Args:
            midi_directory: Directory containing MIDI files
            text_directory: Directory to save text representations
        """
        for filename in os.listdir(midi_directory):
            if filename.endswith(".mid"):
                path = os.path.join(midi_directory, filename)
                output_path = f"{text_directory}/{filename.split('.')[0]}.txt"
                
                # Check if text file already exists
                if os.path.isfile(output_path):
                    print(f"Text file {filename.split('.')[0]}.txt already exists.")
                    continue
                
                self.save_text_representation(path, text_directory)
            
    def save_text_representation(self, path, output_dir):
        """
        Convert a MIDI file to text representation and save it to file.
        
        Args:
            path: Path to the MIDI file
            output_dir: Directory to save the text file
        """
        # Check if MIDI file exists
        if not os.path.isfile(path):
            print(f"MIDI file {path} does not exist.")
            return

        # Check if output directory exists
        if not os.path.isdir(output_dir):
            print(f"Output directory {output_dir} does not exist.")
            return

        # Extract the name of the MIDI file
        name = path.split("\\")[-1].split(".")[0]
        
        # Save this to a file
        try:
            with open(f"{output_dir}/{name}.txt", 'w') as f:
                f.write(self.midi_to_text(path))
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")
    
    def midi_to_text(self, path):
        """
        Convert MIDI file to text representation including velocity.
        
        Args:
            path: Path to the MIDI file
            
        Returns:
            String representation of the MIDI file
        """
        data = self.process_midi(path)
        result = []
        for array in data:
            val1, val2, val3 = array[0], array[1], array[2]  # includes velocity
            if val1 == VALTSEP:
                result.append(f"sepxx d{val2}")  # separators don't need velocity
            else:
                result.append(f"n{val1} d{val2} v{val3}")
        result.insert(0, "START")
        return " ".join(result)
    
    def process_midi(self, path):
        """
        Process a MIDI file to numpy encoding.
        
        Args:
            path: Path to the MIDI file
            
        Returns:
            Numpy array encoding of the MIDI file
        """
        if not os.path.exists(path):
            print(f"File {path} does not exist")
            return None

        # Load the MIDI file
        mf = music21.midi.MidiFile()
        mf.open(path)
        mf.read()
        mf.close()

        return self.midi_to_npenc(mf)
    
    def midi_to_npenc(self, midi_file, skip_last_rest=True):
        """
        Convert MIDI file to numpy encoding for language model.
        
        Args:
            midi_file: MIDI file or path
            skip_last_rest: Whether to skip the last rest
            
        Returns:
            Numpy encoding of the MIDI file
        """
        stream = self._file_to_stream(midi_file)
        chordarr = self._stream_to_chordarr(stream)
        return self._chordarr_to_npenc(chordarr, skip_last_rest=skip_last_rest)
    
    def _file_to_stream(self, fp):
        """
        Convert MIDI file to music21 stream.
        
        Args:
            fp: File path or music21.midi.MidiFile
            
        Returns:
            music21.stream.Stream
        """
        if isinstance(fp, music21.midi.MidiFile):
            return music21.midi.translate.midiFileToStream(fp)
        return music21.converter.parse(fp)
    
    def _stream_to_chordarr(self, s):
        """
        Convert music21.Stream to numpy array including velocity.
        
        Args:
            s: music21.stream.Stream
            
        Returns:
            Numpy chord array
        """
        highest_time = max(s.flat.getElementsByClass('Note').highestTime, 
                          s.flat.getElementsByClass('Chord').highestTime)
        maxTimeStep = round(highest_time * self.sample_freq) + 1
        
        # Add third dimension for velocity
        score_arr = np.zeros((maxTimeStep, len(s.parts), self.note_size, 3))
        
        def note_data(pitch, note):
            velocity = note.volume.velocity if note.volume.velocity is not None else DEFAULT_VELOCITY
            return (pitch.midi, 
                    int(round(note.offset * self.sample_freq)),
                    int(round(note.duration.quarterLength * self.sample_freq)),
                    velocity)

        for idx, part in enumerate(s.parts):
            notes = []
            for elem in part.flat:
                if isinstance(elem, music21.note.Note):
                    notes.append(note_data(elem.pitch, elem))
                if isinstance(elem, music21.chord.Chord):
                    for p in elem.pitches:
                        notes.append(note_data(p, elem))
                        
            notes_sorted = sorted(notes, key=lambda x: (x[1], x[2]))
            for n in notes_sorted:
                if n is None: continue
                pitch, offset, duration, velocity = n
                if self.max_note_dur is not None and duration > self.max_note_dur: 
                    duration = self.max_note_dur
                    
                # Store duration in first channel
                score_arr[offset, idx, pitch, 0] = duration
                # Store velocity in second channel
                score_arr[offset, idx, pitch, 1] = velocity
                # Use third channel for continuation marker
                score_arr[offset+1:offset+duration, idx, pitch, 2] = VALTCONT
                
        return score_arr
    
    def _chordarr_to_npenc(self, chordarr, skip_last_rest=True):
        """
        Convert chord array to numpy encoding including velocity.
        
        Args:
            chordarr: Chord array
            skip_last_rest: Whether to skip the last rest
            
        Returns:
            Numpy encoding
        """
        result = []
        wait_count = 0
        for idx, timestep in enumerate(chordarr):
            flat_time = self._timestep_to_npenc(timestep)
            if len(flat_time) == 0:
                wait_count += 1
            else:
                if wait_count > 0: 
                    result.append([VALTSEP, wait_count, DEFAULT_VELOCITY])
                result.extend(flat_time)
                wait_count = 1
        if wait_count > 0 and not skip_last_rest: 
            result.append([VALTSEP, wait_count, DEFAULT_VELOCITY])
        return np.array(result, dtype=int).reshape(-1, 3)
    
    def _timestep_to_npenc(self, timestep, enc_type=None):
        """
        Convert timestep to encoding with velocity.
        
        Args:
            timestep: Timestep
            enc_type: Encoding type
            
        Returns:
            List of encoded notes
        """
        notes = []
        for i, n in zip(*timestep[..., 0].nonzero()):  # look at duration channel
            d = timestep[i, n, 0]  # duration
            if d < 0: continue
            if n < self.piano_range[0] or n >= self.piano_range[1]: continue
            v = int(timestep[i, n, 1])  # velocity
            notes.append([n, d, v, i])
            
        notes = sorted(notes, key=lambda x: x[0], reverse=True)
        
        if enc_type is None:
            return [n[:3] for n in notes]  # note, duration, velocity
        if enc_type == 'parts':
            return notes  # note, duration, velocity, part
        if enc_type == 'full':
            return [[n%12, d, v, n//12, i] for n,d,v,i in notes]  # note_class, duration, velocity, octave, instrument
    
    def is_valid_npenc(self, npenc, min_notes=32, input_path=None, verbose=True):
        """
        Check if the numpy encoding is valid.
        
        Args:
            npenc: Numpy encoding
            min_notes: Minimum number of notes
            input_path: Input file path
            verbose: Whether to print verbose output
            
        Returns:
            Whether the encoding is valid
        """
        if len(npenc) < min_notes:
            if verbose: print('Sequence too short:', len(npenc), input_path)
            return False
        if (npenc[:,1] >= DUR_SIZE).any(): 
            if verbose: print(f'npenc exceeds max {DUR_SIZE} duration:', npenc[:,1].max(), input_path)
            return False
        # Check if notes are within piano range
        if ((npenc[...,0] > VALTSEP) & ((npenc[...,0] < self.piano_range[0]) | (npenc[...,0] >= self.piano_range[1]))).any(): 
            print(f'npenc out of piano note range {self.piano_range}:', input_path)
            return False
        return True
    
    # Data processing methods
    def compress_chordarr(self, chordarr):
        """
        Compress chord array by shortening rests.
        
        Args:
            chordarr: Chord array
            
        Returns:
            Compressed chord array
        """
        return self._shorten_chordarr_rests(self._trim_chordarr_rests(chordarr))
    
    def _trim_chordarr_rests(self, arr, max_rests=4):
        """
        Trim rests at the beginning and end of chord array.
        
        Args:
            arr: Chord array
            max_rests: Maximum number of rests in quarter notes
            
        Returns:
            Trimmed chord array
        """
        # max rests is in quarter notes
        # max 1 bar between song start and end
        start_idx = 0
        max_sample = max_rests * self.sample_freq
        for idx, t in enumerate(arr):
            if (t != 0).any(): break
            start_idx = idx + 1
            
        end_idx = 0
        for idx, t in enumerate(reversed(arr)):
            if (t != 0).any(): break
            end_idx = idx + 1
        start_idx = start_idx - start_idx % max_sample
        end_idx = end_idx - end_idx % max_sample
        return arr[start_idx:(len(arr)-end_idx)]
    
    def _shorten_chordarr_rests(self, arr, max_rests=8):
        """
        Shorten consecutive rests in chord array.
        
        Args:
            arr: Chord array
            max_rests: Maximum number of rests in quarter notes
            
        Returns:
            Chord array with shortened rests
        """
        # max rests is in quarter notes
        # max 2 bar pause
        rest_count = 0
        result = []
        max_sample = max_rests * self.sample_freq
        for timestep in arr:
            if (timestep == 0).all(): 
                rest_count += 1
            else:
                if rest_count > max_sample:
                    rest_count = (rest_count % self.sample_freq) + max_sample
                for i in range(rest_count): 
                    result.append(np.zeros(timestep.shape))
                rest_count = 0
                result.append(timestep)
        for i in range(rest_count): 
            result.append(np.zeros(timestep.shape))
        return np.array(result)


class Decoder:
    """
    Class for decoding text representations back to MIDI files.
    """
    
    def __init__(self, 
                 note_size=NOTE_SIZE,
                 sample_freq=SAMPLE_FREQ,
                 time_sig=TIMESIG):
        """
        Initialize the decoder with configurable parameters.
        
        Args:
            note_size: Size of the note range
            sample_freq: Sampling frequency for the encoding
            time_sig: Time signature
        """
        self.note_size = note_size
        self.sample_freq = sample_freq
        self.time_sig = time_sig
    
    def text_to_midi(self, text, output_dir, name="test", bpm=120):
        """
        Convert text representation to MIDI with velocity information.
        
        Args:
            text: Text representation
            output_dir: Output directory
            name: Output file name
            bpm: Beats per minute
        """
        if not os.path.isdir(output_dir):
            print(f"Output directory {output_dir} does not exist.")
            return

        array = self._process_text(text)
        midi = self.npenc_to_stream(array, bpm=bpm)
        midi.write("midi", f"{output_dir}/{name}.mid")
    
    def _process_text(self, text):
        """
        Process text representation with velocity.
        
        Args:
            text: Text representation
            
        Returns:
            Numpy array
        """
        text = text.replace("START", "").replace("END", "")
        words = text.split()
        
        # Handle groups of three (note, duration, velocity) or two (separator, duration)
        result = []
        i = 0
        while i < len(words):
            if words[i].startswith("sepxx"):
                if i + 1 < len(words) and words[i + 1].startswith("d"):
                    result.append([VALTSEP, int(words[i + 1][1:]), DEFAULT_VELOCITY])
                    i += 2
                else:
                    i += 1
            elif words[i].startswith("n") and i + 2 < len(words):
                if words[i + 1].startswith("d") and words[i + 2].startswith("v"):
                    note = int(words[i][1:])
                    duration = int(words[i + 1][1:])
                    velocity = int(words[i + 2][1:])
                    result.append([note, duration, velocity])
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        return np.array(result)
    
    def npenc_to_stream(self, arr, bpm=120):
        """
        Convert numpy encoding to music21 stream.
        
        Args:
            arr: Numpy encoding
            bpm: Beats per minute
            
        Returns:
            music21.stream.Score
        """
        chordarr = self._npenc_to_chordarr(np.array(arr))
        return self._chordarr_to_stream(chordarr, bpm=bpm)
    
    def _npenc_to_chordarr(self, npenc):
        """
        Convert numpy encoding to chord array with velocity.
        
        Args:
            npenc: Numpy encoding
            
        Returns:
            Chord array
        """
        num_instruments = 1 if len(npenc.shape) <= 2 else npenc.max(axis=0)[-1]
        max_len = self._npenc_len(npenc)
        
        score_arr = np.zeros((max_len, num_instruments, self.note_size, 3))
        
        idx = 0
        for step in npenc:
            n, d, v = step.tolist()  # note, duration, velocity
            if n < VALTSEP: continue
            if n == VALTSEP:
                idx += d
                continue
                
            score_arr[idx, 0, n, 0] = d  # duration
            score_arr[idx, 0, n, 1] = v  # velocity
            
        return score_arr
    
    def _npenc_len(self, npenc):
        """
        Calculate the length of numpy encoding.
        
        Args:
            npenc: Numpy encoding
            
        Returns:
            Length
        """
        duration = 0
        for t in npenc:
            if t[0] == VALTSEP: duration += t[1]
        return duration + 1
    
    def _chordarr_to_stream(self, arr, bpm=120):
        """
        Convert chord array to music21 stream.
        
        Args:
            arr: Chord array
            bpm: Beats per minute
            
        Returns:
            music21.stream.Score
        """
        duration = music21.duration.Duration(1. / self.sample_freq)
        stream = music21.stream.Score()
        stream.append(music21.meter.TimeSignature(self.time_sig))
        stream.append(music21.tempo.MetronomeMark(number=bpm))
        stream.append(music21.key.KeySignature(0))
        for inst in range(arr.shape[1]):
            p = self._partarr_to_stream(arr[:,inst,:], duration)
            stream.append(p)
        stream = stream.transpose(0)
        return stream
    
    def _partarr_to_stream(self, partarr, duration):
        """
        Convert instrument part to music21 chords.
        
        Args:
            partarr: Part array
            duration: Note duration
            
        Returns:
            music21.stream.Part
        """
        part = music21.stream.Part()
        part.append(music21.instrument.Piano())
        self._part_append_duration_notes(partarr, duration, part)
        return part
    
    def _part_append_duration_notes(self, partarr, duration, stream):
        """
        Convert instrument part to music21 chords with velocity.
        
        Args:
            partarr: Part array
            duration: Note duration
            stream: music21.stream.Stream
            
        Returns:
            music21.stream.Stream
        """
        for tidx, t in enumerate(partarr):
            note_idxs = np.where(t[..., 0] > 0)[0]  # check duration channel
            if len(note_idxs) == 0: continue
            
            notes = []
            for nidx in note_idxs:
                note = music21.note.Note(nidx)
                note.duration = music21.duration.Duration(partarr[tidx, nidx, 0] * duration.quarterLength)
                note.volume.velocity = int(partarr[tidx, nidx, 1])  # set velocity
                notes.append(note)
                
            for g in self._group_notes_by_duration(notes):
                if len(g) == 1:
                    stream.insert(tidx * duration.quarterLength, g[0])
                else:
                    chord = music21.chord.Chord(g)
                    stream.insert(tidx * duration.quarterLength, chord)
        return stream
    
    def _group_notes_by_duration(self, notes):
        """
        Separate notes into chord groups.
        
        Args:
            notes: List of notes
            
        Returns:
            List of chord groups
        """
        keyfunc = lambda n: n.duration.quarterLength
        notes = sorted(notes, key=keyfunc)
        return [list(g) for k, g in groupby(notes, keyfunc)]
    
    # Stream processing methods
    def remove_overlaps(self, stream, separate_chords=True):
        """
        Separate overlapping notes to different tracks.
        
        Args:
            stream: music21.stream.Stream
            separate_chords: Whether to separate chords
            
        Returns:
            music21.stream.Stream
        """
        if not separate_chords:
            return stream.flat.makeVoices().voicesToParts()
        return self._separate_melody_chord(stream)
    
    def _separate_melody_chord(self, stream):
        """
        Separate notes and chords to different tracks.
        
        Args:
            stream: music21.stream.Stream
            
        Returns:
            music21.stream.Score
        """
        new_stream = music21.stream.Score()
        if stream.timeSignature: new_stream.append(stream.timeSignature)
        new_stream.append(stream.metronomeMarkBoundaries()[0][-1])
        if stream.keySignature: new_stream.append(stream.keySignature)
        
        melody_part = music21.stream.Part(stream.flat.getElementsByClass('Note'))
        melody_part.insert(0, stream.getInstrument())
        chord_part = music21.stream.Part(stream.flat.getElementsByClass('Chord'))
        chord_part.insert(0, stream.getInstrument())
        new_stream.append(melody_part)
        new_stream.append(chord_part)
        return new_stream


# Utility functions for multi-part processing
def stream2npenc_parts(stream, sort_pitch=True, encoder=None):
    """
    Convert stream to multiple numpy encodings, one per part.
    
    Args:
        stream: music21.stream.Stream
        sort_pitch: Whether to sort by pitch
        encoder: Encoder instance
        
    Returns:
        List of numpy encodings
    """
    if encoder is None:
        encoder = Encoder()
    chordarr = encoder._stream_to_chordarr(stream)
    _, num_parts, _ = chordarr.shape
    parts = [part_enc(chordarr, i, encoder) for i in range(num_parts)]
    return sorted(parts, key=avg_pitch, reverse=True) if sort_pitch else parts

def part_enc(chordarr, part, encoder=None):
    """
    Encode a single part from chord array.
    
    Args:
        chordarr: Chord array
        part: Part index
        encoder: Encoder instance
        
    Returns:
        Numpy encoding
    """
    if encoder is None:
        encoder = Encoder()
    partarr = chordarr[:, part:part+1, :]
    npenc = encoder._chordarr_to_npenc(partarr)
    return npenc

def chordarr_combine_parts(parts):
    """
    Combine multiple parts into a single chord array.
    
    Args:
        parts: List of parts
        
    Returns:
        Combined chord array
    """
    max_ts = max([p.shape[0] for p in parts])
    parts_padded = [pad_part_to(p, max_ts) for p in parts]
    chordarr_comb = np.concatenate(parts_padded, axis=1)
    return chordarr_comb

def pad_part_to(p, target_size):
    """
    Pad part to target size.
    
    Args:
        p: Part
        target_size: Target size
        
    Returns:
        Padded part
    """
    pad_width = ((0, target_size-p.shape[0]), (0, 0), (0, 0))
    return np.pad(p, pad_width, 'constant')

def avg_pitch(t, sep_idx=VALTSEP):
    """
    Calculate average pitch.
    
    Args:
        t: Numpy encoding
        sep_idx: Separator index
        
    Returns:
        Average pitch
    """
    return t[t[:, 0] > sep_idx][:, 0].mean()
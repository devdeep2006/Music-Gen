import tensorflow as tf
import numpy as np
import collections
import pandas as pd
import os
import warnings

if os.name == "nt":
    os.environ["PATH"] = ""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    import pretty_midi
except ImportError:
    raise ImportError("Please install pretty_midi using `pip install pretty_midi`")
def midi_to_notes(midi_file):
    """Extract note features (pitch, step, duration) from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_file)

    if not pm.instruments:
        raise ValueError("MIDI file contains no instruments.")
    instrument = pm.instruments[0]
    if len(instrument.notes) < 2:
         raise ValueError("MIDI file must contain at least 2 notes.")

    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start
    
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)
        notes["step"].append(start - prev_start)
        notes["duration"].append(end - start)
        prev_start = start
        
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(notes_df, out_file, instrument_name="Acoustic Grand Piano"):
    """Convert a pandas DataFrame of notes back into a MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )
    
    prev_start = 0
    for _, note in notes_df.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        pm_note = pretty_midi.Note(
            velocity=100, pitch=int(note["pitch"]), start=start, end=end
        )
        instrument.notes.append(pm_note)
        prev_start = start
        
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

def predict_next_note(notes, model, temperature=1.0):
    """Predict the next note given a sequence of notes."""
    assert temperature > 0

    inputs = tf.expand_dims(notes, 0)
    
    predictions = model.predict(inputs)
    pitch_logits = predictions["pitch"]
    step = predictions["step"]
    duration = predictions["duration"]
    
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    
    return int(pitch), float(step), float(duration)
def load_trained_model(model_path="./music_rnn_model.keras"):
    """Load a pre-trained RNN model saved in .keras format."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Make sure 'music_rnn_model.keras' is in the same directory.")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
    return model
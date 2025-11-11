import os
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import pretty_midi
import fluidsynth
from scipy.io.wavfile import write
import plotly.graph_objs as go
import plotly.express as px
from model_utils import midi_to_notes, notes_to_midi, predict_next_note, load_trained_model

st.set_page_config(page_title="Piano Music Generator", layout="wide")

st.title("Piano Music Generator ")
st.markdown("""
Experience generated piano compositions — listen, visualize, and explore their musical structure  
through **dynamic piano rolls, waveforms, and harmonic spectrograms.**
""")

uploaded_file = st.file_uploader("Upload a piano MIDI file", type=["mid", "midi"])
temperature = st.slider("Creativity Level (Temperature)", 0.5, 2.0, 1.0, 0.1)
num_notes = st.slider("Number of Notes to Generate", 50, 300, 120, 10)
vocab_size = 128
TARGET_SEQ_LENGTH = 25 
@st.cache_resource
def load_model():
    try:
        return load_trained_model("music_rnn_model.keras")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Please ensure 'music_rnn_model.keras' is in the same directory as 'app.py'.")
        return None

model = load_model()
def midi_to_wav(midi_path, wav_path, soundfont="soundfont.sf2"):
    if not os.path.exists(soundfont):
        st.warning(f"SoundFont not found at {soundfont}. WAV export will fail.")
        raise FileNotFoundError(f"SoundFont not found at {soundfont}")
    fs = fluidsynth.Synth()
    fs.start() 
    sfid = fs.sfload(soundfont)
    fs.program_select(0, sfid, 0, 0)
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        st.error(f"Error parsing MIDI file {midi_path}: {e}")
        fs.delete()
        return

    audio_data = pm.fluidsynth(fs=44100)
    fs.delete()
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    write(wav_path, 44100, audio_data_int16)

# Waveform Visualization
# -------------------------------
def display_waveform_plotly(wav_file):
    try:
        y, sr = librosa.load(wav_file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        time = np.linspace(0, duration, len(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=y, mode='lines', line=dict(color='mediumorchid', width=1)))
        fig.update_layout(title="Waveform Visualization", xaxis_title="Time (s)", yaxis_title="Amplitude",
                          template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error generating waveform: {e}")

# -------------------------------
# Spectrogram Visualization
# -------------------------------
def display_spectrogram(wav_file):
    try:
        y, sr = librosa.load(wav_file, sr=None)
        S = np.abs(librosa.stft(y))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr)
        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr)
        fig = go.Figure(data=go.Heatmap(z=S_db, x=times, y=freqs, colorscale='Magma', colorbar=dict(title="dB")))
        fig.update_layout(title="Spectrogram (Harmonic Content)", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                          template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Error generating spectrogram: {e}")

# -------------------------------
# Enhanced Animated Piano Roll
# -------------------------------
def display_animated_piano_roll(notes_df):
    import plotly.colors as pc
    color_scale = pc.sample_colorscale("Plasma", np.linspace(0, 1, len(notes_df)))
    fig = go.Figure()
    for i in range(len(notes_df)):
        dur = notes_df.iloc[i]['duration']
        color = color_scale[i]
        fig.add_trace(go.Bar(
            x=[dur], y=[notes_df.iloc[i]['pitch']], base=[notes_df.iloc[i]['start']],
            orientation='h', marker_color=color, marker_line=dict(color='white', width=0.5), showlegend=False
        ))
    fig.update_layout(title="Animated Piano Roll (Gradient-Colored Notes)",
                      xaxis_title="Time (s)", yaxis_title="Pitch (MIDI Number)",
                      yaxis=dict(autorange="reversed"), template="plotly_dark",
                      height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")
# -------------------------------
# Note Analytics
# -------------------------------
def note_analytics(notes_df):
    st.markdown("### Musical Analytics")
    fig1 = px.histogram(notes_df, x='pitch', nbins=30, title="Pitch Distribution", color_discrete_sequence=['plum'])
    fig2 = px.scatter(notes_df, x='start', y='pitch', color='duration',
                      title="Pitch vs Time", color_continuous_scale='Purples')
    fig1.update_layout(height=300, template="plotly_white")
    fig2.update_layout(height=300, template="plotly_white")
    st.plotly_chart(fig1, width="stretch")
    st.plotly_chart(fig2, width="stretch")
    avg_dur = notes_df['duration'].mean()
    avg_step = notes_df['step'].mean()
    st.info(f"🎼 **Average Step:** {avg_step:.3f}s | **Average Duration:** {avg_dur:.3f}s | **Notes Generated:** {len(notes_df)}")

if model and uploaded_file:
    st.audio(uploaded_file, format="audio/midi")
    
    try:
        with open("input.mid", "wb") as f:
            f.write(uploaded_file.getbuffer())

        notes = midi_to_notes("input.mid")
        st.success(f"MIDI loaded successfully with {len(notes)} notes.")

    except Exception as e:
        st.error(f"Error processing uploaded MIDI file: {e}")
        notes = None

    if notes is not None:
        if len(notes) < TARGET_SEQ_LENGTH:
            st.error(f"Uploaded file is too short. It must have at least {TARGET_SEQ_LENGTH} notes.")
        else:
            key_order = ['pitch', 'step', 'duration']
            sample_notes = np.stack([notes[key] for key in key_order], axis=1)
            input_notes = sample_notes[-TARGET_SEQ_LENGTH:]
            input_notes_normalized = input_notes / np.array([vocab_size, 1.0, 1.0])
            if st.button("🎶 Generate Continuation"):
                with st.spinner("Composing continuation based on your uploaded melody..."):
                    generated = []
                    prev_start = notes['start'].iloc[-1]
                    context_notes = input_notes_normalized.copy()

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(num_notes):
                        pitch, step, duration = predict_next_note(context_notes, model, temperature)
                        step = np.clip(step, 0.0, 2.0)
                        duration = np.clip(duration, 0.05, 2.0)
                        pitch = int(np.clip(pitch, 21, 108))
                        
                        start = prev_start + step
                        end = start + duration
                        
                        generated.append((pitch, step, duration, start, end))

                        new_note = np.array([[pitch / vocab_size, step, duration]])

                        context_notes = np.vstack([context_notes[1:], new_note])
                        
                        prev_start = start
                        
                        progress = (i + 1) / num_notes
                        progress_bar.progress(progress)
                        status_text.text(f"Generating notes... {i + 1}/{num_notes}")

                    progress_bar.empty()
                    status_text.empty()
                    
                    out_wav = "continued_output.wav"
                    out_midi = "continued_output.mid"
                    wav_success = False

                    with st.spinner("Converting to MIDI and audio format..."):
                        generated_df = pd.DataFrame(generated, columns=['pitch', 'step', 'duration', 'start', 'end'])
                        final_df = pd.concat([notes, generated_df], ignore_index=True)
                        notes_to_midi(final_df, out_midi)

                        try:
                            soundfont_path = os.path.join(os.getcwd(), "soundfont.sf2")
                            midi_to_wav(out_midi, out_wav, soundfont=soundfont_path)
                            wav_success = True
                        except Exception as e:
                            st.error(f"WAV conversion failed: {e}")
                            st.info("Ensure 'soundfont.sf2' is in the app directory. You can still download the MIDI file.")

                    st.success("Music continuation generated successfully!")
                    
                    if wav_success and os.path.exists(out_wav):
                        st.audio(out_wav, format="audio/wav")
                    
                    with open(out_midi, "rb") as midi_data:
                        st.download_button("⬇️ Download Continued MIDI",
                                           data=midi_data,
                                           file_name="Continued_Composition.mid")

                    st.markdown("## 🎼 Visualizations & Analysis")
                    tab1, tab2, tab3, tab4 = st.tabs(["🎹 Piano Roll", "📈 Waveform", "🌈 Spectrogram", "📊 Analytics"])
                    
                    with tab1:
                        display_animated_piano_roll(final_df)
                    
                    if wav_success:
                        with tab2:
                            display_waveform_plotly(out_wav)
                        with tab3:
                            display_spectrogram(out_wav)
                    else:
                        with tab2:
                            st.warning("Waveform not available (WAV conversion failed).")
                        with tab3:
                            st.warning("Spectrogram not available (WAV conversion failed).")

                    with tab4:
                        note_analytics(final_df)

                    with st.expander("View Generated Notes"):
                        st.dataframe(generated_df.head(15))
                    
                    with st.expander("Compare Original vs Generated Statistics"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Original Music:**")
                            st.write(f"Pitch range: {notes['pitch'].min():.0f} - {notes['pitch'].max():.0f}")
                            st.write(f"Avg step: {notes['step'].mean():.3f}s")
                            st.write(f"Avg duration: {notes['duration'].mean():.3f}s")
                        
                        with col2:
                            st.write("**Generated Music:**")
                            if not generated_df.empty:
                                st.write(f"Pitch range: {generated_df['pitch'].min():.0f} - {generated_df['pitch'].max():.0f}")
                                st.write(f"Avg step: {generated_df['step'].mean():.3f}s")
                                st.write(f"Avg duration: {generated_df['duration'].mean():.3f}s")
                            else:
                                st.write("No notes were generated.")

elif not model:
    st.warning("Model is not loaded. Please check the 'music_rnn_model.keras' file.")
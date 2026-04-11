# 🎹 Music-Gen — AI Piano Music Generator

A deep learning-powered piano music generation app built with **TensorFlow**, **Streamlit**, and an **RNN-based model**. Upload a MIDI file, and the model will learn from its musical patterns to compose a continuation — complete with interactive visualizations and audio playback.

---

## ✨ Features

- 🎵 **MIDI-based Music Generation** — Upload any piano MIDI file as a seed melody
- 🤖 **RNN Model (Keras)** — Trained recurrent neural network predicts pitch, step, and duration for each new note
- 🎛️ **Adjustable Creativity** — Control the generation temperature (0.5–2.0) for more conservative or experimental outputs
- 🔢 **Variable Output Length** — Generate between 50 and 300 notes
- 🎧 **Audio Playback** — Converts generated MIDI to WAV using FluidSynth for in-browser listening
- 📊 **Rich Visualizations** via Plotly:
  - Animated gradient-colored **Piano Roll**
  - **Waveform** display
  - **Spectrogram** (harmonic content)
  - **Musical Analytics** — pitch distribution, pitch-vs-time scatter plots, and summary statistics
- ⬇️ **Download** the generated MIDI file

---

## 🗂️ Project Structure

```
Music-Gen/
├── app.py                     # Main Streamlit application
├── model_utils.py             # MIDI processing & model inference utilities
├── music_rnn_model.keras      # Pre-trained RNN model
├── requirements.txt           # Python dependencies
├── input.mid                  # Sample input MIDI
├── generated_output.mid/.wav  # Example generated output
├── continued_output.mid/.wav  # Example continued output
├── model_dev/                 # Model development notebooks
├── training_checkpoints/      # Model training checkpoints
└── .streamlit/                # Streamlit configuration
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A SoundFont file (`soundfont.sf2`) for WAV audio export — place it in the project root directory. You can download a free one such as [GeneralUser GS](https://schristiancollins.com/generaluser.php).

### Installation

```bash
git clone https://github.com/devdeep2006/Music-Gen.git
cd Music-Gen
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

1. **Upload** a `.mid` / `.midi` piano file via the UI.
2. The app parses the MIDI into a sequence of notes (pitch, step, duration).
3. The last 25 notes are fed as context to the **pre-trained RNN model**.
4. The model autoregressively predicts the next note, building a continuation of the melody.
5. The original and generated notes are combined and exported to MIDI (and optionally WAV).
6. Interactive charts let you explore the musical structure of the full composition.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `tensorflow` | RNN model training & inference |
| `pretty_midi` | MIDI parsing and generation |
| `librosa` | Audio analysis (waveform, spectrogram) |
| `fluidsynth` | MIDI-to-WAV audio synthesis |
| `plotly` | Interactive visualizations |
| `numpy` / `pandas` | Numerical processing |
| `torch` | Additional deep learning support |
| `mido` | Low-level MIDI I/O |
| `scikit-learn` | Utility ML functions |
| `soundfile` | Audio file I/O |
| `matplotlib` | Supplementary plotting |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Notes

- **WAV export requires** a `soundfont.sf2` file in the project directory. If it's missing, MIDI download will still work but audio playback will be unavailable.
- The uploaded MIDI must contain **at least 25 notes** to provide sufficient context for the model.
- For best results, use **single-instrument piano MIDI files**.

---

## 📸 App Preview

The app features four analysis tabs after generation:

| Tab | Content |
|---|---|
| 🎹 Piano Roll | Animated, gradient-colored note timeline |
| 📈 Waveform | Time-domain amplitude plot |
| 🌈 Spectrogram | Frequency-domain harmonic content heatmap |
| 📊 Analytics | Pitch distribution & pitch-vs-time comparison |

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for bug fixes, new features, or model improvements.

---

## 📄 License

This project is open source. See the repository for details.

---

*Built with ❤️ using TensorFlow, Streamlit, and the magic of music.*

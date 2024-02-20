import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Function to analyze audio features
def analyze_audio(audio_file):
    y, sr = librosa.load(audio_file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroids = spectral_centroids / np.max(spectral_centroids)  # Normalize
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

    return y, sr, tempo, beats, chroma_stft, mfcc, rmse, onset_env, tempogram, spectral_centroids, tonnetz

# Main function to create a combined visualization
def main(audio_file):
    # Analyze audio features
    y, sr, tempo, beats, chroma_stft, mfcc, rmse, onset_env, tempogram, spectral_centroids, tonnetz = analyze_audio(audio_file)
    
    # Estimate key from tonnetz
    key = np.argmax(np.mean(tonnetz, axis=1))
    key_strength = np.max(np.mean(tonnetz, axis=1))
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = key_names[key]
    
    # Create canvas for painting
    plt.figure(figsize=(14, 14))
    
    # Plot beats as vertical lines
    plt.subplot(4, 2, 1)
    plt.vlines(beats, 0, 1, colors='r', alpha=0.5)
    plt.title('Beats')
    
    # Plot chromagram
    plt.subplot(4, 2, 2)
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chromagram')
    
    # Plot MFCC
    plt.subplot(4, 2, 3)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    
    # Plot RMS Energy
    plt.subplot(4, 2, 4)
    plt.semilogy(np.linspace(0, len(y) / sr, len(rmse[0])), rmse[0], label='RMS Energy')
    plt.xlim([0, len(y) / sr])
    plt.legend(loc='best')
    plt.title('RMS Energy')
    
    # Plot onset envelope
    plt.subplot(4, 2, 5)
    plt.plot(np.linspace(0, len(y) / sr, len(onset_env)), librosa.util.normalize(onset_env), label='Onset strength')
    plt.title('Onset Envelope')
    
    # Plot tempogram
    plt.subplot(4, 2, 6)
    librosa.display.specshow(tempogram, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.title('Tempogram')
    
    # Plot spectral centroids
    plt.subplot(4, 2, 7)
    plt.plot(np.linspace(0, len(y) / sr, len(spectral_centroids)), spectral_centroids, label='Spectral Centroids')
    plt.title('Spectral Centroids')
    
    # Plot key and key strength
    plt.subplot(4, 2, 8)
    plt.plot(np.mean(tonnetz, axis=1), label='Tonnetz')
    plt.title(f'Estimated Key: {estimated_key} (Strength: {key_strength})')
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Add title
    plt.suptitle('Audio Feature Analysis', fontsize=16)
    
    # Display the combined chart
    plt.show()

# Example usage
if __name__ == "__main__":
    audio_file = 'song.mp3'
    main(audio_file)

import torch
import torchaudio
import os
from main import LitModule # Import your LightningModule
from config.papez_study_libri2mix import config # Import config

# ======= CONFIGURE PATHS =======
checkpoint_path = "tb_logs/lightning_logs/version_13/checkpoints/epoch=99-step=4826000.ckpt" # Update this!
input_folder = "/home/yandex/APDL2425a/group_4/Papez/inference_examples/input_samples" # Folder containing input mixtures
output_folder = "inference_examples/separated_speakers" # Folder to save separated outputs
os.makedirs(output_folder, exist_ok=True)

# ======= LOAD THE TRAINED MODEL =======
print(f"Loading model from {checkpoint_path}...")
model = LitModule.load_from_checkpoint(checkpoint_path, config=config)
model.eval()
model.freeze()

# ======= HELPER FUNCTIONS =======
def normalize_audio(audio):
    """ Normalize audio to [-1, 1] range """
    max_val = torch.max(torch.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

# ======= PROCESS ALL AUDIO FILES IN THE FOLDER =======
audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

if not audio_files:
    print(f"No audio files found in {input_folder}.")
    exit()

print(f"Found {len(audio_files)} audio files. Processing...")

for file_name in audio_files:
    file_path = os.path.join(input_folder, file_name)
    print(f"\nProcessing: {file_name}")

    # Load the mixture audio file
    mixture, sample_rate = torchaudio.load(file_path)

    # Ensure correct sample rate
    assert sample_rate == config["sample_rate"], f"Expected {config['sample_rate']}Hz but got {sample_rate}Hz"

    # Convert stereo to mono if needed
    if mixture.shape[0] > 1:
        mixture = torch.mean(mixture, dim=0, keepdim=True)

    # Add batch dimension: (1, 1, time_steps)
    mixture = mixture.unsqueeze(0)

    # Run inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mixture = mixture.to(device)

    with torch.no_grad():
        separated_sources = model.model(mixture) # Shape: (1, num_speakers, time_steps)

    # Create sub-folder for this file
    file_output_folder = os.path.join(output_folder, os.path.splitext(file_name)[0])
    os.makedirs(file_output_folder, exist_ok=True)

    # Save each separated speaker
    for i in range(config["num_speakers"]):
        speaker_output = separated_sources[0, i].cpu()
        speaker_output = normalize_audio(speaker_output)
        output_path = os.path.join(file_output_folder, f"speaker_{i+1}.wav")
        torchaudio.save(output_path, speaker_output.unsqueeze(0), sample_rate)
        print(f"Saved: {output_path}")

print("\nBatch inference complete! Separated files are in:", output_folder)
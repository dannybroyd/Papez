import os

def rename_files(folder_path):
    files = sorted(os.listdir(folder_path))  # Sort to maintain order
    
    for index, filename in enumerate(files):
        # Check if the file has a .wav extension
        if filename.lower().endswith('.wav'):
            file_ext = os.path.splitext(filename)[1]  # Get file extension
            new_name = f"input_{index}{file_ext}"  # New file name
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_name}")
        else:
            print(f"Skipping non-wav file: {filename}")

# Use the current working directory (where the Python file is)
folder_path = os.getcwd()  # Gets the current working directory
rename_files(folder_path)
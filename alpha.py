import os
import subprocess
import threading
import time


def extract_alpha(input_file_path, output_file_path):
    if not os.path.exists(output_file_path):
        subprocess.run(
            ["magick", input_file_path, "-strip", "-alpha", "extract", output_file_path]
        )


current_dir = os.getcwd()
input_dir = os.path.join(current_dir, "clean")
output_dir = os.path.join(current_dir, "masks")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

threads = []

print("This is Pequod, arriving shortly at LZ to extract team Alpha!")

# Extracting alpha using ImageMagick
files = os.listdir(input_dir)
total_files = len(files)

for idx, file in enumerate(files, start=1):
    if idx % 20 == 0:
        print(f"Processing file {idx} out of {total_files}")
    if file.endswith(".png"):
        input_file_path = os.path.join(input_dir, file)
        output_file_path = os.path.join(output_dir, file)

        # Start a new thread to process the image
        t = threading.Thread(
            target=extract_alpha, args=(input_file_path, output_file_path)
        )
        threads.append(t)
        t.start()

        # Control the number of threads to prevent overwhelming the system.
        while threading.active_count() > 8:  # reduce as necessary!
            time.sleep(0.5)

# Wait for all threads to finish
for t in threads:
    t.join()

print("Masks extracted!")

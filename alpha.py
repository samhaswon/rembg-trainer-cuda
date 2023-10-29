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

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

threads = []

# Extract alpha using ImageMagick
for file in os.listdir(input_dir):
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
        # Here, we wait until we have only 10 threads active.
        while threading.active_count() > 10:  # 9 + the main thread
            time.sleep(0.5)

# Wait for all threads to finish
for t in threads:
    t.join()

print("Alpha extraction completed!")

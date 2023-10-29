#!/bin/bash

current_dir=$(pwd)
input_dir="${current_dir}/Clean"
output_dir="${current_dir}/Masks"

mkdir -p "$output_dir"

# Extract alpha using ImageMagick
for file in "${input_dir}"/*.png; do
    output_file="${output_dir}/$(basename "$file")"

    # Check if the output file already exists, if it does, skip processing.
    if [ ! -f "$output_file" ]; then
        magick "$file" -strip -alpha extract "$output_file" &
    fi

    # Control the number of background tasks to prevent overwhelming the system.
    # Here, we wait until we have only 10 background tasks running.
    while [ $(jobs -r | wc -l) -gt 8 ]; do
        sleep 0.5
    done
done

wait
echo "Alpha extraction completed!"

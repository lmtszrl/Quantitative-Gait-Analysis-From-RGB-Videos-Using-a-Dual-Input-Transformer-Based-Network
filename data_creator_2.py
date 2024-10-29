import os
import pickle
import numpy as np
import cv2

# Indices for the specific body parts (Left and Right)
LANK = 14
LKNE = 13
LHIP = 12
LBTO = 19

RANK = 11
RKNE = 10
RHIP = 9
RBTO = 22


# Function to read data from pickle and extract the required points for L and R
def extract_sequences_from_pickle(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    all_sequences = []

    for sequence in data:
        seq_id = sequence[0]
        num_frames = sequence[1]
        frame_data = sequence[2]  # This is the array of frame data

        extracted_sequence = []

        for frame in frame_data:
            # Extract the x and y coordinates for each L body part
            lank_x, lank_y = frame[2 * LANK], frame[2 * LANK + 1]
            lkne_x, lkne_y = frame[2 * LKNE], frame[2 * LKNE + 1]
            lhip_x, lhip_y = frame[2 * LHIP], frame[2 * LHIP + 1]

            # Extract the x and y coordinates for each R body part
            rank_x, rank_y = frame[2 * RANK], frame[2 * RANK + 1]
            rkne_x, rkne_y = frame[2 * RKNE], frame[2 * RKNE + 1]
            rhip_x, rhip_y = frame[2 * RHIP], frame[2 * RHIP + 1]

            # Append the extracted points to the result for this frame (L and R combined)
            extracted_sequence.append({
                'L': [lank_x, lank_y, lkne_x, lkne_y, lhip_x, lhip_y],
                'R': [rank_x, rank_y, rkne_x, rkne_y, rhip_x, rhip_y]
            })

        all_sequences.append((seq_id, num_frames, extracted_sequence))

    return all_sequences


# Function to normalize points to fit within a 128x128 square
def normalize_points_for_image(extracted_sequence, image_size=128):
    all_points = [point for frame in extracted_sequence for side in frame for point in frame[side]]
    min_val = np.min(all_points)  # Minimum value in the data
    max_val = np.max(all_points)  # Maximum value in the data

    normalized_sequence = []

    for frame in extracted_sequence:
        normalized_frame = {
            'L': [],
            'R': []
        }
        for side in ['L', 'R']:
            for i in range(0, len(frame[side]), 2):
                # Normalize x and y to fit in the range [0, image_size-1]
                norm_x = np.interp(frame[side][i], [min_val, max_val], [0, image_size - 1])
                norm_y = np.interp(frame[side][i + 1], [min_val, max_val], [0, image_size - 1])
                normalized_frame[side].append((int(norm_x), int(norm_y)))
        normalized_sequence.append(normalized_frame)

    return normalized_sequence


# Function to plot the points onto a 128x128 image
def plot_points_on_image(normalized_sequence, image_size=128, side='L'):
    # Create a blank black image
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    # Plot each frame's points on the image
    for frame in normalized_sequence:
        for point in frame[side]:
            x, y = point
            if 0 <= x < image_size and 0 <= y < image_size:
                image[y, x] = 255  # Set the pixel to white (255)

    return image


# Read and extract the sequences from the pickle file
all_extracted_data = extract_sequences_from_pickle('Data/all_processed_video_segments_doublesided_cleaned.pickle')

# Create folders for left and right images
os.makedirs('Data/Legs/Left_Images', exist_ok=True)
os.makedirs('Data/Legs/Right_Images', exist_ok=True)

# Process each sequence to generate images
for seq_id, num_frames, extracted_data in all_extracted_data:
    # Normalize the extracted data to fit within a 128x128 square
    normalized_data = normalize_points_for_image(extracted_data)
    print(normalized_data)
    # Convert the normalized data into 128x128 images for L and R
    image_L = plot_points_on_image(normalized_data, side='L')
    image_R = plot_points_on_image(normalized_data, side='R')

    # Save images in the corresponding folders
    cv2.imwrite(os.path.join('Data/Legs/Left_Images', f'{seq_id}_L.jpg'), image_L)
    cv2.imwrite(os.path.join('Data/Legs/Right_Images', f'{seq_id}_R.jpg'), image_R)

    # Display the images (optional)
    #cv2.imshow(f'Left Points Image - Seq {seq_id}', image_L)
    #cv2.imshow(f'Right Points Image - Seq {seq_id}', image_R)
    #cv2.waitKey(0)  # Wait for a key press to close the image windows

#cv2.destroyAllWindows()

import cv2 as cv
import numpy as np
import sys
import pafy
import os
import re

# Data design
# This programs creates data from video and then analyzes it
# The data created from the video will be called brightness data; even though it also contains the video's fps
# It is a numpy array structured like this: (fps, brightness_averages)
# fps is a single number (floating point); brightness_averages is an array of floating points
# brightness_averages takes the average over all pixel's brightness in a frame
# The n-th index corresponds to the n-th frame of the video


# Following numbers are based on:
# https://www.epilepsy.com/article/2014/3/shedding-light-photosensitivity-one-epilepsys-most-complex-conditions-0

# Treshhold to recognize a brightness change as significant enough (it will then count towards hz)
brightness_treshold = 20
# With how much hz flashings are categories as dangerous
hz_treshold = 3


def get_video_brightness(video_reference, save_data=True):
    """
	:param video_reference: A local path or a youtube link.
	:param save_data: If true it will save the data as .npy file in the ./data folder (default: True)
	:return: Numpy Array: [fps, [brightness_data]]
	"""

    # Deciding it it's a youtube link or a path
    if re.match("youtube.com", video_reference) is not None:
        try:
            url = video_reference
            vPafy = pafy.new(url)
            video = vPafy.getbest(preftype="webm")
            video_handler = cv.VideoCapture(video.url)
        except Exception as e:
            # A typical one might be an 403 not allowed. This one is video dependent
            print("An Error occured: ", e)
            exit()
    else:
        try:
            video_handler = cv.VideoCapture(video_reference)
        except Exception as e:
            print("An error occured: ", e)
            exit()

    # The brightness averages will be stored here. data[n] := average brightness of frame n
    data = []
    fps = video_handler.get(cv.CAP_PROP_FPS)


    # Keeping track of how many frames have been processed
    i = 0

    while 1 == 1:
        ret, frame = video_handler.read()

        # Break out when the end of the video is reached
        if not ret:
            break
        elif frame is None:
            print("An unexpected error occurred. No frame was supplied. The video file might not have been fully processed.")
            break

        """
        Processing the current single frame as follows:
		Scale down the picture (for faster processing)
		Unpacking the x- and y-coordinates of the picture to one index:
			[x_coordinate][y_coordinate][Blue, Green, Red] -> [pixel][Blue, Green, Red]
		Calculate the brightness for each pixel
		Take the average over the brightness of all pixels
		"""

        # This means to shrink the frame to an n-th of it's original size.
        scale_down_factor = 16
        (height, width, _) = frame.shape
        frame = cv.resize(frame, (width // scale_down_factor, height // scale_down_factor))

        # Will contain the brightness value for every pixel
        brightness_values = []

        unpacked_square = frame.reshape((height // scale_down_factor * width // scale_down_factor, 3))

        # open cv uses the order of Blue, Red, Green (BGR)
        for (blue, green, red) in unpacked_square:
            # based on https://en.wikipedia.org/wiki/Relative_luminance
            brightness_values.append(0.0722 * blue + 0.7152 * green + 0.2126 * red)

        # Take the average brightness of all pixels and save it
        average = np.average(brightness_values)
        data.append(average)

        # Mainly for feedback that the process is running and how far alonge it has come
        i += 1
        print("{} frames added".format(i))

    fps_and_brightness_average = np.array([fps, data])

    # Optionally saving the data (see function parameter)
    if save_data:
        store_data(fps_and_brightness_average)

    return fps_and_brightness_average


def store_data(data_to_save):
    """

	:param data_to_save: The data to save a .npy file
	:return: void
	"""

    # Checking for existing files for enumeration. Will take the largest number and add one on top of that. Will skip lower, skipped numbers
    data_folder_content = os.listdir("data")
    data_numbers = []
    for file in data_folder_content:
        match = re.match("data_(\d*).npy", file)
        if match is not None:
            data_numbers.append(int(match.group(1)))

    if data_numbers:
        largest_number = np.amax(data_numbers)
    else:
        # Enumeration starting with 0
        largest_number = 0

    np.save("data/data_{}".format(largest_number + 1), data_to_save)


def analyze_brightness_data(brightness_data):
    (fps, data) = brightness_data

    fps_rounded = round(fps)
    frame_count = len(data)

    # Avoiding an array out of bounds when the video doesn't end on an exact second
    last_frames_amount = frame_count % fps_rounded
    if last_frames_amount == 0:
        last_frames_amount = fps_rounded
    end_frame = frame_count - last_frames_amount

    current_frame = 0
    # was it getting darker or brighter? 0 - undefined; 1 - brighter; -1 - darker
    # needed to not count every monotone change as an hz; i.e. +20 to another +20 as 2 hz
    current_direction = 0

    dangerous_frames = []

    # approximately one second is being viewed here
    while current_frame < end_frame - fps_rounded:
        # initializes the frames of the seconds and set's the reference frame to the first one in this set
        current_view = data[current_frame:current_frame + fps_rounded]
        current_reference_frame = current_view[0]

        # keeps track of the measured hz
        hz = 0

        # keeps track of not overstepping the second
        i = 0

        # every frame in this second is being viewed here
        while i < fps_rounded:

            brightness_difference = current_reference_frame - current_view[i]
            if abs(brightness_difference) > brightness_treshold:
                if (brightness_difference // abs(brightness_difference)) == current_direction:
                    # If the brightness threshold is exceeded but in the same direction, just update the reference frame
                    current_reference_frame = current_view[i]
                else:
                    # If the brightness threshold is exceeded in another direction than previously count it as an hz and update the reference_frame
                    hz += 1
                    current_reference_frame = current_view[i]
                    # No direction of the light has been set; so it will assigne one if this condition is true
                    # It shouldn't become zero again; since it takes the direction from previous frame-second-sets
                    if current_direction == 0:
                        if (current_view[i] - current_reference_frame) > 0:
                            current_direction = 1
                        else:
                            current_direction = -1
                    # Changing directon to the opposite
                    else:
                        current_direction *= -1
            i += 1

        if hz >= hz_treshold:
            print("Warning for the frames {} - {} with {}".format(current_frame, current_frame + fps_rounded, hz))
            dangerous_frames.append(current_frame)
        current_frame += fps_rounded

    # parse to seconds; slightly inaccurate since the fps are rounded; more inaccurate with longer videos
    dangerous_seconds = [x / fps_rounded for x in dangerous_frames]
    return dangerous_seconds

if __name__ == "__main__":
    # If called from console analyze the first argument and save it's data
    brightness_data = get_video_brightness(sys.argv[1])
    dangerous_seconds = analyze_brightness_data(brightness_data)
    print(dangerous_seconds)

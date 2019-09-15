import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pafy


if __name__ == "__main__":
	video_link = sys.argv[1]
	vPafy = pafy.new(video_link)
	video = vPafy.getbest(preftype = "webm")
	video_handler = cv.VideoCapture(video.url)
	data = []
	i = 0

	while 1 == 1:
		ret, frame = video_handler.read()

		if frame is None or not ret:
			break

		(height, width, _) = frame.shape
		frame = cv.resize(frame, (width//16, height//16))

		first_square = frame

		brightness_values = []

		unpacked_square = first_square.reshape((height//16 * width//16, 3))

		for (blue, green, red) in unpacked_square:
			# based on https://en.wikipedia.org/wiki/Relative_luminance
			brightness_values.append(0.0722 * blue + 0.7152 * green + 0.2126 * red)

		print(brightness_values)
		average = np.average(brightness_values)
		print(average)

		data.append(average)
		i += 1 
		print("{} frames added".format(i))

	np.save("average", data)

	data = np.load("average.npy")

	fps = round(video_handler.get(cv.CAP_PROP_FPS))
	frame_count = len(data)

	start_frame = 0
	last_frames_amount = frame_count % fps
	if last_frames_amount == 0:
		last_frames_amount = fps

	end_frame = frame_count - last_frames_amount

	current_frame = start_frame
	# was it getting darker or brighter? 0 - undefined; 1 - brighter; -1 - darker
	current_direction = 0
	dangerous_frames = []


	while current_frame < end_frame - fps:
		current_view = data[current_frame:current_frame + fps]
		current_reference_frame = current_view[0]
		hz = 0
		i = 0
		while i < fps:
			brightness_difference = current_reference_frame - current_view[i]
			if abs(brightness_difference) > 20:
				if (brightness_difference // abs(brightness_difference)) == current_direction:
					current_reference_frame = current_view[i]
				else:
					hz += 1
					current_reference_frame = current_view[i]
					if current_direction == 0:
						if (current_view[i] - current_reference_frame) > 0:
							current_direction = 1
						else:
							current_direction = -1
					else:
						current_direction *= -1
			i += 1

		if hz >= 3:
			print("Warning for the frames {} - {} with {}".format(current_frame, current_frame + fps, hz))
			dangerous_frames.append(current_frame)
		current_frame += fps

	dangerous_seconds = [x / fps for x in dangerous_frames]
	print(dangerous_seconds) # somewhat inaccurate since the fps are rounded


	matplotlib.use('TkAgg')
	plt.plot(data)
	plt.xlabel("Frames")
	plt.show()
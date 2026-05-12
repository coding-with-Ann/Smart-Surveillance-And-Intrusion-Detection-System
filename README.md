# Smart Surveillance and Intrusion Detection

This project uses a webcam and OpenCV to detect movement, track moving objects, and raise an intrusion alert when an object enters a restricted zone.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python surveillance.py
```

Press `q` to quit.

## How it works

MOG2 background subtraction learns the usual background from the webcam feed. When something changes in the frame, MOG2 marks that area as foreground motion.

The motion mask is cleaned with erosion and dilation. Erosion removes small noisy dots, and dilation grows the remaining motion areas back into clearer shapes.

Contours are found from the cleaned mask. Very small contours below 500 pixels are ignored because they are usually noise. For the remaining contours, the script draws green bounding boxes and red centroid dots.

The object tracker compares centroids between frames using Euclidean distance. If a new centroid is within 50 pixels of an existing object, it keeps the same ID. Otherwise, it creates a new object ID.

The restricted zone is the center of the frame, from 25 percent to 75 percent of the width and height. If any tracked centroid enters this red rectangle, the script displays:

```text
INTRUSION DETECTED!
```

## Expected behavior

When the script starts, two windows should appear: the webcam feed and the motion mask.

Wave your hand to see a green box, red centroid, and object ID. Move into the red center zone to trigger the red intrusion alert.

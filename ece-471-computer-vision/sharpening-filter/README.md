# Sharpening Filter

Sharpens greyscale images using an approximation of Laplacian of Gaussian filter i.e. sharpening filter.

# Prerequisites

- matplotlib
- numpy
- cv2 (opencv-python)
- argparse

# Usage

Run using the default image; image of the pear.

```sh
python solution.py
```

Run using a different image:

```sh
python solution.py -i image.png
or
python solution.py --image image.png
```

Help:

```sh
python solution.py -h
```

# Behaviour

User will first see the unsharpened image. After closing the window, the user will see the sharpened image. After closing the window, the program will save the sharpened image then terminate.

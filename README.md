# Vehicle Counting with YOLOv4 and DeepSORT Tracking
This project is designed to count the number of vehicles that cross a particular line, which is also drawn by the user. The system uses YOLOv4 for vehicle detection and DeepSORT tracking for vehicle tracking.




https://user-images.githubusercontent.com/43640058/235049740-eaef019a-153f-419f-a29e-6ed195019cc9.mp4



## Installation
To use this project, you will need to install the following dependencies:

Python 3.x
TensorFlow 2.x
OpenCV
Numpy
To install these dependencies, you can use pip. Here's an example of how to install the dependencies:

    `pip install tensorflow opencv-python numpy`

## Usage
Before running the system, you will need to download the YOLOv4 weights file from the official repository. Save this file in the model_data directory.

To run the system, use the following command:


    `python drawline2.py --video /path/to/video/file.mp4`

You can also use a camera feed as the input by setting the --video parameter to 0.

When the system starts, you will be prompted to draw a line on the screen. Use your mouse to draw a line across the road where you want to count the vehicles. Once you have drawn the line, press any key to start the vehicle counting.

The system will track each vehicle as it crosses the line, and the total number of vehicles will be displayed on the screen.

## Acknowledgements
This project is based on the YOLOv4 object detection algorithm and the DeepSORT tracking algorithm. The implementation of YOLOv4 is based on the Darknet framework, and the DeepSORT tracking is implemented using the DeepSORT repository.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

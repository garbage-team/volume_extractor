# volume_extractor
Volume extractor for garbage fill rate measuring


## Dependencies
* Python 3.7 or 3.8
* tensorflow or tflite_runtime (depending on system)
* numpy
* scipy
* matplotlib
* opencv-python

## Guide: Installation and setup
1. Clone the repository.
2. Install libraries and dependencies, installation will vary depending on system. Tensorflow Lite (TFLite) is recommended for running on RPi.
3. Train a model or download ours, and place it inside the folder. Ensure you are running a TFLite model. Update the MODEL_PATH variable in main.py so that it matches your models path.
   * If you only have a Tensorflow model it can be converted using [save_to_tflite()](https://github.com/garbage-team/garbage_view/blob/67b5575ce4821a43bd88860c34d9fbcef89b8e03/src/main.py#L91) from the [garbage-view](https://github.com/garbage-team/garbage_view) repository.

## Guide: Setup garbage bin measurement in the lab
1. Ensure the RPi is connected to the Logitech camera on the tripod. The tripod height should be such as the USB cable to the RPi is almost fully stretched. Ensure that the power adapter is plugged in and that the RPi is running. 

2. On the host computer (that will be connected to the screen during the live demo), connect to the “scania-smartlab” network and run “VNC viewer”. Connect to the RPi using the IP and login credentials given in the document on Teams.

3. When connected to the RPi, start the terminal and navigate to the “/volume_extractor” folder (“cd / volume_extractor”). Then run the application using “python3 main.py”. 

4. After a few seconds an image of the garbage bin should appear, along with a fill rate. 
   * Ensure that the entire garbage bin fits horizontally, and that the bin is roughly centered vertically in the image. The view will update every 5-10 seconds. 
   * Make sure that the inside of the bin is as free from shadows as possible to enable better results. 

5. The program should now be running! A new measurment is presented around every 5-10 seconds depending on system. Note that accuracy is not great with this method, and +/- 10% error is to be expected. The program runs until closed using “ctrl + c” in the terminal. 

## Authors
* Jonas Jungåker
* Victor Hanefors

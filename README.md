# Drone Gesture Control Using Tensorflow Lite
Controlling a Drone using Pose Estimation in Python and Tensorflow Lite

# Demo

[![IMAGE ALT TEXT](http://img.youtube.com/vi/bVfHFXI65Vo/0.jpg)](http://www.youtube.com/watch?v=bVfHFXI65Vo)

# Architecture

The project is organized to run as a client-server architecture

![image](https://user-images.githubusercontent.com/6872080/118691542-30d43f80-b7d7-11eb-8813-628afedcd472.png)

Server:

Jetson Nano with the model running on top of it acts as a server and sends the identified gestures to the client.

Client:

A Laptop acted as a client with drone controller script and Mission Planner running on it. 
Drone controller script accepted the stream of identified gestures and translated them into control commands for the drone. 

# Code

The repository has a two main jupyter notebooks in `./production` folder.
1. `static-gesture-recognition-server` - this has all code needed to recognize gestures
   - This should be run on `Jetson Nano`
2. `client_control` - this has all code needed to convert recognized gestures into control signals for the drone
   - This should be run on your laptop

Each notebook is documented in itself. Start with the server notebook to understand the flow.

# Report

There is also an overall report available to give an overview of the project [here](https://github.com/abhijeetdtu/drone_control_gesture/blob/master/report.MD)

PDF version is in `./docs/`

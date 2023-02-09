# Object Tracking for video files with SageMaker and GStreamer

Process multiple **video files** with ML models, using [SageMaker Processing Jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html) and [GStreamer](https://gstreamer.freedesktop.org/) in batch mode. 

In this tutorial you'll learn how to create a people pathing mechanism that tracks people on video files in batch mode. The output is a set of Numpy files with the predictions of each frame, which contain:  
 - bouding boxes for each person;
 - keypoints for pose estimation;
 - id of each detecte person across the frames;


### Notebooks:
 - [01_Yolov7SageMakerInferentia.ipynb]: First deploy a real-time endpoint on SageMaker on an Inferentia (inf1) instance, with the Object Detection & Pose estimation model.
 - [02_CVPipeline.ipynb]: Launch a SageMaker Processing Job with a Python script that defines a GStreamer pipeline that processes multiple files at once by sending each frame to the endpoint and saving the predictions as Numpy files.


### Activities
  - First upload some **.mp4** to an S3 bucket.
  - Run notebook 01: 1/ follow the instructions there to compile a Yolov7 for Inferentia; 2/ deploy the compiled model to an enpoint
  - Run notebook 02: Prepare a python application that will be executed by SageMaker to read the .mp4 files and get the predictions.
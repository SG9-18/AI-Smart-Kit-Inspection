# AI-Smart-Kit-Inspection System
Real-time AI-based inspection system for automated detection and validation of automotive components using YOLOv8, OpenCV, Flask, and Roboflow.

## Overview

The system uses a camera to detect automotive components placed on a trolley and compares the detected parts against the expected kit. It provides real-time **OK / NOT OK** inspection results through a Flask-based dashboard.

## Key Features

- Real-time component detection using YOLOv8
- Detection of 50+ automotive components
- Custom object detection model trained on 1,500+ annotated images
- Roboflow-based dataset annotation and model training
- OpenCV-based live camera processing
- Automated kit validation
- Flask-based monitoring dashboard
- Real-time inspection status

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Object Detection | YOLOv8 |
| Computer Vision | OpenCV |
| Dataset / Annotation | Roboflow |
| Web Framework | Flask |
| Interface | HTML, CSS, JavaScript |
| Development | VS Code |

## System Workflow

```text
Camera
   ↓
OpenCV Video Capture
   ↓
YOLOv8 Object Detection
   ↓
Component Identification
   ↓
Kit Validation
   ↓
OK / NOT OK
   ↓
Flask Dashboard

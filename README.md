# Air Drawing AI 🎨

A real-time hand gesture based drawing application built using Python, OpenCV, and Computer Vision.

This project allows users to draw on a virtual canvas using hand movements without touching the screen. By tracking hand landmarks through a webcam feed, the application creates an interactive air-drawing experience similar to a digital whiteboard.

---

## Features

* ✋ Real-time hand tracking
* 🎨 Draw in the air using finger gestures
* 🌈 Multiple color selection options
* 🧽 Eraser functionality
* 📷 Webcam-based interaction
* ⚡ Smooth and responsive drawing experience
* 🖥️ Interactive virtual whiteboard interface

---

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy

---

## How It Works

1. The webcam captures live video input.
2. MediaPipe detects and tracks hand landmarks.
3. Specific finger gestures are interpreted as drawing actions.
4. Movement of the index finger is mapped onto a virtual canvas.
5. Users can switch colors, draw, and erase using gestures.

---

## Learning Outcomes

Through this project I explored:

* Computer Vision fundamentals
* Real-time hand tracking
* Gesture recognition systems
* OpenCV image processing
* Interactive Human-Computer Interaction (HCI)

---

## Future Improvements

* AI-powered shape recognition
* Save drawings automatically
* Handwriting-to-text conversion
* Multi-hand support
* Gesture customization
* Virtual classroom whiteboard features

---

## Deploy to Cloudflare Pages

1. Push this project to a Git provider (GitHub, GitLab, Bitbucket).
2. Open Cloudflare Pages and create a new project.
3. Connect the repository and select the `main` branch.
4. Use these settings:
   - Framework preset: `None`
   - Build command: *(leave empty)*
   - Build output directory: `.`
5. If this repo is part of a larger repository, set the root directory to `air-drawing-ai`.
6. Save and deploy.

Your live site will serve `index.html` from the project root.

> Note: The app page is available at `/app.html` once deployed.

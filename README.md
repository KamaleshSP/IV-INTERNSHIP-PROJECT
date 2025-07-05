# IV-INTERNSHIP-PROJECT

# AI-Based Student Attentiveness Monitoring System using ML/DL

A real-time, intelligent attentiveness monitoring solution built with Python. This project uses computer vision, facial landmark analysis, and deep learning logic to track and respond to student attention levels during online sessions. Designed with modularity and practicality in mind, it aims to support smart education environments.

> **Project Completed During Internship at [Retech Solutions Pvt Ltd](https://www.retechsolutions.in)**  
> **Internship Duration:** 16th June 2025 – 30th June 2025  
> **Tutorial Video:** See [`tutorial.mp4`](./tutorial.mp4) inside this project folder for a full demo walkthrough.
> To watch the video, open the project folder in your system’s file explorer and double-click the video file.

---

## Project Objective

To develop a **student attentiveness detection system** that:
- Detects signs of drowsiness, yawning, and absence
- Responds with voice alerts, screen flashes, and sirens
- Logs all critical events with timestamped entries
- Runs offline using local ML logic (no cloud/internet dependencies)

---

## Core Features

- **Active State Detection:** Eyes open, face aligned
- **Yawning Detection:** Mouth Aspect Ratio (MAR) based
- **Drowsiness Detection:** Eye Aspect Ratio (EAR) based microsleep check
- **Multiple Face Handling:** Disables monitoring if more than one person is detected
- **Face Missing Detection:** Flags temporary absence
- **Emergency Wake-Up Protocol** *(when inactive > 5 sec)*:
  - Siren sound using `playsound`
  - Red/blue flashing GUI screen
  - Voice warning with `pyttsx3`
  - Log entry with timestamp using `csv`

---

## Tech Stack

| Component            | Technology/Library           |
|----------------------|------------------------------|
| Programming Language | Python                       |
| GUI Framework        | PyQt5                        |
| Face Detection       | OpenCV + MediaPipe FaceMesh  |
| EAR/MAR Logic        | Custom landmark-based        |
| Audio Feedback       | pyttsx3                      |
| Alarm System         | playsound                    |
| Data Logging         | csv + datetime               |

---

## Project Structure

project-root/
│
├── master_controller_gui.py # Main GUI controller
│
├── utils/
│ ├── __pycache__ # stores Python’s temporary compiled files
│ ├── activity_logger.py # CSV logging
│ ├── ai_feedback.py # TTS voice feedback
│ ├── eye_tracking.py # EAR-based drowsiness detection
│ ├── yawn_detection.py # MAR-based yawning detection
│ ├── head_turn.py # Head movement monitoring
│ ├── multiple_faces.py # Multi-face detection
│ ├── emergency_wakeup.py # Flash + siren trigger logic
│ ├── face_presence.py # Active/inactive state logic
│
├── .gitignore # tells Git to skip tracking such unnecessary files
├── tutorial.mp4 # Full demo of the project
├── requirements.txt # All required libraries
├── README.md # Detailed report
│
└── attentiveness_log.csv # Timestamped event log

---

## Included Modules (Imports Used)

- `cv2`, `numpy`, `pyqt5`, `mediapipe`, `math`, `scipy`
- `pygame`, `threading`, `time`, `os`, `pyttsx3`, `queue`
- `csv`, `datetime`

---

## Install required packages:

See [`requirements.txt`](./requirements.txt) and install it.

## Acknowledgements
> **Retech Solutions Pvt Ltd – For the internship opportunity and guidance(https://www.retechsolutions.in)**
> **Rajalakshmi Engineering College – For academic support and mentorship(https://www.rajalakshmi.org)**

## License

This project is for academic and research purposes only. Feel free to fork, contribute, or build upon it with proper credits.

---

### That's a wrap!

Thank you for checking out my project!  
I truly enjoyed building this system during my internship and solving real-world problems using AI, Python, and a touch of creativity.

If you'd like to see more cool projects, connect, or collaborate:

[Visit My GitHub Profile](https://github.com/SP-Kamalesh)
[Visit My LinkedIn Profile](https://www.linkedin.com/in/kamalesh-sp)

Let’s keep innovating, one line of code at a time.  
**– Kamalesh S P**

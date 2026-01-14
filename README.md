# AI-Driven Smart Traffic Management System 🚗🚦

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehical-detection.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)](https://github.com/ultralytics/ultralytics)

A real-time computer vision system that detects, tracks, and counts vehicles in video feeds. This project utilizes Deep Learning (**YOLOv8**) for accurate object detection and **BoT-SORT** for multi-object tracking, integrated with a full-stack data dashboard.

## 🔗 Live Demo
**View the Analytics Dashboard here:** [https://vehical-detection.streamlit.app](https://vehical-detection.streamlit.app)  
*(Note: The cloud dashboard visualizes historical traffic data. For real-time detection, the system runs locally.)*

---

## 📌 Project Overview
This system is designed to automate traffic analysis. It processes video footage to identify vehicle types (Car, Truck, Bus, Motorbike), estimates their speed, and logs the data into a structured database for historical analysis.

### 🚀 Key Features
* **Real-Time Detection:** Uses **YOLOv8n** to classify vehicles with high accuracy.
* **Object Tracking:** Implements **BoT-SORT** to track unique vehicles across frames (prevents double counting).
* **Speed Estimation:** Calculates approximate vehicle speed based on pixel displacement over time.
* **Data Persistence:** Automatically logs `Vehicle ID`, `Type`, `Speed`, and `Timestamp` into an **SQLite** database.
* **Interactive Dashboard:** A **Streamlit** web app that visualizes traffic trends, peak hours, and vehicle classification distributions.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Computer Vision:** OpenCV, Ultralytics YOLOv8
* **Data Engineering:** SQLite, Pandas
* **Visualization:** Streamlit
* **Deployment:** Streamlit Community Cloud

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/Devansh-AIML/Vehicle_detection.git](https://github.com/Devansh-AIML/Vehicle_detection.git)
cd Vehicle_detection
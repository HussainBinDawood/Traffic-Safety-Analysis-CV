# Vision-Based Pedestrian Safety Analysis using YOLOv8 🚦🚶‍♂️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

## 📌 Project Overview
This repository contains the source code for my Senior Design Project (SDP) at **Imam Abdulrahman Bin Faisal University**. 

The project aims to modernize **Traffic Safety Analysis** by automating the extraction of **Surrogate Safety Measures (SSMs)** from video data. Using Deep Learning (**YOLOv8**) and Computer Vision techniques, the system tracks interactions between vehicles and pedestrians at unsignalized mid-block crosswalks to quantify collision risks without waiting for historical crash data.

## 🚀 Key Features
* **Object Detection & Tracking:** Utilizes `YOLOv8` and `ByteTrack` for robust multi-object tracking of pedestrians and vehicles.
* **Zone-Based Analysis:** Defines dynamic polygons to classify pedestrian behaviors (Waiting, Crossing, Jaywalking).
* **Automated Speed Estimation:** Calculates vehicle and pedestrian speeds using pixel-to-meter calibration grids.
* **Risk Assessment:** Computes critical safety metrics including:
    * **TTA (Time-to-Accident):** Proximity risk calculation.
    * **Gap Acceptance:** Analyzing pedestrian decision-making against oncoming vehicle gaps.
    * **Traffic Conflicts:** Identifying near-miss scenarios.

## 🛠️ Technical Stack
* **Core Logic:** Python
* **Computer Vision:** OpenCV, Supervision
* **AI Model:** Ultralytics YOLOv8 (Custom trained weights)
* **Data Processing:** NumPy, Pandas

## 📊 Methodology
The system processes video feeds through the following pipeline:
1.  **Frame Pre-processing:** ROI cropping and perspective calibration.
2.  **Inference:** Detecting agents (Pedestrians/Vehicles) using custom `.pt` models.
3.  **Trajectory Tracking:** Assigning unique IDs to agents across frames.
4.  **Logic Engine:**
    * Mapping coordinates to real-world meters.
    * Detecting lane encroachments.
    * calculating `Delta-V` and `Distance-Gap`.
5.  **Reporting:** Exporting a detailed `.csv` report for traffic engineering analysis.

## 📂 Output Sample
The script generates a comprehensive CSV report containing:
- `Ped_ID` & `Veh_ID`
- `Veh_Headway` (sec)
- `Veh_Dist_Gap` (m)
- `Ped_Decision` (Accepted/Rejected gap)
- `Ped_Wait_Time` (sec)

## 👨‍💻 Author
**Hussain Bin Dawood** *Transportation & Traffic Engineer | AI & Computer Vision Researcher* [LinkedIn Profile](ADD_YOUR_LINKEDIN_URL_HERE)

---
*Note: This code is part of an academic research project and is calibrated for specific camera angles used in the study.*

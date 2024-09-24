# Indoor Object Detection Device for the Visually Impaired

Led a team of four to develop a device for the Philippine National School for the Blind in Pasay City. The device, built using a Raspberry Pi 4, Python, TensorFlow Lite, and OpenCV, was designed to aid visually impaired individuals by detecting indoor household objects.

## Features

-   **Object Detection**: Identified indoor household objects using a Raspberry Pi Camera Module 2.
-   **Adjustable Detection Modes**:
    -   Auto Mode: Automatically detected objects at intervals (10, 20, or 30 seconds).
    -   Manual Mode: Allowed manual detection and could mute the device when auto mode was noisy.
-   **User Controls**:
    -   Five push buttons to switch detection modes and adjust detection parameters.
    -   Detection quantity could be set to detect a maximum of 5 or 10 objects simultaneously.
    -   Probability threshold settings: low (25%), medium (50%, default), high (75%).
-   **Audio Feedback**: Provided audio notifications through wired earphones using the pyttsx3 library.

## Limitations

-   **Field of View**: The camera had a limited front viewing angle and did not offer 360-degree coverage, magnification, or zoom capabilities.
-   **No Custom Model**: The device did not utilize a custom-trained TensorFlow Lite model due to time and resource constraints.
-   **No Thermal Imaging**: The device did not include thermal imaging to detect surface temperatures.
-   **Distance and Size**: The device did not provide information about the distance, size, or speed of detected objects.
-   **Detection Interference**: The study did not evaluate potential detection interference from printed media.
-   **Testing Environment**: The field experiment was conducted in indoor environments only, with a limited set of object classes (34).

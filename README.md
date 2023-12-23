# Employee_Face_Recogination


 Employee Face Recognition Attendance System is designed to streamline attendance management using face recognition technology. The system allows users to add, list, and delete employees, and it provides real-time attendance tracking through a webcam. The application stores attendance records in CSV files and maintains employee details, including images and relevant information, in dedicated directories.

Features
Face Recognition: Utilizes OpenCV and scikit-learn's K-Nearest Neighbors algorithm for face detection and recognition.
Real-time Attendance Tracking: Captures attendance through a webcam, recording the date and time of each recognized face.
User Management: Allows adding, listing, and deleting employees, with details and images stored for each user.
Employee Details: Maintains individual employee details, including joining date, attendance history, and present status on the current day.
Web Interface: Implements a user-friendly web interface using Flask for easy interaction and navigation.

Usage
Visit the web interface to manage employees, view attendance records, and start real-time attendance tracking.
Ensure that the required directories (static/faces, Employee_Details, Attendance) are present before running the application.

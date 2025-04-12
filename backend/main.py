from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from passlib.context import CryptContext
from typing import List, Dict, Any
import os
import json
import csv
import pandas as pd
from pydantic import BaseModel
import face_recognition
import numpy as np
import cv2
import time
import asyncio
from pymongo.server_api import ServerApi
import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import certifi
from bson.objectid import ObjectId  # Added for proper ObjectId handling

ca = certifi.where()

load_dotenv()
app = FastAPI()
app.state.processtime = 0

# MongoDB Connection
uri = os.getenv("MONGODB_URI")
MONGODB = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("connected successfully")
# MongoDB connection
db = MONGODB["AttendanceSystem"]
teachers = db["teachers"]
students = db["students"]
videos = db["videos"]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Local storage paths
UPLOAD_DIR = "uploads"
DATA_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "student_images"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "attendance"), exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to CLARA"}

class TeacherLogin(BaseModel):
    username: str
    password: str

class TeacherRegister(BaseModel):
    id: str
    name: str
    password: str
    classes: List[str]

async def create_attendance_files(teacher_id: str, teacher_name: str, classes: List[str]):
    """Create local CSV files for attendance tracking."""
    teacher_dir = os.path.join(DATA_DIR, "attendance", f"{teacher_id}_{teacher_name}")
    os.makedirs(teacher_dir, exist_ok=True)
    
    batch_offsets = {'e': 0, 'f': 23, 'g': 46, 'h': 69}
    
    for cls in classes:
        try:
            if len(cls) < 3:
                print(f"Invalid class format: {cls}")
                continue
                
            grade, batch, section = cls[0], cls[1].lower(), cls[2]
            offset = batch_offsets.get(batch)
            if offset is None:
                print(f"Unknown batch letter in class {cls}")
                continue
                
            # Create CSV file for the class
            csv_path = os.path.join(teacher_dir, f"{cls}.csv")
            
            # Generate header and roll numbers
            header = ["Roll number"]
            roll_numbers = []
            
            for i in range(1, 24):
                roll_numbers.append([f"{grade}1{section}{offset + i:02d}"])
                
            # Write to CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(roll_numbers)
                
            print(f"Created attendance file for class {cls}")
        except Exception as e:
            print(f"Error processing class {cls}: {e}")

@app.post("/teacher")
async def register_teacher(teacher: TeacherRegister):
    try:
        if teachers.find_one({"teacher_id": teacher.id}):
            raise HTTPException(400, "Teacher already registered")
            
        teachers.insert_one({
            "teacher_id": teacher.id,
            "name": teacher.name,
            "password": pwd_context.hash(teacher.password),
            "classes": teacher.classes
        })
        
        # Create local attendance files instead of spreadsheets
        await create_attendance_files(teacher.id, teacher.name, teacher.classes)
        
        return {"message": "Teacher registered successfully", "local_data_path": f"{DATA_DIR}/attendance/{teacher.id}_{teacher.name}"}
    except Exception as e:
        print(f"Error in register_teacher: {e}")
        raise HTTPException(500, f"Internal Server Error: {e}")

@app.post("/login")
async def login(teacher: TeacherLogin):
    print(teacher)
    db_teacher = teachers.find_one({"name": teacher.username})
    if not db_teacher:
        raise HTTPException(status_code=400, detail="Invalid username")
    if not pwd_context.verify(teacher.password, db_teacher["password"]):
        raise HTTPException(status_code=400, detail="Invalid password")
    return {"message": "Login successful", "id": db_teacher["teacher_id"], "name": db_teacher["name"]}

@app.post("/student")
async def register_student(
    name: str = Form(...),
    roll: str = Form(...),
    batch: str = Form(...),
    email: str = Form(...),
    images: List[UploadFile] = File(...)
):
    if students.find_one({"roll": roll}):
        raise HTTPException(status_code=400, detail="Student already registered")

    image_paths = []
    encodings = []
    
    for image in images:
        image_path = os.path.join(UPLOAD_DIR, "student_images", f"{roll}_{image.filename}")
        with open(image_path, "wb") as f:
            f.write(await image.read())
        image_paths.append(image_path)

        try:
            img = face_recognition.load_image_file(image_path)
            encoding_list = face_recognition.face_encodings(img)
            if encoding_list:
                encodings.append(encoding_list[0].tolist())
            else:
                print(f"Warning: No face detected in image {image_path}")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    if not encodings:
        raise HTTPException(status_code=400, detail="No valid face encodings could be extracted from the images")

    student_data = {
        "name": name,
        "roll": roll,
        "batch": batch,
        "image_paths": image_paths,
        "Email": email,
        "encodings": encodings
    }
    students.insert_one(student_data)
    return {"message": "Student registered successfully"}

@app.post("/video")
async def upload_video(
    id: str = Form(...),
    batch: str = Form(...),
    video: UploadFile = File(...)
):
    # Verify teacher exists
    teacher = teachers.find_one({"teacher_id": id})
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")

    # Save video to disk
    video_path = os.path.join(UPLOAD_DIR, "videos", f"{id}_{batch}_{video.filename}")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Store video metadata in MongoDB
    video_data = {
        "teacher_id": id,
        "batch": batch,
        "video_path": video_path,
        "status": "pending",
        "upload_time": datetime.datetime.now()
    }
    print(video_data,"<-video data")
    result = db["videos"].insert_one(video_data)
    print("result:->",result)

    return {"message": "Video uploaded successfully", "video_id": str(result.inserted_id)}

@app.post("/flow")
async def process_videos(background_tasks: BackgroundTasks):
    scan_videos = list(videos.find({"status": "pending"}))
    if not scan_videos:
        raise HTTPException(status_code=404, detail="No videos found")
    
    processed_count = 0
    for video_data in scan_videos:
        background_tasks.add_task(
            process_video_task, 
            video_data["teacher_id"], 
            video_data["video_path"], 
            video_data["batch"], 
            video_data["_id"]
        )
        videos.update_one({"_id": video_data["_id"]}, {"$set": {"status": "processing"}})
        processed_count += 1

    return {"message": f"Video processing started for {processed_count} video(s)."}

async def process_video_task(teacher_id, video_path, batch, video_id):
    try:
        start_time = time.time()
        
        # Ensure video_id is ObjectId (if it's not already)
        if not isinstance(video_id, ObjectId):
            video_id = ObjectId(video_id)
        
        # Load student encodings
        batch_students = list(students.find({"batch": batch}))
        print(f"Found {len(batch_students)} students in batch {batch}")
        
        known_face_encodings = []
        known_face_names = []
        student_emails = {}  # Store emails for later use
        detected_students = set()

        # Get encodings for all students in the batch
        for student in batch_students:
            encodings = student.get("encodings", [])
            if encodings:
                for enc in encodings:
                    known_face_encodings.append(np.array(enc))
                    known_face_names.append(student["roll"])
                # Store email for notifications
                if "Email" in student and student["Email"]:
                    student_emails[student["roll"]] = student["Email"]

        if not known_face_encodings:
            print(f"No encodings found for batch {batch}")
            videos.update_one({"_id": video_id}, {"$set": {"status": "failed", "error": "No student encodings found"}})
            return False

        # Verify video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Process video
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
            
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0 or fps <= 0:
            raise Exception(f"Invalid video properties: frames={total_frames}, fps={fps}")
        
        # Adjust sampling rate based on video length
        frames_per_second = 5  # Target frames to process per second of video
        video_duration = total_frames / max(fps, 1)
        max_frames_to_process = max(int(video_duration * frames_per_second), 20)
        
        processed_frames = 0
        print(f"Starting video: {video_path}, FPS: {fps}, Total Frames: {total_frames}")

        try:
            while video_capture.isOpened() and processed_frames < min(max_frames_to_process, total_frames):
                # Sample frames at regular intervals
                target_frame = int(total_frames * (processed_frames / max_frames_to_process))
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = video_capture.read()
                
                if not ret:
                    print(f"Failed to read frame {target_frame}")
                    break

                # Optimize frame processing
                frame = cv2.resize(frame, (1600, 900))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                print(f"Processing frame {target_frame}/{total_frames} ({processed_frames}/{max_frames_to_process})")
                frame_start = time.time()

                # Face detection and recognition
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        # Stop processing if we've already detected all students
                        if len(detected_students) == len(set(known_face_names)):
                            break
                        
                        # Compare with known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            roll = known_face_names[first_match_index]
                            if roll not in detected_students:
                                detected_students.add(roll)
                                print(f"{roll} detected in {time.time() - frame_start:.2f}s")
                
                processed_frames += 1
        except Exception as frame_error:
            print(f"Error processing frame: {frame_error}")
            # Continue with detected students so far
        finally:
            # Always release video capture
            video_capture.release()

        # Clean up video file
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Removed video file: {video_path}")
        except Exception as file_error:
            print(f"Warning: Could not remove video file: {file_error}")
            
        # Record processing time
        total_time = time.time() - start_time
        app.state.processtime += total_time
        
        # Update attendance records
        attendance_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Update attendance CSV and handle errors properly
        try:
            update_result = update_attendance_csv(teacher_id, batch, attendance_date, list(detected_students))
            if not update_result:
                print(f"Warning: Failed to update attendance CSV for batch {batch}")
        except Exception as csv_error:
            print(f"Error updating attendance CSV: {csv_error}")
            # Continue execution - don't fail the whole process for CSV errors
        
        print(f"Finished video for batch {batch} in {total_time:.2f}s. Detected: {detected_students}")
        
        # Update database status
        try:
            videos.update_one({"_id": video_id}, 
                             {"$set": {"status": "completed", 
                                      "processing_time": total_time,
                                      "detected_students": list(detected_students)}})
        except Exception as db_error:
            print(f"Error updating video status: {db_error}")
        
        # Send emails to absent students - handle errors properly
        try:
            await send_attendance_emails(teacher_id, batch, attendance_date, list(detected_students))
        except Exception as email_error:
            print(f"Error sending attendance emails: {email_error}")
        
        return True
    
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        try:
            videos.update_one({"_id": video_id}, {"$set": {"status": "failed", "error": str(e)}})
        except Exception as update_error:
            print(f"Failed to update error status: {update_error}")
        return False

def update_attendance_csv(teacher_id, batch, date, present_students):
    try:
        # Find teacher details
        teacher = teachers.find_one({"teacher_id": teacher_id})
        if not teacher:
            print(f"Teacher not found: {teacher_id}")
            return False
        
        # Path to attendance CSV
        teacher_dir = os.path.join(DATA_DIR, "attendance", f"{teacher_id}_{teacher['name']}")
        csv_path = os.path.join(teacher_dir, f"{batch}.csv")
        
        # Create directory if it doesn't exist
        if not os.path.exists(teacher_dir):
            os.makedirs(teacher_dir)
        
        # Create CSV if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Roll number"])
                
                # Get all students in the batch and add them to the CSV
                batch_students = list(students.find({"batch": batch}))
                for student in batch_students:
                    writer.writerow([student["roll"]])
        
        # Load existing CSV
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            # Handle empty CSV
            df = pd.DataFrame(columns=["Roll number"])
        except Exception as read_error:
            print(f"Error reading CSV {csv_path}: {read_error}")
            # Create new DataFrame
            df = pd.DataFrame(columns=["Roll number"])
        
        # Add new date column if it doesn't exist
        if date not in df.columns:
            df[date] = "Absent"
        
        # Add new roll numbers if they don't exist
        for roll in present_students:
            if roll not in df["Roll number"].values:
                df.loc[len(df)] = {"Roll number": roll}
        
        # Update attendance status
        for roll in present_students:
            df.loc[df["Roll number"] == roll, date] = "Present"
        
        # Save updated CSV
        try:
            df.to_csv(csv_path, index=False)
            print(f"Attendance updated for {teacher_id} in batch {batch} for {date}")
            return True
        except Exception as write_error:
            print(f"Error writing CSV {csv_path}: {write_error}")
            return False
        
    except Exception as e:
        print(f"Error updating attendance: {e}")
        return False

async def send_attendance_emails(teacher_id, batch, date, present_students):
    try:
        # Get all students in the batch
        batch_students = list(students.find({"batch": batch}))
        
        # Find absent students
        absent_students = []
        for student in batch_students:
            if student["roll"] not in present_students:
                absent_students.append(student)
        
        if not absent_students:
            print("No absent students to notify")
            return
        
        # Email configuration
        sender_email = os.getenv("SMTP_EMAIL")
        sender_password = os.getenv("SMTP_PASSWORD")
        
        if not sender_email or not sender_password:
            print("SMTP email credentials not found in environment variables")
            return
            
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        # Set up connection
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            
            for student in absent_students:
                if "Email" in student and student["Email"]:
                    try:
                        message = MIMEMultipart()
                        message["From"] = sender_email
                        message["To"] = student["Email"]
                        message["Subject"] = f"Absence Notification - {date}"
                        
                        body = f"""
                        Dear {student['name']},
                        
                        You were marked absent for class {batch} on {date}.
                        If you believe this is an error, please contact your teacher.
                        
                        Best regards,
                        Attendance System
                        """
                        
                        message.attach(MIMEText(body, "plain"))
                        server.send_message(message)
                        print(f"Absence email sent to {student['name']} ({student['Email']})")
                    except Exception as email_error:
                        print(f"Error sending email to {student['Email']}: {email_error}")
            
            server.quit()
            print(f"Finished sending absence notifications for {date}")
            
        except Exception as server_error:
            print(f"SMTP server error: {server_error}")
            
    except Exception as e:
        print(f"Error in send_attendance_emails: {e}")

@app.get("/getAttendanceFilePath")
async def get_attendance_file_path(teacher_id: str):
    try:
        teacher = teachers.find_one({"teacher_id": teacher_id})
        if not teacher:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        attendance_dir = os.path.join(DATA_DIR, "attendance", f"{teacher_id}_{teacher['name']}")
        
        if not os.path.exists(attendance_dir):
            raise HTTPException(status_code=404, detail="Attendance directory not found")
            
        return {
            "path": attendance_dir,
            "teacher_id": teacher_id,
            "teacher_name": teacher["name"]
        }
    except Exception as e:
        print(f"Error fetching attendance path: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch attendance path: {e}")

@app.get("/getAttendanceData")
async def get_attendance_data(teacher_id: str, batch: str):
    try:
        teacher = teachers.find_one({"teacher_id": teacher_id})
        if not teacher:
            raise HTTPException(status_code=404, detail="Teacher not found")
        
        # Path to attendance CSV
        csv_path = os.path.join(DATA_DIR, "attendance", f"{teacher_id}_{teacher['name']}", f"{batch}.csv")
        
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail=f"Attendance data not found for batch {batch}")
            
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        # Convert to dict for JSON response
        attendance_data = df.to_dict(orient="records")
        
        return {
            "batch": batch,
            "data": attendance_data
        }
    except Exception as e:
        print(f"Error fetching attendance data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch attendance data: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
// migrate.js
const mongoose = require('mongoose');
const { MongoClient } = require('mongodb');
require('dotenv').config();

const { Teacher, Student, Video, Attendance } = require('./schema');

// Connection URI
const uri = process.env.MONGODB_URI

async function migrateData() {
  try {
    // Connect directly with MongoDB driver
    const client = new MongoClient(uri);
    await client.connect();
    console.log("Connected to MongoDB");
    const db = client.db("AttendanceSystem");
    
    // Connect with Mongoose for the new schema
    await mongoose.connect(uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log("Connected to MongoDB with Mongoose");
    
    // Migrate teachers
    const teachersCollection = db.collection("teachers");
    const teachers = await teachersCollection.find({}).toArray();
    console.log(`Found ${teachers.length} teachers to migrate`);
    
    for (const teacher of teachers) {
      const newTeacher = new Teacher({
        teacher_id: teacher.teacher_id,
        name: teacher.name,
        password: teacher.password,
        classes: teacher.classes || []
      });
      await newTeacher.save();
    }
    console.log("Teachers migration completed");
    
    // Migrate students
    const studentsCollection = db.collection("students");
    const students = await studentsCollection.find({}).toArray();
    console.log(`Found ${students.length} students to migrate`);
    
    for (const student of students) {
      const newStudent = new Student({
        name: student.name,
        roll: student.roll,
        batch: student.batch,
        Email: student.Email || "",
        image_paths: student.image_paths || [],
        encodings: student.encodings || []
      });
      await newStudent.save();
    }
    console.log("Students migration completed");
    
    // Migrate videos
    const videosCollection = db.collection("videos");
    const videos = await videosCollection.find({}).toArray();
    console.log(`Found ${videos.length} videos to migrate`);
    
    for (const video of videos) {
      const newVideo = new Video({
        teacher_id: video.teacher_id,
        batch: video.batch,
        video_path: video.video_path,
        status: video.status,
        error: video.error || null,
        processed_at: video.processed_at || null
      });
      await newVideo.save();
    }
    console.log("Videos migration completed");
    
    // Create indexes for better performance
    await Teacher.collection.createIndex({ teacher_id: 1 }, { unique: true });
    await Student.collection.createIndex({ roll: 1 }, { unique: true });
    await Student.collection.createIndex({ batch: 1 });
    await Video.collection.createIndex({ teacher_id: 1 });
    await Video.collection.createIndex({ status: 1 });
    await Attendance.collection.createIndex({ date: 1, batch: 1 });
    
    console.log("Migration completed successfully");
  } catch (error) {
    console.error("Migration failed:", error);
  } finally {
    await mongoose.disconnect();
    console.log("Disconnected from MongoDB");
    process.exit(0);
  }
}

migrateData();
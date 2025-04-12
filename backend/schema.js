// schema.js
const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Teacher Schema
const TeacherSchema = new Schema({
  teacher_id: {
    type: String,
    required: true,
    unique: true
  },
  name: {
    type: String,
    required: true
  },
  password: {
    type: String,
    required: true
  },
  classes: [String]
}, { timestamps: true });

// Student Schema
const StudentSchema = new Schema({
  name: {
    type: String,
    required: true
  },
  roll: {
    type: String,
    required: true,
    unique: true
  },
  batch: {
    type: String,
    required: true
  },
  Email: {
    type: String,
    required: true
  },
  image_paths: [String],
  encodings: [[Number]]
}, { timestamps: true });

// Video Schema
const VideoSchema = new Schema({
  teacher_id: {
    type: String,
    required: true,
    ref: 'Teacher'
  },
  batch: {
    type: String,
    required: true
  },
  video_path: {
    type: String,
    required: true
  },
  status: {
    type: String,
    enum: ['pending', 'processing', 'completed', 'failed'],
    default: 'pending'
  },
  error: String,
  processed_at: Date
}, { timestamps: true });

// Attendance Schema (new - for tracking in MongoDB alongside Google Sheets)
const AttendanceSchema = new Schema({
  date: {
    type: Date,
    required: true
  },
  batch: {
    type: String,
    required: true
  },
  teacher_id: {
    type: String,
    required: true,
    ref: 'Teacher'
  },
  attendance: [{
    student_roll: {
      type: String,
      required: true,
      ref: 'Student'
    },
    status: {
      type: String,
      enum: ['Present', 'Absent'],
      required: true
    }
  }]
}, { timestamps: true });

// Create models
const Teacher = mongoose.model('Teacher', TeacherSchema);
const Student = mongoose.model('Student', StudentSchema);
const Video = mongoose.model('Video', VideoSchema);
const Attendance = mongoose.model('Attendance', AttendanceSchema);

module.exports = {
  Teacher,
  Student,
  Video,
  Attendance
};
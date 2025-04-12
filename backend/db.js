// db.js
const mongoose = require('mongoose');
require('dotenv').config();

// Connection URI from environment variables or use default
const uri =  process.env.MONGODB_URI

// Connect to MongoDB
async function connectDB() {
  try {
    await mongoose.connect(uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log('MongoDB connected successfully');
  } catch (error) {
    console.error('MongoDB connection error:', error);
    process.exit(1);
  }
}

module.exports = connectDB;
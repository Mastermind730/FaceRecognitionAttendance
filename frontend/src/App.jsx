import React from "react"
import {BrowserRouter,Route,Routes} from 'react-router-dom';
import ProtectedRoutes from "./components/ProtectedRoute";
import Login from "./pages/login";
import RegisterStudents from "./pages/RegisterStudents";
import Register from "./pages/Register";
import CaptureVideos from "./pages/CaptureVideos";
import Navbar from "./components/navbar";
import ViewAttendance from "./pages/viewAttendance";
function App() {


  return (
    <>
      
      <BrowserRouter>   
      <Navbar></Navbar>    
        <Routes>
          <Route path ='/' element = {<CaptureVideos route = "http://localhost:8000/video/" />}></Route>
          <Route path ='/login' element = {<Login/>}></Route>
          <Route path ='/register' element = {<Register/>}></Route>
          <Route path ='/registerstudents' element = {<RegisterStudents/>}></Route>
          <Route path = '/ViewAttendance' element = {<ViewAttendance></ViewAttendance>}></Route>
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

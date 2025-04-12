import React from 'react'
import Form from '../components/Form';
function Login(){
  return(
    <>
      <Form route ="http://localhost:8000/login/" method ='login'></Form>
    </>
  );
}

export default Login;
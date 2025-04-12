import React from 'react'
import {Navigate} from 'react-router-dom';
import api from '../api';
import {jwtDecode} from 'jwt-decode';
import { ACCESS_TOKEN,REFRESH_TOKEN } from '../constants';
import {useState,useEffect} from 'react';

function ProtectedRoutes({children}){

    useEffect(()=>{
        auth().catch(()=>{isAuthorized(false)})
    },[]);

    let [isAuthorized,setAuthorized] = useState(null);

    const refresh_token = async ()=>{
        const refreshToken = localStorage.getItem(REFRESH_TOKEN);
        try{

            const response = await api.post("refresh-token url",
            {refresh : refreshToken});

            if(response.status = 200){
                localStorage.setItem(ACCESS_TOKEN,response.data.access);
                setAuthorized(true);
            }else   
                setAuthorized(false);

        }catch(error){
            console.log(error);
            setAuthorized(false);
        }
    }
    const auth = async ()=>{
        const token = localStorage.getItem(ACCESS_TOKEN);
        if(!token){
            setAuthorized(false);
            return ;
        }
        const decoded = jwtDecode(token);
        const tokenExpiration = decoded.exp;
        const now = Date.now()/1000;
        if(tokenExpiration<now)
            await refresh_token();
        else
            setAuthorized(true);
    }

    if(isAuthorized === null){
        return (<div>Loading...</div>)
    }

    return isAuthorized ? children : <Navigate to = '/login'></Navigate>

}

export default ProtectedRoutes;
import { useState, useEffect } from 'react';
import api from '../api';
import { useNavigate } from 'react-router-dom';
import { ACCESS_TOKEN, REFRESH_TOKEN } from '../constants';
import Loading from './Loading';
import './Form.css';

function Form({ route, method }) {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [id, setId] = useState("");
    const [classes, setClasses] = useState([""]);
    const [loading, setLoading] = useState(false);
    const [animateIn, setAnimateIn] = useState(false);
    const navigate = useNavigate();
    const methodName = method === 'login' ? "Login" : "Register";

    // Animation on mount
    useEffect(() => {
        setAnimateIn(true);
    }, []);

    const handleSubmit = async (e) => {
        setLoading(true);
        e.preventDefault();
        try {
            const data = method === 'login' ? { username, password } : { id, name: username, password, classes };
            const response = await api.post(route, data);
            if (method === 'login') {
                localStorage.setItem(ACCESS_TOKEN, response.data.access);
                localStorage.setItem(REFRESH_TOKEN, response.data.refresh);
                console.log(ACCESS_TOKEN);
                // Success animation before redirect
                setAnimateIn(false);
                setTimeout(() => {
                    navigate('/');
                }, 600);
            } else {
                // Success animation before redirect
                setAnimateIn(false);
                setTimeout(() => {
                    navigate('/login');
                }, 600);
            }
        }
        catch (error) {
            console.log(error);
            if (methodName === 'Login')
                alert("Invalid Credentials Or Not registered");
            else
                alert("Enter valid credentials or Account already exists")
        } finally {
            setLoading(false);
        }
    }

    const handleClassChange = (index, value) => {
        const newClasses = [...classes];
        newClasses[index] = value;
        setClasses(newClasses);
    };

    const addClassField = () => {
        setClasses([...classes, ""]);
    };

    const removeClassField = (index) => {
        const newClasses = classes.filter((_, i) => i !== index);
        setClasses(newClasses);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 via-white to-gray-100">
            <div 
                className={`max-w-md w-full bg-white p-8 rounded-xl shadow-lg transform transition-all duration-500 
                ${animateIn ? 'translate-y-0 opacity-100 scale-100' : 'translate-y-8 opacity-0 scale-95'} 
                hover:shadow-2xl`}
            >
                <form onSubmit={handleSubmit} className="w-full space-y-6">
                    <div className="text-center">
                        <h1 className="text-3xl font-bold text-blue-900 mb-1 transition-all duration-300">{methodName}</h1>
                        <div className="w-16 h-1 bg-blue-600 mx-auto mb-4 rounded-full transition-all duration-300 hover:w-24"></div>
                        <p className="text-gray-500 text-sm mb-6">
                            {method === 'login' ? 'Welcome back! Please enter your details' : 'Create your account to get started'}
                        </p>
                    </div>
                    
                    <div className="space-y-4">
                        <div className="transform transition-all duration-300 hover:translate-x-1">
                            <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-1">{method === 'login' ? 'Username' : 'Name'}</label>
                            <input
                                id="username"
                                type="text"
                                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 outline-none backdrop-blur-sm"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder={method === 'login' ? 'Enter your username' : 'Enter your name'}
                                required
                            />
                        </div>
                        
                        {methodName === 'Register' && (
                            <>
                                <div className="transform transition-all duration-300 hover:translate-x-1">
                                    <label htmlFor="id" className="block text-sm font-medium text-gray-700 mb-1">ID</label>
                                    <input
                                        id="id"
                                        type="text"
                                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 outline-none backdrop-blur-sm"
                                        value={id}
                                        onChange={(e) => setId(e.target.value)}
                                        placeholder="Enter your ID"
                                        required
                                    />
                                </div>
                                
                                <div className="space-y-2">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Classes</label>
                                    {classes.map((cls, index) => (
                                        <div 
                                            key={index} 
                                            className="flex items-center space-x-2 transform transition-all duration-300 hover:translate-x-1"
                                        >
                                            <input
                                                type="text"
                                                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 outline-none backdrop-blur-sm"
                                                value={cls}
                                                onChange={(e) => handleClassChange(index, e.target.value)}
                                                placeholder={`Class ${index + 1}`}
                                                required
                                            />
                                            <button
                                                type="button"
                                                className="p-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-all duration-200 flex items-center justify-center hover:rotate-12"
                                                onClick={() => removeClassField(index)}
                                            >
                                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                                    <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                                                </svg>
                                            </button>
                                        </div>
                                    ))}
                                    <div className="transform transition-all duration-300 hover:translate-y-1">
                                        <button
                                            type="button"
                                            className="w-full p-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all duration-200 flex items-center justify-center space-x-2 group"
                                            onClick={addClassField}
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 transition-transform duration-300 group-hover:rotate-180" viewBox="0 0 20 20" fill="currentColor">
                                                <path fillRule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clipRule="evenodd" />
                                            </svg>
                                            <span>Add Class</span>
                                        </button>
                                    </div>
                                </div>
                            </>
                        )}
                        
                        <div className="transform transition-all duration-300 hover:translate-x-1">
                            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                            <input
                                id="password"
                                type="password"
                                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 outline-none backdrop-blur-sm"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter your password"
                                required
                            />
                        </div>
                    </div>
                    
                    {loading && (
                        <div className="flex justify-center">
                            <Loading />
                        </div>
                    )}
                    
                    <div className="space-y-3 pt-2">
                        <button
                            type="submit"
                            className="w-full p-3 bg-blue-900 text-white rounded-lg hover:bg-blue-800 transform transition-all duration-300 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 font-medium group"
                            disabled={loading}
                        >
                            <span className="group-hover:animate-pulse">{methodName}</span>
                        </button>
                        
                        {methodName === 'Login' ? (
                            <div className="text-center pt-2 transform transition-opacity duration-500">
                                <p className="text-gray-600 text-sm">Don't have an account?</p>
                                <button
                                    type="button"
                                    onClick={() => {
                                        setAnimateIn(false);
                                        setTimeout(() => navigate('/register'), 400);
                                    }}
                                    className="w-full p-3 bg-white text-blue-900 border border-blue-900 rounded-lg hover:bg-blue-50 transform transition-all duration-300 hover:scale-105 mt-2 font-medium"
                                >
                                    Register Now
                                </button>
                            </div>
                        ) : (
                            <div className="text-center pt-2 transform transition-opacity duration-500">
                                <p className="text-gray-600 text-sm">Already have an account?</p>
                                <button
                                    type="button"
                                    onClick={() => {
                                        setAnimateIn(false);
                                        setTimeout(() => navigate('/login'), 400);
                                    }}
                                    className="w-full p-3 bg-white text-blue-900 border border-blue-900 rounded-lg hover:bg-blue-50 transform transition-all duration-300 hover:scale-105 mt-2 font-medium"
                                >
                                    Login Instead
                                </button>
                            </div>
                        )}
                    </div>
                </form>
            </div>
        </div>
    );
}

export default Form;
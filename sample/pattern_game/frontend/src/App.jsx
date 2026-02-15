import './App.css'
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Security from './pages/security_section'

function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Security />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

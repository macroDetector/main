import './App.css'
import { BrowserRouter, Routes, Route } from 'react-router-dom';

import Record from './pages/record'

function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Record />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

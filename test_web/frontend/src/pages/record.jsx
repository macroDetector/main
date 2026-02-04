import { useState, useRef, useEffect } from "react";
import './styles/record.css';
import SendData from './services/send_record'

export default function Record_Page() {
    // UI 표시용 상태
    const [displayData, setDisplayData] = useState({ x: 0, y: 0, deltaTime: 0 });
    const [record, setRecord] = useState([]);
    
    // 파이썬의 state = { 'last_ts': ..., 'i': 1 } 와 대응
    const last_ts = useRef(performance.now());
    const i = useRef(1);
    
    // 설정값 (g_vars.tolerance)
    const tolerance = 0.02;
    const MAX_QUEUE_SIZE = 600;

    const on_move = (e) => {
        const now_ts = performance.now();
        // 파이썬의 delta = now_ts - state['last_ts']
        const delta = (now_ts - last_ts.current) / 1000; // 초 단위 변환

        // 1. 설정한 tolerance보다 시간이 더 흘렀을 때만 기록
        if (delta >= tolerance) {
            const data = {
                timestamp: new Date().toISOString(),
                x: Math.floor(e.clientX),
                y: Math.floor(e.clientY),
                deltatime: Number(delta.toFixed(4))
            };

            // state['last_ts'] = now_ts (마지막 기록 시점 업데이트)
            last_ts.current = now_ts;

            // 2. 데이터 "큐"에 넣기 (여기서는 배열 상태 업데이트)
            setRecord(prev => {
                const newRecord = [...prev, data];
            
                return newRecord;
            });

            // 실시간 카드 정보 업데이트
            setDisplayData({ x: data.x, y: data.y, deltaTime: data.deltatime });
        }
    };

    useEffect(() => {
        if (record.length > MAX_QUEUE_SIZE) {
            SendData(record)
            
            setRecord([])
        }}
    , [record])

    const saveToJSON = () => {
        if (record.length === 0) return alert("데이터가 없습니다!");
        const dataStr = JSON.stringify(record, null, 2);
        const blob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `mouse_data_${Date.now()}.json`;
        link.click();
    };

    return (
        <div className="record-container" onMouseMove={on_move}>
            <div className="record-header">
                <h2>Mouse Path Recorder</h2>
                <button className="save-button" onClick={saveToJSON}>JSON 저장</button>
            </div>
            
            <div className="card-container">
                <div className="card">
                    <div className="label">POS</div>
                    <div>{displayData.x}, {displayData.y}</div>
                </div>
                <div className="card delta-card">
                    <div className="label">DELTA (sec)</div>
                    <div>{displayData.deltaTime}s</div>
                </div>
                <div className="card count-card">
                    <div className="label">TOTAL PT</div>
                    <div>{record.length} / 600</div>
                </div>
            </div>

            <div className="log-box">
                <h4>실시간 데이터 수집 현황 (i: {i.current})</h4>
                <div className="log-list">
                    {record.slice(-5).reverse().map((d, idx) => (
                        <div key={idx} className="log-item">
                            <span>{d.timestamp.split('T')[1].split('Z')[0]}</span>
                            <span>X:{d.x} Y:{d.y}</span>
                            <span className="log-delta">Δ {d.deltatime}s</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
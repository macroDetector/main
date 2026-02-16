import { useState, useRef, useEffect } from "react";
import { SendData } from './services/send_record';
import './styles/security_section.scss'
import PatternGame from "./pattern_trajectory";

export default function Record() {
    const [isDragging, setIsDragging] = useState(false);
    const [record, setRecord] = useState([]);
    const [_error_mean, set_Error_Mean] = useState(0.0);
    const isSending = useRef(false);
    const [score, setScore] = useState(0);
    const areaRef = useRef(null);
    const last_ts = useRef(performance.now());
    
    const tolerance = 0.001; 
    const MAX_QUEUE_SIZE = 350;

    const on_move = (e) => {
        // 드래그 중이고, 전송 중이 아니며, 기록이 350개 미만일 때만 수집
        if (isDragging && !isSending.current && record.length < MAX_QUEUE_SIZE && areaRef.current) {
            const now_ts = performance.now();
            const delta = (now_ts - last_ts.current) / 1000;

            if (delta >= tolerance) {
                const rect = areaRef.current.getBoundingClientRect();
                const relX = Math.round(e.clientX - rect.left);
                const relY = Math.round(e.clientY - rect.top);

                const data = {
                    timestamp: new Date().toISOString(),
                    x: relX, y: relY,
                    deltatime: Number(delta.toFixed(4))
                };

                last_ts.current = now_ts;
                setRecord(prev => {
                    // 상태 업데이트 직전 한 번 더 개수 확인 (초과 방지)
                    if (prev.length >= MAX_QUEUE_SIZE) return prev;
                    return [...prev, data];
                });
            }
        }
    };

    const stop_tracking = () => {
        setIsDragging(false);
        // 전송 중이 아닐 때만 리셋 (전송 중 리셋하면 데이터 유실 위험)
        if (!isSending.current) {
            setRecord([]);
            setScore(0);
        }
    };

    useEffect(() => {
        const fetchSend = async () => {
            // 350개에 도달했는지 확인
            if (isSending.current || record.length < MAX_QUEUE_SIZE) return;

            try {
                isSending.current = true;
                
                // 1. 현재까지 모인 350개의 데이터를 복사
                const dataToSend = [...record];
                
                // 2. 즉시 UI 리셋 (이게 늦으면 350개 이상으로 표시됨)
                setRecord([]); 
                setScore(0);
   
                // 3. 서버 전송
                const result = await SendData(dataToSend);
                if (result) set_Error_Mean(result.toFixed(4));
            } catch (err) {
                console.error("전송 에러:", err);
            } finally {
                isSending.current = false;
            }
        };
        fetchSend();
    }, [record.length]);

    return (
        <div className="security-container">
            <header className="security-header">
                <div className="stat-box">
                    <span className="label">SCORE</span>
                    <span className="value">{score}</span>
                </div>
                <div className="stat-box highlighted">
                    <span className="label">POINTS</span>
                    {/* 표시할 때도 MAX_QUEUE_SIZE를 넘지 않도록 제한 */}
                    <span className="value">{Math.min(record.length, MAX_QUEUE_SIZE)}</span>
                    <div className="progress-bar">
                        <div 
                            className="fill" 
                            style={{ width: `${(Math.min(record.length, MAX_QUEUE_SIZE) / MAX_QUEUE_SIZE) * 100}%` }}
                        ></div>
                    </div>
                </div>
                <div className="stat-box">
                    <span className="label">AVG ERROR</span>
                    <span className="value">{_error_mean}</span>
                </div>
            </header>

            <main 
                className="security-area"
                ref={areaRef} 
                onMouseMove={on_move}
                onMouseLeave={stop_tracking}
                onContextMenu={(e) => {
                    e.preventDefault();
                    stop_tracking();
                }}
            >
                <PatternGame 
                    isDragging={isDragging} 
                    setIsDragging={setIsDragging}
                    setScore={setScore}
                />
            </main>

            <footer className="security-panel">
                <div className="status-indicator">
                    <div className={`dot ${isDragging ? 'active' : ''}`}></div>
                    <span>{isDragging ? 'RECORDING...' : 'READY'}</span>
                </div>
                <p className="hint">350 포인트 도달 시 자동 분석 및 리셋</p>
                <p>1 이하 99% 확률로 유저 1% 확률로 고급 유저 움직임 기반 기록기</p>
                <p>3 초과시 매크로 의심</p>
                <p>10 초과시 매크로</p>
                <p>중요! 마우스 전용o, 마우스 패드나 터치는 아직 학습x</p>
            </footer>
        </div>
    );
}
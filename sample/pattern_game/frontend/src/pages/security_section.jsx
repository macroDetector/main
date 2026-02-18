import { useState, useRef, useEffect } from "react";
import { SendData } from './services/send_record';
import PatternGame from "./pattern_trajectory";
import CircularUnlock from "./circular_lock";
import "./styles/security_section.scss";

export default function Record() {
    const [mode, setMode] = useState("pattern"); 
    const [isDragging, setIsDragging] = useState(false);
    const [record, setRecord] = useState([]);
    const [score, setScore] = useState(0);
    const [_error_mean, set_Error_Mean] = useState(0.0);
    const [isSending, setIsSending] = useState(false);
    
    const areaRef = useRef(null);
    const last_ts = useRef(performance.now());
    const isProcessing = useRef(false);

    const MAX_QUEUE_SIZE = 130; 
    const tolerance = 0.001;
    const isToggleMode = true; 

    // [핵심] 중단 및 초기화를 동시에 수행하는 함수
    const stop_and_clear = () => {
        setIsDragging(false);
        setRecord([]); // 기록 즉시 삭제
        setScore(0);   // 점수 즉시 삭제
        console.log("수집 중단 및 데이터 초기화 완료");
    };

    const handle_press_start = () => {
        if (isSending || isProcessing.current) return;
        setRecord([]); // 새로 시작할 때도 비우기
        setIsDragging(true);
        last_ts.current = performance.now();
    };

    const handle_press_end = () => {
        // 일반적인 마우스 뗌(MouseUp) 상황
        if (!isSending && !isProcessing.current && !isToggleMode) {
            stop_and_clear();
        }
    };

    // 우클릭 핸들러: 클릭 시 무조건 초기화 및 토글
    const handle_context_menu = (e) => {
        e.preventDefault(); 
        if (isSending || isProcessing.current) return;

        if (!isDragging) {
            // 새로 수집 시작
            setRecord([]);
            setScore(0);
            last_ts.current = performance.now();
            setIsDragging(true);
        } else {
            // [요청사항] 수집 중 우클릭 시 즉시 중단 및 초기화
            stop_and_clear();
        }
    };

    const on_handle_move = (e) => {
        if (!isDragging || isSending || isProcessing.current || record.length >= MAX_QUEUE_SIZE) return;

        if (areaRef.current) {
            const now_ts = performance.now();
            const delta = (now_ts - last_ts.current) / 1000;

            if (delta >= tolerance) {
                const rect = areaRef.current.getBoundingClientRect();
                const clientX = e.touches ? e.touches[0].clientX : e.clientX;
                const clientY = e.touches ? e.touches[0].clientY : e.clientY;

                if (clientX === undefined || clientY === undefined) return;

                const newData = {
                    timestamp: new Date().toISOString(),
                    x: Math.round(clientX - rect.left),
                    y: Math.round(clientY - rect.top),
                    deltatime: Number(delta.toFixed(4))
                };

                last_ts.current = now_ts;
                setRecord(prev => (prev.length >= MAX_QUEUE_SIZE ? prev : [...prev, newData]));
            }
        }
    };

    useEffect(() => {
        const fetchSend = async () => {
            if (isProcessing.current || record.length < MAX_QUEUE_SIZE) return;

            try {
                isProcessing.current = true; 
                setIsSending(true);
                setIsDragging(false);
                
                const dataToSend = [...record]; 
                setRecord([]); 
                setScore(0);

                const result = await SendData(dataToSend);
                if (result !== undefined) set_Error_Mean(result);
            } catch (err) {
                console.error("Transmission failed:", err);
            } finally {
                setTimeout(() => {
                    setIsSending(false);
                    isProcessing.current = false;
                }, 800); 
            }
        };
        fetchSend();
    }, [record.length]);

    return (
        <div className="security-container">
            <div className="mode-selector">
                <button className={mode === "pattern" ? "active" : ""} onClick={() => { setMode("pattern"); stop_and_clear(); }}>Pattern</button>
                <button className={mode === "circular" ? "active" : ""} onClick={() => { setMode("circular"); stop_and_clear(); }}>Circular</button>
            </div>

            {isSending && (
                <div className="loading-overlay">
                    <div className="loader"></div>
                    <p>분석 중...</p>
                </div>
            )}

            <div className={`content-wrapper ${isSending ? 'blur' : ''}`}>
                <header className="security-header">
                    <div className="stat-box"><span className="label">SCORE</span><span className="value">{score}</span></div>
                    <div className="stat-box highlighted">
                        <span className="label">POINTS</span>
                        <span className="value">{record.length} / {MAX_QUEUE_SIZE}</span>
                        <div className="progress-bar">
                            <div className="fill" style={{ width: `${(record.length / MAX_QUEUE_SIZE) * 100}%` }}></div>
                        </div>
                    </div>
                    <div className="stat-box">
                        <span className="label">ERROR</span>
                        <span className="value">{(Number(_error_mean) * 100).toFixed(2)} %</span>
                    </div>
                </header>

                <main 
                    className="security-area"
                    ref={areaRef} 
                    onMouseMove={on_handle_move}
                    onTouchMove={on_handle_move}
                    onMouseDown={(e) => { if (e.button === 0) handle_press_start(); }}
                    onMouseUp={(e) => { if (e.button === 0) handle_press_end(); }}
                    // [요청사항] 영역 밖으로 나가면 즉시 초기화
                    onMouseLeave={() => stop_and_clear()} 
                    onContextMenu={handle_context_menu}
                    onTouchStart={handle_press_start}
                    onTouchEnd={() => stop_and_clear()}
                >
                    {mode === "pattern" ? (
                        <PatternGame isDragging={isDragging} setIsDragging={setIsDragging} setScore={setScore} />
                    ) : (
                        <CircularUnlock isDragging={isDragging} setIsDragging={setIsDragging} setScore={setScore} />
                    )}
                </main>

                <footer className="security-panel">
                    <div className="status-indicator">
                        <div className={`dot ${isDragging ? 'active' : ''}`}></div>
                        <span>
                            {isDragging 
                                ? '수집 중 (우클릭/영역이탈 시 초기화)' 
                                : '대기 중 (우클릭/눌러서 시작)'}
                        </span>
                    </div>
                </footer>
            </div>
        </div>
    );
}
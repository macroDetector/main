import { useState, useRef, useEffect } from "react";
import { SendData } from './services/send_record';
import PatternGame from "./pattern_trajectory";
import CircularUnlock from "./circular_lock";
import SimpleDrawing from "./simple_drawing";
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

    const MAX_QUEUE_SIZE = 120;
    const tolerance = 0.001;
    const isToggleMode = false; // false = 클릭 유지해야 기록  true = 클릭 한번만 해도 기록

    const stop_and_clear = () => {
        setIsDragging(false);
        setRecord([]);
        setScore(0);
    };

    const handle_press_start = () => {
        if (isSending || isProcessing.current) return;

        setRecord([]);
        setScore(0);
        last_ts.current = performance.now();
        setIsDragging(true);
    };

    const handle_press_end = () => {
        if (!isSending && !isProcessing.current && !isToggleMode) {
            stop_and_clear();
        }
    };

    const handle_context_menu = (e) => {
        e.preventDefault();
        if (isSending || isProcessing.current) return;

        if (!isDragging) {
            handle_press_start();
        } else {
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

                setRecord(prev =>
                    (prev.length >= MAX_QUEUE_SIZE ? prev : [...prev, newData])
                );
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

                <button
                    className={mode === "pattern" ? "active" : ""}
                    onClick={() => { setMode("pattern"); stop_and_clear(); }}
                >Pattern</button>

                <button
                    className={mode === "circular" ? "active" : ""}
                    onClick={() => { setMode("circular"); stop_and_clear(); }}
                >Circular</button>

                <button
                    className={mode === "drawing" ? "active" : ""}
                    onClick={() => { setMode("drawing"); stop_and_clear(); }}
                >Drawing</button>

            </div>

            {isSending && (
                <div className="loading-overlay">
                    <div className="loader"></div>
                    <p>분석 중...</p>
                </div>
            )}

            <div className={`content-wrapper ${isSending ? 'blur' : ''}`}>

                <header className="security-header">

                    <div className="stat-box">
                        <span className="label">SCORE</span>
                        <span className="value">{score}</span>
                    </div>

                    <div className="stat-box highlighted">
                        <span className="label">POINTS</span>
                        <span className="value">{record.length} / {MAX_QUEUE_SIZE}</span>

                        <div className="progress-bar">
                            <div
                                className="fill"
                                style={{ width: `${(record.length / MAX_QUEUE_SIZE) * 100}%` }}
                            ></div>
                        </div>
                    </div>

                    <div className="stat-box">
                        <span className="label">ERROR</span>
                        <span className="value">
                            {(Number(_error_mean) * 100).toFixed(2)} %
                        </span>
                    </div>

                </header>

                <main
                    className="security-area"
                    ref={areaRef}
                    onMouseMove={on_handle_move}
                    onTouchMove={on_handle_move}

                    onMouseDown={(e) => {
                        if (mode === "drawing" && e.button === 0) handle_press_start();
                    }}

                    onTouchStart={() => {
                        if (mode === "drawing") handle_press_start();
                    }}

                    onMouseUp={handle_press_end}
                    onMouseLeave={stop_and_clear}
                    onContextMenu={handle_context_menu}
                    onTouchEnd={stop_and_clear}
                >

                    {mode === "pattern" && (
                        <PatternGame
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            setScore={setScore}
                            onStart={handle_press_start}
                        />
                    )}

                    {mode === "circular" && (
                        <CircularUnlock
                            isDragging={isDragging}
                            setIsDragging={setIsDragging}
                            setScore={setScore}
                            onStart={handle_press_start}
                        />
                    )}

                    {mode === "drawing" && (
                        <SimpleDrawing
                            isDragging={isDragging}
                            setScore={setScore}
                        />
                    )}

                </main>

                <footer className="security-panel">
                    <div className="status-indicator">
                        <div className={`dot ${isDragging ? 'active' : ''}`}></div>
                        <span>
                            {isDragging
                                ? '수집 중 (우클릭/영역이탈 시 초기화)'
                                : '대기 중'}
                        </span>
                    </div>
                </footer>

            </div>
        </div>
    );
}

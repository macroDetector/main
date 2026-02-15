import { useState, useRef, useEffect } from "react";
import {SendData} from './services/send_record';
import './styles/security_section.scss'
import PatternGame from "./pattern_trajectory";

export default function Record() {
    const [isDragging, setIsDragging] = useState(false);
    const [record, setRecord] = useState([]);
    const [raw_error_mean, setRow_Error_Mean] = useState(0.0);
    const [human, setHuman] = useState(true);
    const isSending = useRef(false);
    const [threshold, setThreshold] = useState(0.0);
    const [state, setState] = useState("None")

    const areaRef = useRef(null);

    const [mouseleave, setMouseLeave] = useState(false)
    
    const last_ts = useRef(performance.now());
    
    const tolerance = 0.001;
    const MAX_QUEUE_SIZE = 350;

    const on_move = (e) => {
        if (isDragging == true) {
            const now_ts = performance.now();
            const delta = (now_ts - last_ts.current) / 1000;

            if (delta >= tolerance && areaRef.current) {
                // 1. 영역의 위치 정보 가져오기
                const rect = areaRef.current.getBoundingClientRect();

                // 2. 상대 좌표 계산 (정수 처리)
                const relX = Math.round(e.clientX - rect.left);
                const relY = Math.round(e.clientY - rect.top);

                const data = {
                    timestamp: new Date().toISOString(),
                    x: relX,
                    y: relY,
                    deltatime: Number(delta.toFixed(4))
                };

                last_ts.current = now_ts;

                setRecord(prev => [...prev, data]);
            }
        }
        else {
            setRecord([]);
        }   
    };

    const on_leave = () => {
        if (isDragging) {
            setIsDragging(false);
            setMouseLeave(true);
            setRecord([]); 
        }
    };    
    
    const on_enter = () => {
        // 영역에 다시 들어오면 '취소 상태' 해제
        setMouseLeave(false);
    };

    useEffect(() => {
        const fetchSend = async () => {
            if (isSending.current || record.length < MAX_QUEUE_SIZE) return;

            try {
                isSending.current = true;
                const dataToSend = [...record];
                setRecord([]); 
                setRow_Error_Mean(0.0);
                setThreshold(0.0);
                setState("계산중")
                const result = await SendData(dataToSend);
                
                if (result) {
                    setRow_Error_Mean(result.raw_error_mean.toFixed(4));
                    setHuman(result.human);
                    setThreshold(result.threshold.toFixed(4))
                    setState("계산완료")
                }
            } catch (err) {
                console.error("전송 에러:", err);
            } finally {
                isSending.current = false;
            }
        };

        fetchSend();
    }, [record.length]);

    return (
        <div 
            className="security"
            ref={areaRef} 
            onMouseMove={on_move}
            onMouseLeave={on_leave}
            onMouseEnter={on_enter}        
        >
            <div className="security-area" >
                <PatternGame 
                isDragging = {isDragging} 
                setIsDragging = {setIsDragging} 
                />

            </div>

            <div className="security-panel">
                <p>현재 기록된 좌표 : {record.length}</p>
                <p>State : {state} </p>
                <p>Error Mean : {raw_error_mean} </p>
                <p>Threshold : {threshold}</p>
                {human ? (
                    <p>Human</p>
                ) : 
                    <p>Macro</p>
                }
            </div>
        </div>
    );
}
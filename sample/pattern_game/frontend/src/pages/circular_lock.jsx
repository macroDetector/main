import React, { useRef, useState } from "react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import "./styles/circular_lock.scss";

export default function CircularUnlock({ isDragging, setIsDragging, setScore }) {
    const RADIUS = 120;
    const x = useMotionValue(0);
    const y = useMotionValue(-RADIUS);
    const angle = useMotionValue(0);
    const containerRef = useRef(null);
    
    // 💡 이전 각도를 추적하기 위한 ref (State보다 빠름)
    const lastAngle = useRef(0);

    const handleColor = useTransform(angle, [0, 360], ["#ffffff", "#61c9aa"]);

    const handleDrag = (_, info) => {
        if (!containerRef.current || !isDragging) return;

        const rect = containerRef.current.getBoundingClientRect();
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        // 중앙 기준 상대 좌표
        const clientX = info.point.x - rect.left;
        const clientY = info.point.y - rect.top;

        // 현재 각도 계산 (0 ~ 360)
        let currentAngle = Math.atan2(clientY - centerY, clientX - centerX) * (180 / Math.PI);
        currentAngle = (currentAngle + 90 + 360) % 360;

        // 💡 [핵심] 1회전 판정 로직
        // 이전 각도는 300도 이상이었는데, 현재 각도가 60도 이하로 떨어졌다면 한 바퀴 돌린 것임
        if (lastAngle.current > 300 && currentAngle < 60) {
            setScore(prev => prev + 1);
        }
        
        lastAngle.current = currentAngle; // 이전 각도 업데이트
        angle.set(currentAngle);

        // 시각적 위치 고정
        const rad = (currentAngle - 90) * (Math.PI / 180);
        x.set(Math.cos(rad) * RADIUS);
        y.set(Math.sin(rad) * RADIUS);
    };

    const handleDragEnd = () => {
        // 드래그 종료 시 12시 방향으로 부드럽게 복귀
        animate(x, 0, { type: "spring", stiffness: 300 });
        animate(y, -RADIUS, { type: "spring", stiffness: 300 });
        angle.set(0);
        lastAngle.current = 0;
        setIsDragging(false);
    };

    return (
        <div className="circular-unlock-wrapper" ref={containerRef}>
            <div className="track-circle">
                <svg className="progress-ring" width="400" height="400">
                    <circle cx="200" cy="200" r={RADIUS} />
                </svg>

                <motion.div
                    className="handle"
                    drag
                    // 💡 dragConstraints를 제거하거나 부모 크기만큼 넓게 잡아야 드래그가 끊기지 않음
                    dragConstraints={containerRef} 
                    dragElastic={0}
                    onDragStart={() => setIsDragging(true)}
                    onDrag={handleDrag}
                    onDragEnd={handleDragEnd}
                    style={{ x, y, backgroundColor: handleColor }}
                >
                    <span className="arrow">↻</span>
                </motion.div>

                <div className="center-info">
                    <span className="label">TURN</span>
                    <span className="percent">{Math.round((angle.get() / 360) * 100)}%</span>
                </div>
            </div>
        </div>
    );
}
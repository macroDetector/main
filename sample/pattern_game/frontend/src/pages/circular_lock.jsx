import React, { useRef } from "react";
import { motion, useMotionValue, animate } from "framer-motion";
import "./styles/circular_lock.scss";

export default function CircularUnlock({
    isDragging,
    setIsDragging,
    setScore,
    onStart
}) {

    const RADIUS = 120;

    const x = useMotionValue(0);
    const y = useMotionValue(-RADIUS);

    const angle = useMotionValue(0);

    const containerRef = useRef(null);
    const lastAngle = useRef(0);

    const handleDrag = (_, info) => {

        if (!isDragging || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();

        const cx = info.point.x - rect.left - rect.width / 2;
        const cy = info.point.y - rect.top - rect.height / 2;

        let cur = (Math.atan2(cy, cx) * (180 / Math.PI) + 90 + 360) % 360;

        // 한 바퀴 돌면 점수 증가
        if (lastAngle.current > 300 && cur < 60) {
            setScore(s => s + 1);
        }

        lastAngle.current = cur;

        angle.set(cur);

        const r = (cur - 90) * (Math.PI / 180);

        x.set(Math.cos(r) * RADIUS);
        y.set(Math.sin(r) * RADIUS);
    };

    return (
        <div className="circular-unlock-wrapper" ref={containerRef}>

            <div className="track-circle">

                {/* ⭐ 궤적 원 복구 */}
                <svg className="progress-ring" width="400" height="400">
                    <circle
                        cx="200"
                        cy="200"
                        r={RADIUS}
                    />
                </svg>

                <motion.div
                    className="handle"
                    drag
                    dragConstraints={containerRef}
                    dragElastic={0}

                    onDragStart={() => {
                        onStart();           // 좌표 수집 시작
                        setIsDragging(true);
                    }}

                    onDrag={handleDrag}

                    onDragEnd={() => {

                        setIsDragging(false);

                        animate(x, 0);
                        animate(y, -RADIUS);

                        angle.set(0);
                        lastAngle.current = 0;

                    }}

                    style={{
                        x,
                        y,
                        backgroundColor: isDragging ? "#007bff" : ""
                    }}
                >
                    <span className="arrow">↻</span>
                </motion.div>

                {/* 중앙 정보 */}
                <div className="center-info">
                    <span className="label">TURN</span>
                    <span className="percent">
                        {Math.round((angle.get() / 360) * 100)}%
                    </span>
                </div>

            </div>
        </div>
    );
}

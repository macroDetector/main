import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, useMotionValue, animate } from "framer-motion";
import "./styles/pattern_trajectory.scss";

const CONFIG = {
  GRID_SIZE: 3,
  SPACING: 100,
  OFFSET: 80,
  ARRIVAL_THRESHOLD: 20,
};

export default function PatternGame({ isDragging, setIsDragging, setScore }) {
  const containerRef = useRef(null);
  const [targetIdx, setTargetIdx] = useState(4);
  const targetIdxRef = useRef(targetIdx);
  const isUpdating = useRef(false);

  const containerSize = (CONFIG.GRID_SIZE - 1) * CONFIG.SPACING + CONFIG.OFFSET * 2;

  const gridPoints = useMemo(() => {
    const points = [];
    for (let row = 0; row < CONFIG.GRID_SIZE; row++) {
      for (let col = 0; col < CONFIG.GRID_SIZE; col++) {
        points.push({
          x: col * CONFIG.SPACING + CONFIG.OFFSET,
          y: row * CONFIG.SPACING + CONFIG.OFFSET,
        });
      }
    }
    return points;
  }, []);

  const mX = useMotionValue(gridPoints[4].x);
  const mY = useMotionValue(gridPoints[4].y);

  useEffect(() => {
    targetIdxRef.current = targetIdx;
    isUpdating.current = false;
  }, [targetIdx]);

  // 중앙 복귀 애니메이션 (isDragging이 false가 될 때 실행)
  useEffect(() => {
    if (!isDragging) {
      animate(mX, gridPoints[4].x, { type: "spring", stiffness: 200, damping: 25 });
      animate(mY, gridPoints[4].y, { type: "spring", stiffness: 200, damping: 25 });
    }
  }, [isDragging, gridPoints, mX, mY]);

  // 좌표 추적 및 클램핑 (영역 밖으로 나가지 않도록 제한)
  useEffect(() => {
    const handleGlobalMove = (e) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      
      let nextX = e.clientX - rect.left;
      let nextY = e.clientY - rect.top;

      // 영역 내로 좌표 강제 고정 (Clamping)
      nextX = Math.max(0, Math.min(nextX, containerSize));
      nextY = Math.max(0, Math.min(nextY, containerSize));

      mX.set(nextX);
      mY.set(nextY);
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleGlobalMove);
    }
    return () => window.removeEventListener("mousemove", handleGlobalMove);
  }, [isDragging, mX, mY, containerSize]);

  useEffect(() => {
    const checkArrival = () => {
      const currentTarget = gridPoints[targetIdxRef.current];
      const d = Math.sqrt(Math.pow(mX.get() - currentTarget.x, 2) + Math.pow(mY.get() - currentTarget.y, 2));

      if (d < CONFIG.ARRIVAL_THRESHOLD && !isUpdating.current && isDragging) {
        isUpdating.current = true;
        setScore(s => s + 1);
        setTargetIdx(prev => {
          let next;
          do { next = Math.floor(Math.random() * gridPoints.length); } while (next === prev);
          return next;
        });
      }
    };
    const unsubX = mX.on("change", checkArrival);
    const unsubY = mY.on("change", checkArrival);
    return () => { unsubX(); unsubY(); };
  }, [gridPoints, mX, mY, isDragging, setScore]);

  return (
    <div className="game-wrapper">
      <div 
        ref={containerRef}
        className={`pattern-container ${isDragging ? 'active' : ''}`}
        style={{ 
            width: containerSize, 
            height: containerSize, 
            position: 'relative',
            overflow: 'hidden' // 시각적으로도 공이 삐져나가지 않게 함
        }}
      >
        {gridPoints.map((point, i) => (
          <div key={i} className={`grid-dot ${i === targetIdx ? 'is-target' : ''}`} style={{ left: point.x, top: point.y, position: 'absolute' }}>
            {i === targetIdx && <motion.div layoutId="pulse" className="target-pulse" />}
          </div>
        ))}

        <motion.div
          className="player-ball"
          onMouseDown={(e) => {
            if (e.button === 0) {
              e.stopPropagation();
              setIsDragging(prev => !prev); // 클릭 토글
            }
          }}
          style={{ x: mX, y: mY, left: -20, top: -20, position: 'absolute' }}
          animate={{ 
            scale: isDragging ? 1 : 1.1,
            backgroundColor: isDragging ? "#007bff" : "#dee2e6",
            boxShadow: isDragging ? "0 8px 20px rgba(0, 123, 255, 0.3)" : "0 4px 10px rgba(0,0,0,0.05)"
          }}
        />
      </div>
    </div>
  );
}
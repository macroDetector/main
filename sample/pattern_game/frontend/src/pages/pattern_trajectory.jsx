import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, useMotionValue, animate } from "framer-motion";
import "./styles/pattern_trajectory.scss";

const CONFIG = {
  GRID_SIZE: 3,
  SPACING: 100,
  OFFSET: 60,
  ARRIVAL_THRESHOLD: 15, // 이 거리(극단값) 이내로 들어오면 점수 획득
};

export default function PatternGame({ isDragging, setIsDragging }) {
  const containerRef = useRef(null);
  const [targetIdx, setTargetIdx] = useState(0);
  const [score, setScore] = useState(0);
  const [currentDistance, setCurrentDistance] = useState(0);

  // 1. 최신 targetIdx를 리스너 안에서 참조하기 위한 Ref
  const targetIdxRef = useRef(targetIdx);
  // 2. 점수 중복 계산 방지용 Flag
  const isUpdating = useRef(false);

  // 컨테이너 전체 크기 계산
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

  // 초기 위치는 중앙 그리드(인덱스 4)
  const mX = useMotionValue(gridPoints[4].x);
  const mY = useMotionValue(gridPoints[4].y);

  // targetIdx가 변할 때마다 Ref 업데이트
  useEffect(() => {
    targetIdxRef.current = targetIdx;
    isUpdating.current = false; // 타겟이 바뀌면 다시 점수 획득 가능 상태로
  }, [targetIdx]);

  useEffect(() => {
    const checkArrival = () => {
      // 리액트 상태 대신 Ref를 사용하여 항상 최신 타겟 좌표 계산
      const currentTarget = gridPoints[targetIdxRef.current];
      
      const d = Math.sqrt(
        Math.pow(mX.get() - currentTarget.x, 2) + 
        Math.pow(mY.get() - currentTarget.y, 2)
      );

      setCurrentDistance(d.toFixed(1));

      // 임계값(Threshold) 체크 및 중복 가산 방지
      if (d < CONFIG.ARRIVAL_THRESHOLD && !isUpdating.current) {
        isUpdating.current = true; // 잠금: 다음 타겟이 설정될 때까지 점수 획득 중단
        
        setScore(s => s + 1);
        
        setTargetIdx(prev => {
          let next;
          do {
            next = Math.floor(Math.random() * gridPoints.length);
          } while (next === prev);
          return next;
        });
      }
    };

    const unsubX = mX.on("change", checkArrival);
    const unsubY = mY.on("change", checkArrival);
    return () => {
      unsubX();
      unsubY();
    };
  }, [gridPoints, mX, mY]);

  const handleMouseMove = (e) => {
    if (!isDragging || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    mX.set(e.clientX - rect.left);
    mY.set(e.clientY - rect.top);
  };

  const handleContextMenu = (e) => {
    e.preventDefault(); 
    setIsDragging(false);

    const centerX = containerSize / 2;
    const centerY = containerSize / 2;

    animate(mX, centerX, { type: "spring", stiffness: 300, damping: 30 });
    animate(mY, centerY, { type: "spring", stiffness: 300, damping: 30 });
  };

  return (
    <div 
      className="game-wrapper"
      onMouseMove={handleMouseMove}
      onContextMenu={handleContextMenu}
    >
      <div className="header-info">
        <div className="score-board">SCORE: {score}</div>
        <div className="mission-text">좌클릭: 잡기 | 우클릭: 놓기 & 중앙복귀</div>
        <div className={`distance-display ${currentDistance < CONFIG.ARRIVAL_THRESHOLD ? 'in-range' : ''}`}>
          현재 중심거리: <span>{currentDistance}</span> PX
        </div>
      </div>

      <div 
        ref={containerRef}
        className={`pattern-container ${isDragging ? 'dragging' : ''}`}
        style={{ width: containerSize, height: containerSize }}
      >
        {gridPoints.map((point, i) => (
          <div
            key={i}
            className={`grid-dot ${i === targetIdx ? 'is-target' : ''}`}
            style={{ left: point.x, top: point.y }}
          >
            {i === targetIdx && <div className="target-pulse" />}
          </div>
        ))}

        <motion.div
          className="player-ball"
          onMouseDown={(e) => { 
            e.preventDefault();
            if (e.button === 0) setIsDragging(true);
          }}
          style={{ x: mX, y: mY, left: 0, top: 0, position: 'absolute' }}
          animate={{ 
            scale: isDragging ? 1 : 1.2,
            backgroundColor: isDragging ? "#3b82f6" : "#ef4444" 
          }}
        />
      </div>
    </div>
  );
}
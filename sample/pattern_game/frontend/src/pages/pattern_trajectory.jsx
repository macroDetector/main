import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, useMotionValue, animate } from "framer-motion";
import "./styles/pattern_trajectory.scss";

export default function PatternGame({ isDragging, setIsDragging, setScore }) {
  const containerRef = useRef(null);
  const [targetIdx, setTargetIdx] = useState(4); // 초기 타겟은 중앙
  const targetIdxRef = useRef(targetIdx);
  const isUpdating = useRef(false);
  
  // 모바일 대응을 위한 가변 사이즈 상태
  const [dimensions, setDimensions] = useState({ size: 300, spacing: 80, offset: 70 });

  // 1. 부모 너비에 맞게 그리드 크기 재계산 (짤림 방지)
  useEffect(() => {
    const updateSize = () => {
      if (!containerRef.current) return;
      const parentWidth = containerRef.current.parentElement.offsetWidth;
      const availableSize = Math.min(parentWidth, 400); // 최대 400px
      
      setDimensions({
        size: availableSize,
        spacing: availableSize * 0.3, 
        offset: availableSize * 0.2
      });
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  // 2. 그리드 좌표 생성
  const gridPoints = useMemo(() => {
    const points = [];
    const { spacing, offset } = dimensions;
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        points.push({
          x: col * spacing + offset,
          y: row * spacing + offset,
        });
      }
    }
    return points;
  }, [dimensions]);

  const mX = useMotionValue(0);
  const mY = useMotionValue(0);

  // 초기 위치 설정
  useEffect(() => {
    if (gridPoints[4]) {
      mX.set(gridPoints[4].x);
      mY.set(gridPoints[4].y);
    }
  }, [gridPoints, mX, mY]);

  // 비활성화(초기화) 시 공을 중앙으로 복귀시킴
  useEffect(() => {
    if (!isDragging && gridPoints[4]) {
      animate(mX, gridPoints[4].x, { type: "spring", stiffness: 250, damping: 30 });
      animate(mY, gridPoints[4].y, { type: "spring", stiffness: 250, damping: 30 });
      setTargetIdx(4); // 타겟도 중앙으로 리셋
    }
  }, [isDragging, gridPoints, mX, mY]);

  // 3. 마우스/터치 이동에 따른 공의 위치 업데이트
  useEffect(() => {
    const handleMove = (e) => {
      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      
      let nextX = clientX - rect.left;
      let nextY = clientY - rect.top;

      // 영역 내부로 제한
      nextX = Math.max(0, Math.min(nextX, dimensions.size));
      nextY = Math.max(0, Math.min(nextY, dimensions.size));

      mX.set(nextX);
      mY.set(nextY);
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("touchmove", handleMove, { passive: false });
    }
    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("touchmove", handleMove);
    };
  }, [isDragging, mX, mY, dimensions]);

  // 4. 타겟 도달 판정 (Extreme Values 수집용)
  useEffect(() => {
    targetIdxRef.current = targetIdx;
    isUpdating.current = false;
  }, [targetIdx]);

  useEffect(() => {
    const checkArrival = () => {
      if (!isDragging) return;
      const currentTarget = gridPoints[targetIdxRef.current];
      if (!currentTarget) return;

      const dist = Math.sqrt(
        Math.pow(mX.get() - currentTarget.x, 2) + 
        Math.pow(mY.get() - currentTarget.y, 2)
      );

      // 타겟 반경 25px 이내 진입 시
      if (dist < 25 && !isUpdating.current) {
        isUpdating.current = true;
        setScore(s => s + 1); // 점수 증가
        setTargetIdx(prev => {
          let next;
          do { next = Math.floor(Math.random() * gridPoints.length); } 
          while (next === prev); // 이전과 같은 위치 제외
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
        className="pattern-container"
        style={{ 
            width: dimensions.size, 
            height: dimensions.size, 
            position: 'relative'
        }}
      >
        {/* 그리드 점들과 타겟 표시 */}
        {gridPoints.map((point, i) => (
          <div 
            key={i} 
            className={`grid-dot ${i === targetIdx ? 'is-target' : ''}`} 
            style={{ left: point.x, top: point.y, position: 'absolute' }}
          >
            {i === targetIdx && (
              <motion.div layoutId="pulse" className="target-pulse" />
            )}
          </div>
        ))}

        {/* 플레이어 공: 토글 모드이므로 직접적인 이벤트는 꺼둠 */}
        <motion.div
          className="player-ball"
          style={{ 
            x: mX, y: mY, 
            left: -18, top: -18, // 반지름 보정
            position: 'absolute',
            pointerEvents: 'none', // 부모의 onClick 방해 금지
            backgroundColor: isDragging ? "#007bff" : "#adb5bd"
          }}
          animate={{ 
            scale: isDragging ? 1 : 0.8,
            opacity: isDragging ? 1 : 0.5
          }}
        />
      </div>
    </div>
  );
}
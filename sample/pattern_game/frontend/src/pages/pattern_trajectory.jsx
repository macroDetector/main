import React, { useState, useEffect, useMemo, useRef } from "react";
import { motion, useMotionValue, animate } from "framer-motion";
import "./styles/pattern_trajectory.scss";

export default function PatternGame({
  isDragging,
  setIsDragging,
  setScore,
  onStart
}) {

  const containerRef = useRef(null);

  const [targetIdx, setTargetIdx] = useState(4);
  const targetIdxRef = useRef(targetIdx);
  const isUpdating = useRef(false);

  const [dimensions, setDimensions] = useState({
    size: 300,
    spacing: 80,
    offset: 70
  });

  // 반응형 사이즈
  useEffect(() => {

    const updateSize = () => {

      if (!containerRef.current) return;

      const parentWidth = containerRef.current.parentElement.offsetWidth;
      const availableSize = Math.min(parentWidth, 400);

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

  // 그리드 생성
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

  // 초기 위치
  useEffect(() => {

    if (gridPoints[4]) {
      mX.set(gridPoints[4].x);
      mY.set(gridPoints[4].y);
    }

  }, [gridPoints]);

  // 비활성화 시 중앙 복귀
  useEffect(() => {

    if (!isDragging && gridPoints[4]) {

      animate(mX, gridPoints[4].x, { type: "spring", stiffness: 250, damping: 30 });
      animate(mY, gridPoints[4].y, { type: "spring", stiffness: 250, damping: 30 });

      setTargetIdx(4);

    }

  }, [isDragging, gridPoints]);

  // 드래그 이동
  useEffect(() => {

    const handleMove = (e) => {

      if (!isDragging || !containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();

      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;

      let nextX = clientX - rect.left;
      let nextY = clientY - rect.top;

      nextX = Math.max(0, Math.min(nextX, dimensions.size));
      nextY = Math.max(0, Math.min(nextY, dimensions.size));

      mX.set(nextX);
      mY.set(nextY);

    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("touchmove", handleMove);
    }

    return () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("touchmove", handleMove);
    };

  }, [isDragging, dimensions]);

  // ⭐ 타겟 도달 판정 (핵심 복구)
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

      if (dist < 25 && !isUpdating.current) {

        isUpdating.current = true;

        setScore(s => s + 1);

        setTargetIdx(prev => {

          let next;

          do {
            next = Math.floor(Math.random() * gridPoints.length);
          }
          while (next === prev);

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

  }, [gridPoints, isDragging]);

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

        {gridPoints.map((point, i) => (
          <div
            key={i}
            className={`grid-dot ${i === targetIdx ? 'is-target' : ''}`}
            style={{
              left: point.x,
              top: point.y,
              position: 'absolute'
            }}
          />
        ))}

        <motion.div
          className="player-ball"
          onMouseDown={onStart}
          onTouchStart={onStart}
          style={{
            x: mX,
            y: mY,
            left: -18,
            top: -18,
            position: 'absolute',
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

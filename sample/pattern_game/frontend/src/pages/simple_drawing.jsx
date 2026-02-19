import React, { useState, useEffect } from "react";

export default function SimpleDrawing({ isDragging, setScore }) {

    const [currentPath, setCurrentPath] = useState("");

    useEffect(() => {

        if (!isDragging) {
            setCurrentPath("");
        }

    }, [isDragging]);

    const handleMouseMove = (e) => {

        if (!isDragging) return;

        const rect = e.currentTarget.getBoundingClientRect();

        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
        const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;

        const newPoint = `${x},${y}`;

        if (currentPath === "") {
            setCurrentPath(`M ${newPoint}`);
        } else {
            setCurrentPath(prev => `${prev} L ${newPoint}`);
            setScore(prev => prev + 1);
        }

    };

    return (
        <div
            className="drawing-container"
            onMouseMove={handleMouseMove}
            onTouchMove={handleMouseMove}
            style={{
                width: '100%',
                height: '100%',
                cursor: 'crosshair'
            }}
        >

            <svg width="100%" height="100%">
                <path
                    d={currentPath}
                    fill="transparent"
                    stroke="#4facfe"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
            </svg>

            <div className="drawing-guide">
                자유롭게 선을 그려 데이터를 수집하세요
            </div>

        </div>
    );
}

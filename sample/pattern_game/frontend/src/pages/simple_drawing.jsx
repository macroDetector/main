import React, { useMemo } from "react";
import "./styles/drawing_canvas.scss";

export default function SimpleDrawing({ isDragging, record }) {
    // 수집된 데이터를 SVG 경로로 변환
    const linePath = useMemo(() => {
        if (record.length === 0) return "";
        return record.reduce((path, pt, i) => {
            return i === 0 ? `M ${pt.x} ${pt.y}` : `${path} L ${pt.x} ${pt.y}`;
        }, "");
    }, [record]);

    return (
        <div className="drawing-canvas-container">
            {/* 가이드 텍스트: 필기체로 human 표시 */}
            <div className="text-guide-layer">human</div>

            <svg className="drawing-svg-layer" width="100%" height="100%">
                <path 
                    d={linePath} 
                    fill="none" 
                    stroke="#61c9aa" 
                    strokeWidth="5" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                />
            </svg>

            {!isDragging && record.length === 0 && (
                <div className="onboarding-hint">
                    가이드를 따라 <b>human</b>을 써주세요
                </div>
            )}
        </div>
    );
}
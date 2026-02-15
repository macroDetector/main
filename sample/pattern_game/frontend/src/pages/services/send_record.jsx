export async function SendData(data) {
  try {
    const res = await fetch(
      `${import.meta.env.VITE_POST_URL}/api/get_points`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }
    );

    const result = await res.json();
    const messageData = result.message;

    let raw_error = [];
    let threshold = 0.0;
    let raw_error_mean = 0.0;

    if (messageData && messageData.raw_error) {
      raw_error = messageData.raw_error;
    }
    
    if (messageData && messageData.threshold && messageData.threshold.length > 0) {
      threshold = messageData.threshold[0];
    }

    // --- 평균(Mean) 계산 로직 수정 ---
    if (raw_error.length > 0) {
      // reduce를 사용하여 배열의 합계를 구합니다.
      const sum = raw_error.reduce((acc, val) => acc + val, 0);
      raw_error_mean = sum / raw_error.length;
    } else {
      raw_error_mean = 0.0;
    }

    let human = false;

    if (raw_error_mean < threshold) {
      human = true
    }

    return {
      raw_error_mean: raw_error_mean,
      threshold:threshold,
      human:human
    };

  } catch (err) {
    console.error("SendData failed:", err);
    return false;
  }
}

export async function SendDataLive(data) {
  let socket = null;

  socket = new WebSocket(`${import.meta.env.VITE_POST_URL_WS}/ws/get_points_live`);

  socket.onopen = () => {
    console.log("WebSocket 연결이 열렸습니다.");
    socket.send(JSON.stringify(data));
  };

  socket.onmessage = (event) => {
    const result = JSON.parse(event.data);
    // 서버에서 온 판정 결과(Human/Macro) 처리
    console.log("서버 응답:", result);
  };  
  
  socket.onerror = (err) => {
    console.error("❌ 웹소켓 에러:", err);
  };

  socket.onclose = () => {
    console.log("🛑 연결 종료. 재연결 시도...");
    socket = null;
    // 필요 시 setTimeout으로 재연결 로직 추가
  };  
}
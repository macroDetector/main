const BASE_URL = import.meta.env.VITE_POST_URL || ""

export async function SendData(data) {
  try {
    const res = await fetch(
      `${BASE_URL}/api/get_points`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      }
    );

    const result = await res.json();
    return result.message

  } catch (err) {
    console.error("SendData failed:", err);
    return false;
  }
}
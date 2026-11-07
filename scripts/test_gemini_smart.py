import requests
import json
from app.config import settings

def test():
    prompt = """Bạn là chuyên gia thị giác máy tính cao cấp trong hệ thống nhận diện sản phẩm tự động.
Phân tích kỹ hình ảnh sản phẩm được đưa vào (có thể do người dùng cầm trên tay hoặc đặt trên bàn).

HÃY SUY LUẬN THEO CÁC BƯỚC:
1. Quan sát hình thái học:
   - Nếu là khối chữ nhật phẳng mỏng, có màn hình kính đen, camera sau, phím nguồn/âm lượng ở cạnh: Đây là ĐIỆN THOẠI (cell phone / smartphone).
   - Nếu là hình trụ tròn bằng kim loại có nắp bật: Đây là LON NƯỚC (can).
   - Nếu là chai nhựa/thủy tinh có nắp vặn: Đây là CHAI (bottle).
   - Nếu là khối hộp chữ nhật bằng giấy/carton hoặc túi nilon kín: Đây là HỘP/GÓI (box).
2. Đọc toàn bộ chữ, nhãn hiệu, logo xuất hiện trên thân vật thể (nếu có logo Xiaomi, Apple, Samsung, Coca-Cola, Vinamilk, v.v.).
3. Bỏ qua hoàn toàn khuôn mặt người, bàn tay và hậu cảnh phòng.

Trả về chuỗi JSON chính xác cấu trúc sau:
{
  "visual_inspection": "Mô tả ngắn gọn đặc điểm vật lý nhận dạng (hình dáng, chất liệu, màn hình hoặc nắp lon hoặc nhãn mác)",
  "product_name": "Tên chi tiết sản phẩm tiếng Việt",
  "brand": "Thương hiệu",
  "category": "Ngành hàng",
  "visual_class": "can | bottle | box | cup | bowl | cell phone | book | product",
  "specs": "Quy cách hoặc thông số",
  "confidence": 0.95
}"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent?key={settings.GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }
    r = requests.post(url, json=payload, timeout=15)
    print("Status:", r.status_code)
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(txt)
    print("Parsed JSON successfully:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test()

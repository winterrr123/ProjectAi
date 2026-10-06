# PROJECT AI - NGUYÊN TẮC THIẾT KẾ VÀ PHÁT TRIỂN

Tài liệu này lưu trữ nguyên tắc phân định cốt lõi giữa hai luồng xử lý chính trong dự án **ProjectAI**, luôn phải tuân thủ nghiêm ngặt trong mọi lần phát triển và cập nhật code tiếp theo:

---

## 1. Phần Upload Video (`/upload`, `upload.html`, `upload.js`, `video_service.py`)
* **Mục tiêu duy nhất:** **TẬP TRUNG ĐẾM CHÍNH XÁC VÀ CHỐNG ĐẾM LẶP LẠI (Precision Counting)**.
* **Quy chuẩn đối tượng:**
  * Không phân loại tên chi tiết từng món đồ hay thương hiệu để tránh gây nhiễu thuật toán bám vết.
  * Mọi vật thể hợp lệ khi phát hiện đều được quy chuẩn thống nhất là **`"Sản phẩm"`** (hoặc `Sản phẩm #ID`).
* **Thuật toán & Theo dõi:**
  * Sử dụng ByteTrack duy trì quỹ đạo tracking liên tục cho từng vật thể từ lúc xuất hiện đến khi rời khung hình.
  * Mỗi vật thể được cấp một mã `tracking_id` duy nhất và chỉ được xác nhận đếm đúng **1 lần duy nhất** trong suốt video.
* **Giao diện người dùng:**
  * Ẩn bảng phân loại chi tiết.
  * Nhãn khung Bounding box chỉ hiển thị ngắn gọn: `Sản phẩm #ID` (xanh dương khi đang track, xanh lá khi đã đếm `✓ Sản phẩm #ID`).
  * Ẩn các huy hiệu che tầm nhìn (`AI DETECTION ACTIVE`, `Chế độ: REALTIME AI`).

---

## 2. Phần Live Camera (`/camera`, `camera.html`, `camera.js`, `camera_service.py`)
* **Mục tiêu kép:** **ĐẾM CHÍNH XÁC KHÔNG LẶP LẠI** VÀ **PHÂN LOẠI CHI TIẾT DANH MỤC**.
* **Phân loại đối tượng:**
  * Nhận diện và phân loại rõ ràng tên của:
    * **Đồ vật / Hàng hóa:** chai nước, ly cốc, điện thoại, sách vở, máy tính, balo, túi xách, thùng hộp...
    * **Đồ ăn / Thực phẩm:** gói cà phê, bánh kẹo, snack, táo, chuối, cam, sandwich, pizza...
    * **Con vật / Động vật:** các loài thú cưng hoặc động vật quen thuộc.
* **Thuật toán & Theo dõi:**
  * Sử dụng ByteTrack + Tracking ID để chống đếm lặp lại cho cùng một vật thể xuất hiện trước ống kính.
  * Khóa nhãn phân loại ổn định (`track_class_votes`) để không bị nhảy tên chập chờn giữa các frame.
* **Giao diện người dùng:**
  * Hiển thị bảng danh mục phân loại chi tiết và danh sách vật thể đang có trong khung hình kèm tỷ lệ tin cậy (%).
  * Ẩn các huy hiệu che khung hình (`AI DETECTION ACTIVE`, `Chế độ: REALTIME AI`), để không gian camera thông thoáng và hiển thị gọn các thông số FPS / Thời gian ở góc phải.

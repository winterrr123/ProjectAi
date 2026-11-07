import io
import openpyxl
from fastapi.testclient import TestClient

from app.database.crud import UserCRUD
from app.database.mysql import SessionLocal, init_db
from app.main import app
from app.utils.auth import create_access_token

client = TestClient(app)


def test_export_excel_api_generates_valid_workbook():
    init_db()
    db = SessionLocal()
    try:
        user = UserCRUD.get_or_create_google_user(
            db, "g_excel_user", "excel@test.com", "Excel Tester", None
        )
        token = create_access_token({"sub": str(user.id), "email": user.email})

        payload = {
            "video_name": "bang_chuyen_hang.mp4",
            "total_count": 48,
            "by_class": {
                "Điện thoại Xiaomi Redmi": 28,
                "Lon nước ngọt Sting": 20,
            },
            "target_name": "Điện thoại Xiaomi Redmi",
            "detections": [
                {
                    "tracking_id": 1,
                    "class_name": "Điện thoại Xiaomi Redmi",
                    "timestamp": 2.4,
                    "confidence": 0.95,
                },
                {
                    "tracking_id": 2,
                    "class_name": "Lon nước ngọt Sting",
                    "timestamp": 5.1,
                    "confidence": 0.91,
                },
            ],
        }

        response = client.post(
            "/api/video/export-excel",
            json=payload,
            cookies={"access_token": token},
        )
        assert response.status_code == 200
        assert "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" in response.headers.get("content-type", "")

        wb = openpyxl.load_workbook(io.BytesIO(response.content))
        assert "Báo Cáo Kiểm Đếm" in wb.sheetnames
        ws = wb["Báo Cáo Kiểm Đếm"]

        # Check title and content
        assert "BÁO CÁO KẾT QUẢ KIỂM ĐẾM SẢN PHẨM" in str(ws["A1"].value)
        
        # Search for product names and counts
        found_xiaomi = False
        found_sting = False
        for row in ws.iter_rows(values_only=True):
            row_str = " ".join([str(v) for v in row if v is not None])
            if "Điện thoại Xiaomi Redmi" in row_str:
                found_xiaomi = True
            if "Lon nước ngọt Sting" in row_str:
                found_sting = True

        assert found_xiaomi is True
        assert found_sting is True
    finally:
        db.close()

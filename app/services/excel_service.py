"""
Excel Reporting Service for Product Vision.
Generates styled professional spreadsheets (.xlsx) containing counting summaries,
product names, quantities, percentage breakdown, and tracking logs.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from app.config import settings
from app.utils.helpers import logger


def generate_counting_excel(
    output_path: str,
    video_name: str,
    total_objects: int,
    by_class: Dict[str, int],
    target_name: Optional[str] = None,
    session_id: Optional[int] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Creates an Excel report file (.xlsx) with professional corporate styling.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Báo Cáo Kiểm Đếm"
    ws.views.sheetView[0].showGridLines = True

    # Styling definitions
    font_family = "Segoe UI"
    
    # Palette
    COLOR_PRIMARY_DARK = "1E3A8A"    # Dark blue
    COLOR_PRIMARY = "2563EB"         # Brand blue
    COLOR_SUBTLE_BG = "F8FAFC"       # Light gray-blue
    COLOR_ROW_ALT = "F1F5F9"         # Slate-100
    COLOR_BORDER = "CBD5E1"          # Slate-300
    COLOR_TEXT = "0F172A"            # Slate-900
    COLOR_MUTED = "64748B"           # Slate-500
    COLOR_TOTAL_BG = "EFF6FF"        # Blue-50

    thin_border = Border(
        left=Side(style="thin", color=COLOR_BORDER),
        right=Side(style="thin", color=COLOR_BORDER),
        top=Side(style="thin", color=COLOR_BORDER),
        bottom=Side(style="thin", color=COLOR_BORDER),
    )
    double_bottom_border = Border(
        left=Side(style="thin", color=COLOR_BORDER),
        right=Side(style="thin", color=COLOR_BORDER),
        top=Side(style="thin", color=COLOR_BORDER),
        bottom=Side(style="double", color=COLOR_PRIMARY_DARK),
    )

    # 1. Main Header Title (Row 1-2)
    ws.merge_cells("A1:E1")
    title_cell = ws["A1"]
    title_cell.value = "BÁO CÁO KẾT QUẢ KIỂM ĐẾM SẢN PHẨM"
    title_cell.font = Font(name=font_family, size=15, bold=True, color="FFFFFF")
    title_cell.fill = PatternFill(start_color=COLOR_PRIMARY_DARK, end_color=COLOR_PRIMARY_DARK, fill_type="solid")
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 36

    ws.merge_cells("A2:E2")
    sub_cell = ws["A2"]
    sub_cell.value = "Hệ thống thị giác máy tính nhận diện & kiểm đếm tự động - Product Vision"
    sub_cell.font = Font(name=font_family, size=10, italic=True, color=COLOR_MUTED)
    sub_cell.fill = PatternFill(start_color=COLOR_SUBTLE_BG, end_color=COLOR_SUBTLE_BG, fill_type="solid")
    sub_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 20

    # 2. Metadata Information Block (Rows 4-6)
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    info_rows = [
        ("Nguồn video / tệp xử lý:", str(video_name), "Thời gian xuất báo cáo:", now_str),
        ("Mã phiên kiểm đếm:", f"#{session_id}" if session_id else "Phiên trực tiếp", "Mục tiêu áp dụng:", target_name or "Đếm tất cả sản phẩm"),
        ("Tổng số lượng đã kiểm đếm:", f"{total_objects:,} sản phẩm / đồ vật", "Trạng thái:", "Hoàn tất kiểm đếm"),
    ]

    start_info_row = 4
    for idx, (lbl1, val1, lbl2, val2) in enumerate(info_rows):
        r = start_info_row + idx
        ws.row_dimensions[r].height = 20
        
        c_lbl1 = ws.cell(row=r, column=1, value=lbl1)
        c_lbl1.font = Font(name=font_family, size=10, bold=True, color=COLOR_MUTED)
        c_lbl1.alignment = Alignment(horizontal="left", vertical="center")

        c_val1 = ws.cell(row=r, column=2, value=val1)
        if "Tổng số lượng" in lbl1:
            c_val1.font = Font(name=font_family, size=11, bold=True, color=COLOR_PRIMARY)
        else:
            c_val1.font = Font(name=font_family, size=10, bold=False, color=COLOR_TEXT)
        c_val1.alignment = Alignment(horizontal="left", vertical="center")

        c_lbl2 = ws.cell(row=r, column=4, value=lbl2)
        c_lbl2.font = Font(name=font_family, size=10, bold=True, color=COLOR_MUTED)
        c_lbl2.alignment = Alignment(horizontal="left", vertical="center")

        c_val2 = ws.cell(row=r, column=5, value=val2)
        c_val2.font = Font(name=font_family, size=10, bold=False, color=COLOR_TEXT)
        c_val2.alignment = Alignment(horizontal="left", vertical="center")

    # 3. Main Summary Table (Rows 8+)
    table_start_row = 8
    ws.merge_cells(f"A{table_start_row}:E{table_start_row}")
    section_title = ws[f"A{table_start_row}"]
    section_title.value = "I. BẢNG TỔNG HỢP SỐ LƯỢNG THEO TÊN SẢN PHẨM / ĐỒ VẬT"
    section_title.font = Font(name=font_family, size=11, bold=True, color=COLOR_PRIMARY_DARK)
    ws.row_dimensions[table_start_row].height = 24

    headers = ["STT", "Tên Sản Phẩm / Đồ Vật", "Nhãn Phân Loại", "Số Lượng (Đã Đếm)", "Tỷ Lệ (%)"]
    header_row = table_start_row + 1
    ws.row_dimensions[header_row].height = 26

    for col_idx, h_text in enumerate(headers, 1):
        cell = ws.cell(row=header_row, column=col_idx, value=h_text)
        cell.font = Font(name=font_family, size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color=COLOR_PRIMARY, end_color=COLOR_PRIMARY, fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = thin_border

    # Populate item rows
    current_row = header_row + 1
    stt = 1
    total_calc = total_objects if total_objects > 0 else 1

    # Sorting items descending by count
    sorted_items = sorted(by_class.items(), key=lambda x: x[1], reverse=True) if by_class else []
    
    if not sorted_items:
        display_name = target_name or "Sản phẩm / Đồ vật"
        sorted_items = [(display_name, total_objects)]

    for class_name, count in sorted_items:
        ws.row_dimensions[current_row].height = 22
        pct = (count / total_calc) * 100 if total_calc > 0 else 0
        fill_color = COLOR_ROW_ALT if stt % 2 == 0 else "FFFFFF"
        row_fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")

        # Map english label names if applicable
        friendly_name = class_name
        category_label = "Mặt hàng"
        lower_name = class_name.lower()
        if "can" in lower_name:
            category_label = "Nước đóng lon"
        elif "bottle" in lower_name:
            category_label = "Nước đóng chai"
        elif "box" in lower_name:
            category_label = "Hộp / Thùng đóng gói"
        elif "phone" in lower_name or "điện thoại" in lower_name:
            category_label = "Thiết bị điện tử"
        elif "cup" in lower_name:
            category_label = "Ly / Cốc"

        c1 = ws.cell(row=current_row, column=1, value=stt)
        c1.alignment = Alignment(horizontal="center", vertical="center")
        
        c2 = ws.cell(row=current_row, column=2, value=friendly_name)
        c2.alignment = Alignment(horizontal="left", vertical="center")
        c2.font = Font(name=font_family, size=10, bold=True, color=COLOR_TEXT)

        c3 = ws.cell(row=current_row, column=3, value=category_label)
        c3.alignment = Alignment(horizontal="center", vertical="center")

        c4 = ws.cell(row=current_row, column=4, value=count)
        c4.alignment = Alignment(horizontal="right", vertical="center")
        c4.font = Font(name=font_family, size=10, bold=True, color=COLOR_PRIMARY)
        c4.number_format = "#,##0"

        c5 = ws.cell(row=current_row, column=5, value=f"{pct:.1f}%")
        c5.alignment = Alignment(horizontal="right", vertical="center")

        for col_i in range(1, 6):
            cell = ws.cell(row=current_row, column=col_i)
            cell.border = thin_border
            cell.fill = row_fill
            if col_i not in (2, 4):
                cell.font = Font(name=font_family, size=10, color=COLOR_TEXT)

        stt += 1
        current_row += 1

    # Total row
    ws.row_dimensions[current_row].height = 24
    total_fill = PatternFill(start_color=COLOR_TOTAL_BG, end_color=COLOR_TOTAL_BG, fill_type="solid")

    ws.cell(row=current_row, column=1, value="")
    c_tot_name = ws.cell(row=current_row, column=2, value="TỔNG CỘNG TẤT CẢ SẢN PHẨM")
    c_tot_name.font = Font(name=font_family, size=10, bold=True, color=COLOR_PRIMARY_DARK)
    c_tot_name.alignment = Alignment(horizontal="left", vertical="center")

    ws.cell(row=current_row, column=3, value="")

    c_tot_val = ws.cell(row=current_row, column=4, value=total_objects)
    c_tot_val.font = Font(name=font_family, size=11, bold=True, color=COLOR_PRIMARY_DARK)
    c_tot_val.alignment = Alignment(horizontal="right", vertical="center")
    c_tot_val.number_format = "#,##0"

    c_tot_pct = ws.cell(row=current_row, column=5, value="100.0%")
    c_tot_pct.font = Font(name=font_family, size=10, bold=True, color=COLOR_PRIMARY_DARK)
    c_tot_pct.alignment = Alignment(horizontal="right", vertical="center")

    for col_i in range(1, 6):
        cell = ws.cell(row=current_row, column=col_i)
        cell.border = double_bottom_border
        cell.fill = total_fill

    current_row += 2

    # 4. Optional Table 2: Unique tracking logs
    if detections:
        unique_tracks = {}
        for d in detections:
            tid = d.get("tracking_id")
            if tid is not None and tid not in unique_tracks:
                unique_tracks[tid] = d

        if unique_tracks:
            ws.merge_cells(f"A{current_row}:E{current_row}")
            det_title = ws[f"A{current_row}"]
            det_title.value = "II. NHẬT KÝ THEO DÕI VẬT THỂ CHI TIẾT (UNIQUE TRACKING ID)"
            det_title.font = Font(name=font_family, size=11, bold=True, color=COLOR_PRIMARY_DARK)
            ws.row_dimensions[current_row].height = 24
            current_row += 1

            det_headers = ["STT", "Mã Vật Thể (Track ID)", "Tên Vật Thể / Sản Phẩm", "Thời Điểm Xuất Hiện", "Độ Tin Cậy"]
            ws.row_dimensions[current_row].height = 24
            for col_idx, h_text in enumerate(det_headers, 1):
                cell = ws.cell(row=current_row, column=col_idx, value=h_text)
                cell.font = Font(name=font_family, size=9, bold=True, color="FFFFFF")
                cell.fill = PatternFill(start_color=COLOR_MUTED, end_color=COLOR_MUTED, fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = thin_border

            current_row += 1
            t_idx = 1
            for tid, item in sorted(unique_tracks.items(), key=lambda x: x[1].get("timestamp", 0)):
                ws.row_dimensions[current_row].height = 20
                ts = item.get("timestamp", 0)
                m, s = divmod(int(ts), 60)
                time_str = f"{m:02d}:{s:02d}"
                conf_pct = float(item.get("confidence", 0)) * 100
                d_fill = PatternFill(start_color="FFFFFF" if t_idx % 2 == 1 else COLOR_ROW_ALT, fill_type="solid")

                c1 = ws.cell(row=current_row, column=1, value=t_idx)
                c1.alignment = Alignment(horizontal="center", vertical="center")
                
                c2 = ws.cell(row=current_row, column=2, value=f"ID #{tid}")
                c2.alignment = Alignment(horizontal="center", vertical="center")
                c2.font = Font(name=font_family, size=9, bold=True)

                c3 = ws.cell(row=current_row, column=3, value=item.get("class_name", "Sản phẩm"))
                c3.alignment = Alignment(horizontal="left", vertical="center")

                c4 = ws.cell(row=current_row, column=4, value=time_str)
                c4.alignment = Alignment(horizontal="center", vertical="center")

                c5 = ws.cell(row=current_row, column=5, value=f"{conf_pct:.1f}%")
                c5.alignment = Alignment(horizontal="right", vertical="center")

                for col_i in range(1, 6):
                    cell = ws.cell(row=current_row, column=col_i)
                    cell.border = thin_border
                    cell.fill = d_fill
                    if col_i != 2:
                        cell.font = Font(name=font_family, size=9, color=COLOR_TEXT)

                t_idx += 1
                current_row += 1

    # Auto-adjust column widths
    column_widths = {
        "A": 8,   # STT
        "B": 38,  # Tên Sản Phẩm / Đồ Vật
        "C": 26,  # Nhãn Phân Loại
        "D": 22,  # Số Lượng
        "E": 18,  # Tỷ Lệ
    }
    for col_letter, width in column_widths.items():
        ws.column_dimensions[col_letter].width = width

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wb.save(output_path)
    logger.info("Excel counting report generated successfully: %s", output_path)
    return output_path

# Olist Customer Review Classification & Early Warning System

Dự án môn học **CSW439 - Khai phá dữ liệu (Data Mining)** - Nhóm 1.

## 📌 Tổng quan bài toán
Dự án xây dựng mô hình Học máy phân loại nhị phân (Binary Classification) để dự đoán một đơn hàng Olist sau khi giao thành công (`delivered`) sẽ nhận được đánh giá Tốt (Positive - 4, 5 sao) hay Xấu (Negative - 1, 2, 3 sao). 

Mục tiêu chính là **cảnh báo sớm các đơn hàng có nguy cơ nhận review xấu (Class 0)** để doanh nghiệp kịp thời thực hiện các chương trình chăm sóc khách hàng, cải thiện trải nghiệm người dùng. Do dữ liệu mất cân bằng nghiêm trọng (~21% Class 0), mô hình tập trung tối ưu hóa chỉ số **Recall** và **F1-score của Class 0**.

---

## 🛠️ Phương pháp và Cấu trúc dự án
Notebook gộp **4 mô hình chính** trên cùng một luồng tiền xử lý (Preprocessing Pipeline) chung sử dụng **14 đặc trưng** được trích xuất từ dữ liệu vận hành, tài chính và thông tin sản phẩm của Olist:
1. **Decision Tree** (Cây quyết định)
2. **K-Nearest Neighbors (KNN)** (K lân cận)
3. **Logistic Regression** (Hồi quy Logistic)
4. **Random Forest** (Rừng ngẫu nhiên)

### Các kỹ thuật cải thiện mô hình:
* **SMOTE & Undersampling**: Giải quyết bài toán mất cân bằng dữ liệu gốc.
* **Threshold Tuning (Tinh chỉnh ngưỡng quyết định)**: Điều chỉnh ngưỡng phân lớp phù hợp để tối ưu hóa khả năng bắt review xấu (tối ưu hóa Recall Class 0).

---

## 📊 Kết quả so sánh mô hình

| STT | Mô hình | Accuracy | Precision (Class 0) | Recall (Class 0) | F1-score (Class 0) | F1-weighted |
|---|---|---|---|---|---|---|
| 0 | Decision Tree (baseline) | 0.7356 | 0.3923 | 0.4681 | 0.4268 | 0.7437 |
| 1 | **Decision Tree (optimized)** | 0.6083 | 0.2980 | **0.6356** | 0.4057 | 0.6443 |
| 2 | KNN baseline (k=5) | 0.7947 | 0.5250 | 0.2538 | 0.3421 | 0.7656 |
| 3 | **KNN (SMOTE k=15 + threshold 0.45)** | 0.5824 | 0.2752 | **0.6034** | 0.3780 | 0.6209 |
| 4 | Logistic Regression (baseline) | 0.7450 | 0.3997 | 0.4230 | 0.4110 | 0.7476 |
| 5 | **Logistic Regression (SMOTE + threshold 0.55)** | 0.6493 | 0.3155 | **0.5707** | 0.4064 | 0.6786 |
| 6 | Random Forest (baseline) | 0.8193 | 0.7286 | 0.2248 | 0.3436 | 0.7792 |
| 7 | **Random Forest (undersampling + threshold 0.6)** | 0.5202 | 0.2667 | **0.7324** | 0.3911 | 0.5593 |

*Mô hình Random Forest khi kết hợp giảm mẫu ngẫu nhiên và dịch ngưỡng quyết định lên 0.6 đạt độ nhạy (Recall) cao nhất trong việc nhận diện đánh giá tiêu cực (73.24%).*

---

## 👥 Thành viên nhóm 1 & Mức độ đóng góp
* **Trần Quốc An** (22%) - Train model, làm slide, thuyết trình
* **Lê Minh Trí** (22%) - Train model, làm slide, thuyết trình
* **Nguyễn Phúc Hậu** (22%) - Train model, làm slide, thuyết trình
* **Đặng Cao Cường** (22%) - Train model, làm slide, thuyết trình
* **Lê Triết Huân** (12%) - Xử lý data

---

## 📂 Hướng dẫn chạy dự án
1. Clone repository này về máy.
2. Tải và giải nén bộ dữ liệu Olist để vào thư mục `data/` ở thư mục gốc của dự án.
3. Mở Jupyter Notebook và chạy toàn bộ các cell trong `Olist_Early_Warning_AllModels_Group_1.ipynb`.

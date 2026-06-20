## **PROJECT DESCRIPTION**

**1. Tổng quan**

    Ở lab 2, chúng ta đã giải quyết bài toán Khai phá Hành vi vận hành để tìm các các tập luật thể hiện mối quan hệ nhân quả giữa các yếu tố liên quan đến khâu vận hành (thời gian giao hàng, phương thức thanh toán,..) có ảnh hưởng đến sự hài lòng của khách hàng. Với mini-project này, chúng ta tiếp tục triển khai một bài toán với mục đích tương tự.

    **Vấn đề thực tế**: Olist đóng vai trò trung gian, nhận sản phẩm từ các nhà bán lẻ, bán hàng với thương hiệu Olist. Khi khách hàng có trải nghiệm không tốt (giao chậm, phí giao hàng đắt, hàng hỏng), họ có thể đánh giá 1 sao. Doanh nghiệp kỳ vọng dự đoán được nguy cơ khách hàng sẽ để lại đánh giá thấp để có thể kịp thời áp dụng các hoạt động chăm sóc tới đúng đối tượng khách hàng và đúng thời điểm.

    **Ý tưởng giải quyết**: Xây dựng một hệ thống Cảnh báo sớm. Ngay tại thời điểm hệ thống ghi nhận trạng thái đơn hàng là delivered (đã giao), mô hình Machine Learning sẽ lập tức phân tích các dữ liệu trong quá khứ của đơn hàng đó để dự đoán xem: Khách hàng này có nguy cơ để lại đánh giá xấu hay không?

    **Bài toán**: Xây dựng một mô hình Machine Learning phân loại nhị phân (Binary Classification) để dự đoán một đơn hàng sẽ nhận được đánh giá Tốt (Positive) hay Xấu (Negative) dựa trên các thông tin vận hành, sản phẩm và thanh toán.

**2. Tóm tắt nguồn dữ liệu và gợi ý thông tin cần khai thác**

    1. Bảng `reviews` (olist_order_reviews_dataset):
    - Đây là bảng quan trọng nhất vì nó chứa "Câu trả lời" (Target Variable) mà mô hình cần phải học.
    - Thông tin cần lấy: Mã đơn hàng (`order_id`) để liên kết dữ liệu và Điểm đánh giá (`review_score` từ 1 đến 5).
    
    2. Bảng `orders` (olist_orders_dataset):
    - Bảng này ghi lại toàn bộ các cột mốc thời gian trong vòng đời của một đơn hàng. Trải nghiệm chờ đợi là yếu tố tác động mạnh nhất đến tâm lý khách hàng.
    - Thông tin cần lấy: Trạng thái đơn hàng (`order_status`), Thời điểm mua (`order_purchase_timestamp`), Ngày giao hàng dự kiến (`order_estimated_delivery_date`), và Ngày giao hàng thực tế (`order_delivered_customer_date`).
    - Từ các mốc thời gian này, chúng ta sẽ tính toán được các đặc trưng vận hành cực kỳ giá trị như: Khách hàng đã phải chờ tổng cộng bao nhiêu ngày? Đơn hàng có bị giao trễ so với cam kết không? Nếu trễ thì trễ bao nhiêu ngày?

    3. Bảng `payments` và `items` (olist_order_items_dataset & olist_order_payments_dataset):
    - Thông tin cần lấy: Giá sản phẩm (`price`), Phí vận chuyển (`freight_value`), Phương thức thanh toán (`payment_type`), và Số kỳ trả góp (`payment_installments`).
    - Chúng ta sẽ ghép nối chúng lại để tính toán Tỷ lệ phí ship trên tổng giá trị đơn hàng. Đồng thời, việc khách hàng mua trả góp nhiều kỳ hay thanh toán một lần bằng tiền mặt cũng phản ánh mức độ kỳ vọng và áp lực tài chính của họ đối với món hàng.

    4. Bảng products (olist_products_dataset):
    - Thông tin cần lấy: Kích thước vật lý (chiều dài (`product_length_cm`), rộng (`product_width_cm`), cao (`product_height_cm`), cân nặng (`product_weight_g	`)), Số lượng hình ảnh minh họa (`product_photos_qty`), và Độ dài bài mô tả sản phẩm (`product_description_lenght`).
    - Từ kích thước, chúng ta sẽ tính ra Thể tích hàng hóa (hàng cồng kềnh dễ bị hư hỏng và giao chậm hơn). Số lượng ảnh và độ dài mô tả thể hiện tính minh bạch của người bán; thông tin càng đầy đủ, khách hàng càng ít bị "vỡ mộng" khi nhận hàng.

    5. Bảng customers (olist_customers_dataset):
    - Thông tin cần lấy: Bang của khách hàng (customer_state).
    - Khách hàng ở các bang xa xôi có thể có mức độ bao dung hoặc kỳ vọng về thời gian giao hàng khác biệt hoàn toàn so với khách hàng ở các trung tâm kinh tế lớn như Sao Paulo.

**3. Thiết kế Biến Mục tiêu (Target Variable)**

    - Vấn đề: Dữ liệu gốc (bảng `reviews`) cung cấp cột `review_score` từ 1 đến 5 sao. Tuy nhiên, dự đoán chính xác từng con số (Multiclass Classification) là rất khó và không cần thiết. Chúng ta chỉ cần biết khách "Hài lòng" hay "Không hài lòng".
    - Hành động: Chuyển đổi bài toán thành Phân loại nhị phân (Binary Classification).
        - Class 0 (Negative): Đánh giá 1, 2, 3 sao. Đại diện cho nhóm khách hàng có trải nghiệm dưới kỳ vọng hoặc gặp sự cố.
        - Class 1 (Positive): Đánh giá 4, 5 sao. Đại diện cho nhóm khách hàng hài lòng, quy trình vận hành suôn sẻ.

**4. Yêu cầu**

    - Làm việc theo nhóm
    - Sử dụng Jupyter Notebook
    - Cuối file notebook, cung cấp thông tin mức độ đóng góp của các thành viên.
    - Modeling: sử dụng ít nhất 4 models và so sánh kết quả.
    - Báo cáo project cần có:
        - Chuẩn bị slides báo cáo, không báo cáo bằng notebook.
        - Trình bày tóm tắt đến chi tiết quá trình thực hiện.
        - Giải thích lý do lựa chọn các kỹ thuật (nếu có).
        - Trình bày cấu trúc và cách hoạt động của các models mà nhóm lựa chọn.
        - So sánh kết quả các models và chỉ ra kết quả tốt nhất.
    - Không bắt buộc tất cả các thành viên đều báo cáo.
    - Khuyến khích tìm hiểu cách áp dụng các kỹ thuật xử lý dữ liệu nhằm cải thiện hiệu suất mô hình.
    - Khuyến khích tìm hiểu các mô hình ngoài phạm vi bài học và áp dụng.

**5. Nộp bài**

    - Tất cả thành viên trong nhóm đều nộp bài
    - Nộp toàn bộ project folder dưới dạng .zip (folder có đầy đủ file data, code)
    - Đặt tên file .zip là tên nhóm (VD: Group1.zip)

**Chấm điểm**: 

    - Có 2 cột điểm:
        - Bài làm (chấm qua file đã nộp)
        - Báo cáo (chấm trực tiếp vào ngày báo cáo)
    - Điểm mỗi thành viên được tính theo mức độ đóng góp mà nhóm cung cấp.

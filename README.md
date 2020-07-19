# THỰC TẬP CHUYÊN MÔN NĂM HỌC 2019-2020
Họ tên: Nguyễn Trung Kiên
Lớp: Công nghệ thông tin Khóa 58
## Đề tài: Nghiên cứu thuật toán CNN và ứng dụng trong việc nhận diện văn bản từ hình ảnh
## Phần 1: Công nghệ sử dụng:
- Nghiên cứu thuật toán qua ngôn ngữ lập trình python.
- Sử dụng thuật toán CNN để nhận diện văn bản từ hình ảnh.
- Sử dụng các thư viện sẵn có từ mã nguồn mở.
## Phần 2: Các chức năng đã hoàn thành:
1. Nhận dạng được chữ số từ hình ảnh.
## Phần 3: Các chức năng chưa hoàn thành:
1. Chưa nhận dạng được chữ tiếng việt và chữ cái khác ngoài chữ cái la-tinh.
2. Một số hình ảnh không nhận dạng đúng 100% chữ cái.
## Hướng dẫn sử dụng:
1. Đối với phần training - folder "Training"
  - Gồm 2 file python: main.py và model.py.
  - file main.py dùng để chạy training.
  - file model.py dùng để lưu mô hình network.
  - Chạy file main.py để bắt đầu thực hiện training.
2. Đối với phần nhận diện - folder "Convolution NN"
  - Thay đổi ảnh input đầu vào bằng cách thay đổi giá trị img_path
  - Chạy file ConvolutionNN.py để nhận dạng.
  - Mặc định là sẽ show tất cả các filter, có thể show từng filter bằng cách bỏ comment phần "Show từng filter".
  - Bức ảnh có thể qua nhiều filter để nhận dạng tốt hơn bằng cách bỏ comment phần "Qua các filter".
  - Có thể thêm các filter bằng cách thêm một mảng 2 chiều 3 * 3 bằng hàm np.array (như ví dụ các filter đã có trên code), sau đó thêm vào filters.
  
Xin cảm ơn!

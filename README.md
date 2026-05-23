# Predictive Models for Healthcare Data

# Mô hình dự đoán trong dữ liệu chăm sóc sức khỏe

## Tổng quan

Đề tài tập trung nghiên cứu và đánh giá hiệu quả của các mô hình học máy trên dữ liệu chăm sóc sức khỏe (healthcare data), đặc biệt đối với các đặc tính phổ biến của dữ liệu y tế như:

* Tính phi tuyến (nonlinearity)
* Dữ liệu số chiều cao (high-dimensional data)
* Mất cân bằng lớp (imbalanced data)

Nghiên cứu sử dụng hai bộ dữ liệu:

1. Breast Cancer Gene Expression Dataset
2. Diabetes Dataset

Thông qua quá trình trực quan hóa dữ liệu, phân cụm, xử lý mất cân bằng và phân loại, đề tài nhằm khảo sát khả năng dự đoán của nhiều mô hình Machine Learning trong các bài toán y tế.

```Predictive-models-for-healthcare-data/
│
├── breast/                          # Breast cancer dataset pipeline
│   ├── datasets/
│   │   └── Breast_GSE45827.csv      # Gene expression dataset (GSE45827)
│   ├── configs/
│   │   ├── breast.yaml              # Config cho breast dataset
│   │   └── icmr.yaml                # Config tham khảo (dataset cũ)
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── feature_analysis.py
│   │   ├── clustering.py
│   │   ├── classification.py
│   │   ├── evaluation.py
│   │   ├── explainability.py
│   │   └── pipeline.py
│   └── breast-2305.ipynb            # Notebook thực nghiệm
│
├── diabetes/                        # Diabetes dataset pipeline
│   ├── archive/
│   │   ├── diabetes_012_health_indicators_BRFSS2015.csv
│   │   ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   └── main.ipynb                   # Notebook thực nghiệm
│
├── report/                          # LaTeX thesis report
│   ├── Section/
│   │   ├── introduction.tex
│   │   ├── imbalance.tex
│   │   └── theoretical formulation.tex
│   ├── Title/
│   │   └── title.tex
│   ├── references.bib
│   └── main.tex
│
└── README.md```

---

# Mục tiêu nghiên cứu

Đề tài hướng đến các mục tiêu sau:

* Khảo sát đặc điểm của dữ liệu y tế.
* Phân tích cấu trúc phi tuyến trong dữ liệu gene expression.
* Xử lý vấn đề mất cân bằng dữ liệu trong bài toán dự đoán bệnh tiểu đường.
* So sánh hiệu năng của nhiều mô hình học máy khác nhau.
* Đánh giá ảnh hưởng của các kỹ thuật clustering và imbalance handling đến kết quả dự đoán.

---

# Các mô hình sử dụng

Các mô hình classification được sử dụng trong đề tài bao gồm:

* Logistic Regression
* Support Vector Machine (SVM) với kernel RBF
* Random Forest
* XGBoost
* Multi-layer Perceptron (MLP)

---

# Bộ dữ liệu 1: Breast Cancer Gene Expression

## Mô tả

Bộ dữ liệu breast cancer chứa dữ liệu biểu hiện gene (gene expression) với số chiều cao và cấu trúc phức tạp.

Bộ dữ liệu này được sử dụng để:

* Phân tích cấu trúc dữ liệu
* Khảo sát tính phi tuyến
* Thực hiện phân cụm
* Đánh giá hiệu quả classification

---

## Quy trình thực hiện

### 1. Tiền xử lý dữ liệu

* Xử lý missing values
* Chuẩn hóa dữ liệu
* Encoding nhãn

---

### 2. Giảm chiều và trực quan hóa dữ liệu

Các kỹ thuật được sử dụng:

* PCA
* t-SNE
* UMAP

Mục đích:

* Quan sát sự phân bố dữ liệu
* Khảo sát xu hướng phân tách giữa các lớp
* Phân tích cấu trúc phi tuyến trong không gian đặc trưng nhiều chiều

---

### 3. Phân cụm dữ liệu

Các thuật toán clustering được sử dụng:

* KMeans
* DBSCAN
* Gaussian Mixture Model (GMM)
* Hierarchical Clustering

Các độ đo đánh giá:

* Silhouette Score
* Davies-Bouldin Index
* Visualization kết quả phân cụm

---

### 4. Classification

Các mô hình classification được huấn luyện và so sánh trên bộ dữ liệu.

Độ đo đánh giá:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

# Bộ dữ liệu 2: Diabetes Dataset

## Mô tả

Bộ dữ liệu diabetes có đặc điểm mất cân bằng lớp, trong đó số lượng mẫu giữa các lớp chênh lệch đáng kể.

Bộ dữ liệu này được sử dụng để:

* Phân tích ảnh hưởng của imbalance data
* Đánh giá hiệu quả của các kỹ thuật resampling
* So sánh khả năng dự đoán sau khi cân bằng dữ liệu

---

## Quy trình thực hiện

### 1. Tiền xử lý dữ liệu

* Làm sạch dữ liệu
* Chuẩn hóa đặc trưng
* Xử lý dữ liệu thiếu

---

### 2. Xử lý mất cân bằng dữ liệu

Các kỹ thuật được sử dụng:

* SMOTE
* SMOTE + ENN

Phương pháp cho kết quả tốt nhất sẽ được sử dụng trong pipeline classification tiếp theo.

---

### 3. Classification

Dữ liệu sau khi xử lý imbalance sẽ được sử dụng để huấn luyện:

* Logistic Regression
* SVM RBF
* Random Forest
* XGBoost
* MLP

Các độ đo đánh giá:

* F1-score
* ROC-AUC
* Confusion Matrix

---

# Quy trình thực nghiệm tổng quát

```text id="pipeline-healthcare"
Load Dataset
      ↓
Tiền xử lý dữ liệu
      ↓
Visualization (PCA / t-SNE / UMAP)
      ↓
Clustering Analysis
      ↓
Imbalance Handling (Diabetes Dataset)
      ↓
Classification
      ↓
Evaluation & Comparison
```

---

# Công nghệ và thư viện sử dụng

## Ngôn ngữ lập trình

* Python

---

## Thư viện chính

### Xử lý dữ liệu

* NumPy
* Pandas
* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Giảm chiều dữ liệu

* PCA
* t-SNE
* UMAP

### Clustering

* KMeans
* DBSCAN
* Gaussian Mixture
* Hierarchical Clustering

### Classification

* Logistic Regression
* SVM
* Random Forest
* XGBoost
* MLPClassifier

### Imbalanced Learning

* imbalanced-learn
* SMOTE
* SMOTEENN

---

# Kết quả kỳ vọng

Đề tài hướng đến:

* So sánh hiệu quả giữa các mô hình học máy
* Phân tích ảnh hưởng của dữ liệu phi tuyến
* Đánh giá chất lượng phân cụm trên dữ liệu y tế
* Khảo sát tác động của imbalance handling
* Xác định mô hình phù hợp cho bài toán dự đoán trong healthcare data

---

# Hướng phát triển

Một số hướng phát triển trong tương lai:

* Deep Learning cho dữ liệu y tế
* Explainable AI (SHAP / LIME)
* Hyperparameter Optimization
* Multi-omics Data
* Federated Learning trong healthcare systems

---

# Kết luận

Đề tài nghiên cứu các mô hình dự đoán trên dữ liệu chăm sóc sức khỏe với hai thách thức chính:

* Tính phi tuyến của dữ liệu
* Mất cân bằng lớp

Thông qua trực quan hóa dữ liệu, phân cụm, xử lý imbalance và classification, nghiên cứu cung cấp cái nhìn tổng quan về hiệu quả của nhiều mô hình Machine Learning trong các bài toán dự đoán y tế.

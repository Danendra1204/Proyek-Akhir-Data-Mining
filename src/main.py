import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def attributeSel(path):
    data = pd.read_excel(path)  
    data = data.drop(columns=["NAMA"])

    # mencari missing value dan mengisinya dengan nilai rata-rata

    for i in data.columns:
        if data[i].isnull().any():
            if data[i].dtype in ['float64', 'int64']:
                data[i] = data[i].fillna(round(data[i].mean()))
            else:
                data[i] = data[i].fillna(data[i].mode()[0])

    # ubah nilai dari coumn Jenis Kelamin, Status Mahasiswa, Status Nikah, Status Kelulusan menjadi 0-1

    label_cols = ["JENIS KELAMIN", "STATUS MAHASISWA", "STATUS NIKAH", "STATUS KELULUSAN"]
    le = LabelEncoder()
    for col in label_cols:
        data[col] = le.fit_transform(data[col])

    # normalisasi data numerik menggunakan Min-Max Scaling

    scaler = MinMaxScaler()
    num_cols = ["UMUR"] + [f"IPS {i}" for i in range(1, 9)] + ["IPK "]
    data[num_cols] = scaler.fit_transform(data[num_cols])

    print(data.head())

    return data


data_2020 = attributeSel("./db/Kelulusan Mahasiswa.xlsx")

X = data_2020.drop(columns=["STATUS KELULUSAN"])
y = data_2020["STATUS KELULUSAN"]

# split data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# hasil
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))



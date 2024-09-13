import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
train=pd.read_csv("./data/space-titanic/train.csv")
test=pd.read_csv("./data/space-titanic/test.csv")
sub_df=pd.read_csv("./data/space-titanic/sample_submission.csv")

## 필요없는 칼럼 제거
all_data=pd.concat([train, test])
all_data=all_data.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼
c = all_data.columns[all_data.dtypes == object][:-1]

# 정수형 전처리
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in c:
    all_data[i] = le.fit_transform(all_data[i])


# 결측치 처리
all_data.fillna(-99, inplace=True)

# train / test 데이터셋
train_n=len(train)
train=all_data.iloc[:train_n,]
test=all_data.iloc[train_n:,]

# 타겟 변수 분리
y = train['Transported'].astype("bool")

# OneHotEncoder 설정
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first',
                              handle_unknown='ignore'), c)
    ], remainder='passthrough')

# 학습 데이터와 테스트 데이터에 전처리 적용
X_train = preprocessor.fit_transform(train.drop(['Transported'], axis=1))
X_test = preprocessor.transform(test)

# 모델 생성 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y)

# 예측 수행
predictions = model.predict(X_test)

# Transported 바꾸기
sub_df["Transported"] = predictions
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/space-titanic/sample_submission_dummies.csv", index=False)
import pandas as pd
import re
from soynlp.normalizer import repeat_normalize

# 1. 데이터 불러오기
df = pd.read_csv("./data/cleaned_reviews.csv")

# 2. 이모티콘/반복 문자/특수기호 정제 함수
def clean_text(text):
    # 1) 줄바꿈, 탭, 유니코드 이상문자 제거
    text = re.sub(r"[\n\t\r\xa0]", " ", str(text))
    
    # 2) 특수문자, 이모지 제거 (이모지 제거는 유니코드 블록 제외)
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text)
    
    # 3) 반복 문자 정규화 (e.g., "ㅋㅋㅋㅋㅋ" → "ㅋㅋ")
    text = repeat_normalize(text, num_repeats=2)
    
    # 4) 연속 공백 하나로 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# 3. 정제 적용
df["cleaned"] = df["content"].apply(clean_text)

# 4. 저장
df.to_csv("./data/cleaned_reviews_strong.csv", index=False)
print("✅ 정제 강화 완료! → cleaned_reviews_strong.csv 저장됨")

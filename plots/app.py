import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from collections import Counter
from konlpy.tag import Okt
import platform
import matplotlib.font_manager as fm

# 📌 한글 폰트 설정 (matplotlib + wordcloud)
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    font_path = '/System/Library/Fonts/AppleGothic.ttf'
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 🌟 Streamlit 기본 설정
st.set_page_config(page_title="감정 분석 & 워드클라우드", layout="wide")

# 모델 로드
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./models/koelectra_sentiment_model_tf_updated")
    model = TFAutoModelForSequenceClassification.from_pretrained("./models/koelectra_sentiment_model_tf_updated")
    return tokenizer, model

tokenizer, model = load_model()

# 감정 예측 함수
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    pred = tf.argmax(probs, axis=1).numpy()[0]
    confidence = float(tf.reduce_max(probs))
    label_map = {0: '부정 😠', 1: '중립 😐', 2: '긍정 🙂'}
    return label_map[pred], confidence * 100

# 워드클라우드 함수
def draw_wordcloud(df, label=None):
    okt = Okt()
    if label is not None:
        df = df[df['label'] == label]
    if df.empty:
        st.warning("선택한 감정에 해당하는 데이터가 없습니다.")
        return
    text = " ".join(df['content'].astype(str))
    nouns = okt.nouns(text)
    freq = Counter(nouns)
    wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=400
    ).generate_from_frequencies(freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# 🔹 탭 구성
tab1, tab2, tab3 = st.tabs(["감정 분석기", "워드클라우드", "분석 리포트"])

# 탭 1: 감정 분석기
with tab1:
    st.title("리뷰 감정 분석기")
    st.write("KoELECTRA 기반 감정 분석 모델을 사용합니다.")
    user_input = st.text_area("리뷰를 입력하세요")
    if st.button("감정 분석"):
        label, conf = predict_sentiment(user_input)
        st.success(f"예측 감정: {label}")
        st.caption(f"신뢰도: {conf:.2f}%")

# 탭 2: 워드클라우드
with tab2:
    st.title("CSV 기반 워드클라우드 생성")
    uploaded = st.file_uploader("CSV 파일 업로드 (컬럼명: content, label)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'content' not in df.columns or 'label' not in df.columns:
            st.error("CSV 파일에 'content' 또는 'label' 컬럼이 없습니다.")
        else:
            label_options = ["전체"] + list(df['label'].dropna().unique())
            selected = st.selectbox("감정 필터링", label_options)
            label = None if selected == "전체" else selected
            draw_wordcloud(df, label)

# 탭 3: 분석 리포트
with tab3:
    st.title("감정 분포 및 통계 시각화")
    if uploaded and 'content' in df.columns and 'label' in df.columns:
        st.subheader("감정 비율")
        chart_data = df['label'].value_counts().reset_index()
        chart_data.columns = ['감정', '갯수']
        fig, ax = plt.subplots()
        sns.barplot(x='감정', y='갯수', data=chart_data, ax=ax)
        st.pyplot(fig)

        st.subheader("상위 키워드 테이블")
        okt = Okt()
        nouns = okt.nouns(" ".join(df['content'].astype(str)))
        freq = Counter(nouns)
        freq_df = pd.DataFrame(freq.most_common(30), columns=['단어', '빈도수'])
        st.dataframe(freq_df)

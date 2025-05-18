import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import shap
from streamlit_shap import st_shap

# 페이지 설정 및 기본 레이아웃
st.set_page_config(
    page_title="공정 실시간 모니터링",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS 적용
st.markdown("""
<style>
    /* 전체 폰트 및 배경 스타일 */
    .main {
        background-color: #f8f9fa;
        font-family: 'Malgun Gothic', sans-serif;
    }
    
    /* 헤더 스타일링 */
    h1 {
        color: #1E88E5;
        font-weight: 700;
        padding-bottom: 20px;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 30px;
    }
    
    h2, h3 {
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    
    /* 카드 스타일 컨테이너 */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    /* 예측 결과 스타일 */
    .prediction-ok {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    
    .prediction-ng {
        background-color: #F44336;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    
    /* 사이드바 스타일링 */
    .sidebar .sidebar-content {
        background-color: #f1f3f6;
    }
    
    /* 슬라이더 스타일 */
    .stSlider {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    /* 데이터프레임 스타일 */
    .dataframe {
        font-size: 14px;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우 '맑은 고딕'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['figure.figsize'] = (10, 6)  # 기본 그래프 크기 설정
plt.rcParams['axes.grid'] = True  # 그리드 표시
plt.rcParams['axes.facecolor'] = '#f9f9f9'  # 그래프 배경색

# 헤더 영역
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/fluency/96/000000/factory.png", width=80)
with col2:
    st.title('공정 실시간 모니터링')

st.markdown("""
<div class="stCard">
    <p>이 시스템은 입력된 공정 파라미터를 기반으로 품질(OK/NG)을 예측합니다. 
    장력1, 기준장력, 스피드1 값을 입력하면 예측 결과와 확률, 그리고 SHAP 값을 통한 예측 설명을 확인할 수 있습니다.</p>
</div>
""", unsafe_allow_html=True)

# 모델 불러오기
@st.cache_resource
def load_model():
    try:
        model = joblib.load('./decision_tree_model.pkl')
        return model
    except:
        st.error("모델 파일을 찾을 수 없습니다. 먼저 모델을 학습시켜 저장해주세요.")
        return None

model = load_model()

# 학습 데이터 로드 (전체 데이터 및 ID 조회용)
@st.cache_data
def load_full_train_data():
    try:
        df = pd.read_csv('./track2_train_participant.csv')
        # 파생 변수 계산
        df['초기장력차이'] = df['기준장력'] - df['장력1']
        return df
    except:
        st.warning("학습 데이터를 찾을 수 없습니다.")
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'id': range(100),
            '스피드1': np.random.uniform(0, 250, 100),
            '장력1': np.random.uniform(0, 500, 100),
            '기준장력': np.random.uniform(0, 500, 100),
            'OK': np.random.choice([0, 1], 100),
            '초기장력차이': np.random.uniform(-200, 200, 100)
        })
        return sample_data

train_data = load_full_train_data()

# 사이드바 설정
st.sidebar.image("https://img.icons8.com/color/96/000000/settings.png", width=50)
st.sidebar.title('입력 방식 선택')
st.sidebar.markdown("---")

# 입력 방식 선택
input_method = st.sidebar.radio(
    "데이터 입력 방식",
    ["파라미터 직접 입력", "ID로 데이터 조회"]
)

if input_method == "파라미터 직접 입력":
    st.sidebar.subheader('파라미터 입력')
    
    def user_input_features():
        장력1 = st.sidebar.slider('장력1', 0.0, 500.0, 250.0, 5.0, 
                               help="제품의 초기 장력값을 입력하세요 (0-500)")
        
        기준장력 = st.sidebar.slider('기준장력', 0.0, 500.0, 320.0, 5.0,
                                 help="제품의 기준 장력값을 입력하세요 (0-500)")
        
        스피드1 = st.sidebar.slider('스피드1', 0.0, 250.0, 120.0, 5.0,
                                 help="제품의 초기 스피드값을 입력하세요 (0-250)")
        
        # 파생 변수 계산
        초기장력차이 = 기준장력 - 장력1
        
        data = {
            '장력1': 장력1,
            '스피드1': 스피드1,
            '초기장력차이': 초기장력차이
        }
        
        features = pd.DataFrame(data, index=[0])
        return features, 기준장력

    input_df, 기준장력 = user_input_features()
    
    # 리셋 버튼
    if st.sidebar.button('입력값 초기화'):
        st.experimental_rerun()

else:  # ID로 데이터 조회
    st.sidebar.subheader('ID로 데이터 조회')
    
    # 가능한 ID 목록 생성
    available_ids = sorted(train_data['id'].dropna().astype(int).unique())
    
    if available_ids:
        selected_id = st.sidebar.selectbox(
            "데이터 ID 선택", 
            available_ids,
            help="조회할 데이터의 ID를 선택하세요"
        )
        
        # 선택한 ID의 데이터 가져오기
        selected_data = train_data[train_data['id'] == selected_id].copy()
        
        if not selected_data.empty:
            # 필요한 컬럼만 선택
            input_columns = ['장력1', '스피드1']
            input_df = selected_data[input_columns].copy()
            
            # 기준장력 값 가져오기
            기준장력 = selected_data['기준장력'].values[0] if '기준장력' in selected_data.columns else 0.0
            
            # 파생 변수 계산
            input_df['초기장력차이'] = 기준장력 - input_df['장력1']
            
            # 전체 데이터 표시용
            st.sidebar.markdown("---")
            st.sidebar.subheader("선택한 ID의 전체 데이터")
            st.sidebar.dataframe(selected_data[['장력1', '기준장력', '스피드1', '초기장력차이', 'OK']])
        else:
            st.sidebar.error(f"ID {selected_id}에 해당하는 데이터를 찾을 수 없습니다.")
            input_df = pd.DataFrame()
            기준장력 = 0.0
    else:
        st.sidebar.error("사용 가능한 ID가 없습니다.")
        input_df = pd.DataFrame()
        기준장력 = 0.0

st.sidebar.markdown("---")
st.sidebar.info("© 2025 공정 모니터링 시스템 v1.0")

# 메인 콘텐츠 영역 - 입력 데이터가 있는 경우에만 표시
if not input_df.empty:
    # 입력 데이터 표시
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader('📝 입력된 파라미터')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="장력1", value=f"{input_df['장력1'].values[0]:.1f}")
    with col2:
        st.metric(label="기준장력", value=f"{기준장력:.1f}")
    with col3:
        st.metric(label="스피드1", value=f"{input_df['스피드1'].values[0]:.1f}")
    
    st.markdown("**파생 변수:**")
    st.metric(label="초기장력차이", value=f"{input_df['초기장력차이'].values[0]:.1f}")
    
    st.markdown("**전체 입력 데이터:**")
    st.dataframe(input_df.style.background_gradient(cmap='Blues'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 예측 수행
    if model is not None:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader('🎯 예측 결과')
        
        try:
            # 모델에 필요한 특성만 선택
            if hasattr(model, 'feature_names_in_'):
                # 모델이 학습된 특성 이름 확인
                feature_names = model.feature_names_in_
                st.info(f"모델이 학습된 특성: {', '.join(feature_names)}")
                
                # 필요한 특성만 포함하는 데이터프레임 생성
                prediction_df = pd.DataFrame()
                for feature in feature_names:
                    if feature in input_df.columns:
                        prediction_df[feature] = input_df[feature]
                    else:
                        st.warning(f"특성 '{feature}'가 입력 데이터에 없습니다. 0으로 대체합니다.")
                        prediction_df[feature] = 0
            else:
                # 기본 특성 사용
                prediction_df = input_df.copy()
            
            # 예측 수행
            prediction = model.predict(prediction_df)
            prediction_proba = model.predict_proba(prediction_df)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 1:
                    st.markdown('<div class="prediction-ok">예측 결과: OK ✅</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-ng">예측 결과: NG ❌</div>', unsafe_allow_html=True)
            
            with col2:
                # 확률 표시
                prob_df = pd.DataFrame({
                    'NG 확률': [f"{prediction_proba[0][0]:.2%}"],
                    'OK 확률': [f"{prediction_proba[0][1]:.2%}"]
                })
                st.dataframe(prob_df.style.highlight_max(axis=1, color='lightgreen'), use_container_width=True)
            
            # 확률 시각화
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(['NG', 'OK'], prediction_proba[0], color=['#F44336', '#4CAF50'])
            ax.set_ylabel('확률', fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('예측 확률 분포', fontsize=14, fontweight='bold')
            
            # 바 위에 확률값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.info("모델이 학습된 특성과 입력 데이터의 특성이 일치하지 않습니다. 모델을 다시 학습시키거나 입력 데이터를 수정해주세요.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # SHAP 값 계산 및 시각화
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader('🧩 SHAP 값 분석')
        st.markdown("SHAP 값은 각 특성이 예측 결과에 어떻게 기여했는지 보여줍니다.")
        
        # 학습 데이터 로드 (SHAP 계산용 배경 데이터)
        @st.cache_data
        def load_train_data_for_shap():
            try:
                df = pd.read_csv('./track2_train_participant.csv')
                # 필요한 특성만 선택
                result_df = df[['장력1', '스피드1']].copy()
                # 파생 변수 계산
                result_df['초기장력차이'] = df['기준장력'] - df['장력1']
                return result_df
            except:
                st.warning("학습 데이터를 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
                # 샘플 데이터 생성 (필요한 특성만)
                sample_data = pd.DataFrame({
                    '장력1': np.random.uniform(0, 500, 100),
                    '스피드1': np.random.uniform(0, 250, 100),
                    '초기장력차이': np.random.uniform(-200, 200, 100)
                })
                return sample_data
        
        background_data = load_train_data_for_shap()
        
        # SHAP 계산
        try:
            with st.spinner('SHAP 값을 계산 중입니다...'):
                # 트리 기반 모델용 SHAP 계산
                explainer = shap.TreeExplainer(model)
                
                # SHAP 포스 플롯
                st.markdown("#### SHAP 포스 플롯")
                st.markdown("각 특성이 예측에 미치는 영향을 보여줍니다.")
                
                # 이진 분류 모델인 경우 클래스별 SHAP 값 계산
                shap_values = explainer.shap_values(prediction_df)
                
                # 클래스 인덱스 선택 (1은 OK 클래스, 0은 NG 클래스)
                class_idx = 1  # OK 클래스 선택
                
                # 방법 1: streamlit-shap 라이브러리 사용 (권장)
                try:
                    if isinstance(shap_values, list):  # 이진 분류 모델
                        st_shap(shap.force_plot(
                            explainer.expected_value[class_idx],
                            shap_values[class_idx][0],
                            prediction_df.iloc[0],
                            matplotlib=False
                        ), height=200)
                    else:
                        st_shap(shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            prediction_df.iloc[0],
                            matplotlib=False
                        ), height=200)
                except Exception as e1:
                    st.warning(f"streamlit-shap 사용 중 오류: {e1}")
                    
                    try:
                        # 방법 2: matplotlib 백엔드 사용
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if isinstance(shap_values, list):  # 이진 분류 모델
                            shap.force_plot(
                                explainer.expected_value[class_idx],
                                shap_values[class_idx][0],
                                prediction_df.iloc[0],
                                matplotlib=True,
                                show=False
                            )
                        else:
                            shap.force_plot(
                                explainer.expected_value,
                                shap_values[0],
                                prediction_df.iloc[0],
                                matplotlib=True,
                                show=False
                            )
                        
                        st.pyplot(fig)
                    except Exception as e2:
                        st.error(f"matplotlib 사용 중 오류: {e2}")
                        
                        # 방법 3: 특성 중요도 막대 그래프로 대체
                        st.error("SHAP 포스 플롯을 표시할 수 없습니다. 특성 중요도 막대 그래프로 대체합니다.")
                        
                        if isinstance(shap_values, list):
                            feature_importance = np.abs(shap_values[class_idx][0]).mean(0)
                        else:
                            feature_importance = np.abs(shap_values[0]).mean(0)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sorted_idx = np.argsort(feature_importance)
                        ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                        ax.set_yticks(range(len(sorted_idx)))
                        ax.set_yticklabels(prediction_df.columns[sorted_idx])
                        ax.set_title('특성 중요도 (SHAP 값 기반)')
                        st.pyplot(fig)
                
                # 전체 데이터에 대한 SHAP 값 계산
                st.markdown("#### 전체 데이터에 대한 SHAP 분석")
                st.markdown("이 분석은 모델 전체의 특성 중요도를 보여줍니다.")
                
                # 샘플링하여 계산 속도 향상
                sample_size = min(100, len(background_data))
                background_sample = background_data.sample(sample_size, random_state=42)
                
                # 전체 데이터에 대한 SHAP 값 계산
                background_shap_values = explainer.shap_values(background_sample)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### SHAP 비스웜 플롯")
                    st.markdown("각 특성의 SHAP 값 분포를 보여줍니다.")
                    
                    # 비스웜 플롯
                    fig, ax = plt.subplots(figsize=(10, 8))
                    if isinstance(background_shap_values, list):  # 이진 분류 모델
                        shap.summary_plot(
                            background_shap_values[class_idx],
                            background_sample,
                            plot_type="dot",
                            show=False
                        )
                    else:
                        shap.summary_plot(
                            background_shap_values,
                            background_sample,
                            plot_type="dot",
                            show=False
                        )
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("##### SHAP 특성 중요도")
                    st.markdown("특성들의 평균 절대 SHAP 값을 기준으로 한 중요도입니다.")
                    
                    # 바 플롯
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if isinstance(background_shap_values, list):  # 이진 분류 모델
                        shap.summary_plot(
                            background_shap_values[class_idx],
                            background_sample,
                            plot_type="bar",
                            show=False
                        )
                    else:
                        shap.summary_plot(
                            background_shap_values,
                            background_sample,
                            plot_type="bar",
                            show=False
                        )
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP 계산 중 오류: {e}")
            
            # 대체 시각화: 모델의 feature_importances_
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 모델 특성 중요도 (대체)")
                importances = model.feature_importances_
                indices = np.argsort(importances)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [prediction_df.columns[i] for i in indices])
                plt.xlabel('특성 중요도')
                plt.title('모델의 특성 중요도')
                st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

# 푸터
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #666;">
    <p>© 2025 공정 실시간 모니터링 시스템 | 버전 1.0</p>
</div>
""", unsafe_allow_html=True)

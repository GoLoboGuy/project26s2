#
# ===================================================
# News Keyword Visualizer V3
# ---------------------------------------------------
#
# - 역할별 함수 분리 
#   (API / 크롤링 / 전처리 / 분석 / 시각화 / 다운로드 / UI)
#
# - 예외 상황 방어 강화 (앱이 죽지 않도록 처리)
#   * API 인증 실패 처리(401/403 등)
#   * 네트워크 오류 처리(timeout, connection error 등)
#   * 크롤링 실패 시 skip 처리
#   * 데이터 부족 시 사용자 안내 강화
#    if res.status_code != 200: st.error("API 요청 실패")
#
# ===================================================
#

import json
import re
import pickle
import html
from datetime import datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from urllib.parse import quote
import zipfile

import requests as rq
import bs4
import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from wordcloud import WordCloud
from streamlit_lottie import st_lottie

from sklearn.feature_extraction.text import TfidfVectorizer
from soynlp.noun import LRNounExtractor_v2


# ===================================================
# 0) 전역 설정(폰트, 경로)
# ===================================================
FONT_PATH = "./resources/NanumSquareR.ttf"
STOPWORDS_PATH = "./resources/stopwords_ko.txt"
TOKENIZER_PATH = "./resources/my_tokenizer1.model"

LOTTIE_PATH = "./resources/lottie-full-movie-experience-including-music-news-video-weather-and-lots-of-entertainment.json"

MASK_BG = {
    "없음": "./resources/background_0.png",
    "타원": "./resources/background_1.png",
    "말풍선": "./resources/background_2.png",
    "하트": "./resources/background_3.png",
}


def setup_matplotlib_korean_font() -> None:
    """
    막대차트 등 matplotlib 출력에서 한글이 깨지지 않게 폰트를 설정합니다.
    폰트 파일이 없으면 윈도우 기본 폰트(Malgun Gothic)로 fallback 합니다.
    """
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    except Exception:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ===================================================
# 1) 로딩/캐시 함수들(리소스)
# ===================================================
def load_json(path: str) -> dict:
    """json 파일을 안전하게 로드합니다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"JSON 로드 실패: {path} ({e})")
        return {}


@st.cache_data(show_spinner=False)
def load_stopwords_file(path: str) -> set[str]:
    """불용어 파일을 읽어 set으로 반환합니다. 파일이 없으면 빈 set."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}
    except FileNotFoundError:
        return set()
    except Exception:
        return set()


@st.cache_resource
def load_tokenizer():
    """사전학습 토크나이저(pickle)를 로드합니다. 실패 시 None."""
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"토크나이저 로드 실패: {e}")
        return None


# ===================================================
# 2) 텍스트/문자열 유틸
# ===================================================
def clean_title(raw_title: str) -> str:
    """네이버 뉴스 title은 <b> 태그가 섞여오는 경우가 많아 제거합니다."""
    t = html.unescape(raw_title or "")
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def format_pubdate(pub_date: str) -> str:
    """RFC 날짜 형식을 사람이 읽기 쉬운 포맷으로 변환합니다."""
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pub_date or ""


@st.cache_data(show_spinner=False)
def clean_text_keep_korean(text: str) -> str:
    """
    한글 중심으로 정제합니다.
    - 숫자/영문/특수문자 제거
    - 공백 정리
    """
    text = re.sub(r"\d|[a-zA-Z]|\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_token(t: str) -> str:
    """토큰 정규화(기호/공백 제거)."""
    if t is None:
        return ""
    t = str(t).strip()
    t = re.sub(r"[\"'“”‘’\(\)\[\]\{\},\.\!\?\:\;]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def build_final_keyword(category: str, user_keyword: str) -> str:
    """
    분야 + 사용자 키워드를 결합합니다.
    - 공백을 1개로 정리
    - 검색 안정성을 위해 '분야 + 공백 + 키워드' 형태를 사용
    """
    category = (category or "").strip()
    user_keyword = re.sub(r"\s+", " ", (user_keyword or "")).strip()
    return f"{category} {user_keyword}".strip()


def safe_filename(s: str) -> str:
    """
    파일명에 들어가면 위험한 문자들을 '_'로 치환합니다.
    """
    s = s.strip()
    s = re.sub(r"[^\w\-가-힣]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "result"


# ===================================================
# 3) 네이버 API 통신(방어 코드 포함)
# ===================================================
def naver_news_api_request(keyword: str, display: int, start: int, client_id: str, client_secret: str):
    """
    네이버 뉴스 검색 API 호출.
    - 인증 실패/네트워크 오류/HTTP 오류를 처리해서 앱이 죽지 않게 함
    - 실패 시 빈 리스트 반환
    """
    if not client_id.strip() or not client_secret.strip():
        st.error("API 인증 정보(Client ID/Secret)가 비어 있습니다.")
        return []

    url = f"https://openapi.naver.com/v1/search/news.json?query={quote(keyword)}&display={display}&start={start}"
    headers = {
        "X-Naver-Client-Id": client_id.strip(),
        "X-Naver-Client-Secret": client_secret.strip(),
    }

    try:
        res = rq.get(url, headers=headers, timeout=10)
    except rq.exceptions.Timeout:
        st.error("네트워크 오류: 요청 시간이 초과되었습니다(timeout).")
        return []
    except rq.exceptions.ConnectionError:
        st.error("네트워크 오류: 서버에 연결할 수 없습니다(ConnectionError).")
        return []
    except rq.exceptions.RequestException as e:
        st.error(f"네트워크 오류: {e}")
        return []

    # ✅ 요구사항 반영: 상태코드가 200이 아니면 안내
    if res.status_code != 200:
        st.error("API 요청 실패")  # 요구사항 문구
        # 인증 관련이면 더 친절하게
        if res.status_code in (401, 403):
            st.warning("API 인증 실패(권한/키 오류). Client ID/Secret을 확인하세요.")
        else:
            st.warning(f"HTTP 상태코드: {res.status_code}")
        return []

    try:
        data = res.json()
        return data.get("items", []) or []
    except Exception:
        st.error("API 응답 JSON 파싱 실패")
        return []


def fetch_news_items(final_keyword: str, total_display: int, client_id: str, client_secret: str) -> list[dict]:
    """
    total_display(100~500)를 100단위로 나누어 여러 번 요청 후 items를 합칩니다.
    - 일부 페이지 실패해도 다른 페이지는 계속 진행하도록 설계
    """
    items: list[dict] = []
    page_count = max(1, total_display // 100)

    for i in range(page_count):
        start = 100 * i + 1
        page_items = naver_news_api_request(final_keyword, display=100, start=start,
                                            client_id=client_id, client_secret=client_secret)
        if page_items:
            items.extend(page_items)

    return items


def build_items_dataframe(items: list[dict]) -> pd.DataFrame:
    """
    items에서 title/pubDate/link만 추출하여 DataFrame 구성.
    """
    rows = []
    for it in items:
        rows.append({
            "title": clean_title(it.get("title", "")),
            "pubDate": format_pubdate(it.get("pubDate", "")),
            "link": it.get("link", ""),
        })
    return pd.DataFrame(rows)


# ===================================================
# 4) 크롤링(실패 시 skip)
# ===================================================
def crawl_naver_news_body(url: str) -> str:
    """
    네이버 뉴스 본문(#dic_area)을 크롤링합니다.
    - 실패하면 "" 반환(=skip)
    """
    try:
        res = rq.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if res.status_code != 200:
            return ""
        soup = bs4.BeautifulSoup(res.text, "html.parser")
        tag = soup.select_one("#dic_area")
        return tag.get_text(separator=" ", strip=True) if tag else ""
    except rq.exceptions.RequestException:
        return ""
    except Exception:
        return ""


def collect_corpus_from_items(items: list[dict]) -> list[str]:
    """
    items 중 네이버 뉴스 링크만 대상으로 본문을 수집합니다.
    - 크롤링 실패는 skip
    - 너무 짧은 본문도 skip
    """
    docs = []
    for it in items:
        link = it.get("link", "")
        if "n.news.naver" not in link:
            continue

        body = crawl_naver_news_body(link)
        if not body:
            continue

        cleaned = clean_text_keep_korean(body)
        if len(cleaned) < 100:
            continue

        docs.append(cleaned)

    return docs


# ===================================================
# 5) 분석(soynlp 명사 필터 + TF-IDF)
# ===================================================
@st.cache_data(show_spinner=False)
def build_noun_set(docs_clean: list[str]) -> set[str]:
    """
    soynlp로 명사 후보를 학습/추출하여 set으로 반환.
    - 데이터가 적으면 빈 set을 반환(=명사 필터 약화)
    """
    sents = []
    for d in docs_clean:
        sents.extend([s.strip() for s in re.split(r"[\.!?]\s*|\n", d) if len(s.strip()) >= 10])

    if len(sents) < 5:
        return set()

    extractor = LRNounExtractor_v2(verbose=False)
    extractor.train(sents)
    nouns = extractor.extract()

    MIN_FREQ = 2
    MIN_SCORE = 0.4

    noun_set = set()
    for w, score in nouns.items():
        freq = getattr(score, "frequency", None)
        sc = getattr(score, "score", None)

        # 버전/구조 방어
        if freq is None and isinstance(score, dict):
            freq = score.get("frequency")
        if sc is None and isinstance(score, dict):
            sc = score.get("score")

        if freq is not None and freq < MIN_FREQ:
            continue
        if sc is not None and sc < MIN_SCORE:
            continue
        noun_set.add(w)

    return noun_set


def tokenize_and_filter_docs(docs_clean: list[str], stopwords: set[str]) -> list[list[str]]:
    """
    - 토크나이저로 토큰화
    - soynlp noun_set 기반으로 명사만 남김
    - 불용어 제거
    """
    tokenizer = load_tokenizer()
    if tokenizer is None:
        # 토크나이저 로드 실패면 앱이 죽지 않도록 "공백 split"으로 fallback
        st.warning("토크나이저 로드 실패로 인해 간단한 split 토큰화를 사용합니다.")
        noun_set = set()
        return [
            [t for t in d.split() if t not in stopwords and len(t) >= 2]
            for d in docs_clean
        ]

    noun_set = build_noun_set(docs_clean)
    if not noun_set:
        st.warning("명사 사전이 약합니다(말뭉치 부족). 명사 필터가 완화됩니다.")

    docs_tokens = []
    for d in docs_clean:
        # flatten=False → (left_token, right_token) 튜플 리스트
        try:
            toks = [t1 for t1, _ in tokenizer.tokenize(d, flatten=False)]
        except Exception:
            toks = d.split()

        filtered = []
        for t in toks:
            nt = normalize_token(t)
            if not nt:
                continue
            if not (2 <= len(nt) <= 8):
                continue
            if nt in stopwords:
                continue
            if noun_set and (nt not in noun_set):
                continue
            filtered.append(nt)

        docs_tokens.append(filtered)

    return docs_tokens


def compute_tfidf_scores(docs_tokens: list[list[str]], top_k: int = 80) -> dict[str, float]:
    """
    TF-IDF로 키워드 점수를 계산합니다.
    - 데이터가 부족하면 빈 dict 반환
    """
    docs_str = [" ".join(ts) for ts in docs_tokens if ts]
    if len(docs_str) < 2:
        return {}

    try:
        vec = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            min_df=2,
        )
        X = vec.fit_transform(docs_str)

        terms = np.array(vec.get_feature_names_out())
        scores = np.asarray(X.sum(axis=0)).ravel()
        if scores.size == 0:
            return {}

        idx = np.argsort(scores)[::-1][:top_k]
        return {terms[i]: float(scores[i]) for i in idx}
    except ValueError:
        # min_df=2 조건 등으로 단어가 하나도 안 남는 경우
        return {}
    except Exception as e:
        st.error(f"TF-IDF 계산 중 오류: {e}")
        return {}


def build_keyword_tables(score_dict: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    score_dict를 DataFrame으로 변환하고 Top50/Top20을 함께 반환합니다.
    """
    df_kw = (
        pd.DataFrame(list(score_dict.items()), columns=["keyword", "score"])
        .sort_values("score", ascending=False)
    )
    return df_kw, df_kw.head(50).copy(), df_kw.head(20).copy()


# ===================================================
# 6) 시각화(figure 반환)
# ===================================================
def make_wordcloud_figure(freq: dict[str, float], mask_name: str):
    """
    워드클라우드 figure를 생성해 반환합니다.
    - freq가 비어있으면 None
    """
    if not freq:
        return None

    bg_path = MASK_BG.get(mask_name, MASK_BG["없음"])
    try:
        img = Image.open(bg_path)
        mask = np.array(img)
    except Exception:
        mask = None

    wc = WordCloud(
        font_path=FONT_PATH,
        background_color="white",
        max_words=80,
        mask=mask,
    ).generate_from_frequencies(freq)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig


def make_top20_bar_figure(df_top20: pd.DataFrame):
    """
    Top20 막대차트 figure를 생성해 반환합니다.
    """
    if df_top20.empty:
        return None

    fig = plt.figure(figsize=(10, 5))
    plt.bar(df_top20["keyword"], df_top20["score"])
    plt.xticks(rotation=45, ha="right")
    plt.title("TF-IDF 상위 키워드 (Top 20)")
    plt.tight_layout()
    return fig


# ===================================================
# 7) 다운로드(이미지 2개를 ZIP으로 제공)
# ===================================================
def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_images_zip_bytes(wc_fig, top20_fig, base_name: str) -> bytes:
    """
    버튼 1개로 워드클라우드 + Top20 차트 이미지를 함께 내려받기 위해 ZIP으로 묶습니다.
    """
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}_wordcloud.png", fig_to_png_bytes(wc_fig))
        zf.writestr(f"{base_name}_top20.png", fig_to_png_bytes(top20_fig))
    zip_buf.seek(0)
    return zip_buf.getvalue()


# ===================================================
# 8) UI 렌더링 함수들
# ===================================================
def render_header_with_lottie():
    """타이틀 옆 Lottie + 타이틀 출력"""
    col1, col2 = st.columns([1, 2])
    with col1:
        lottie = load_json(LOTTIE_PATH)
        if lottie:
            st_lottie(lottie, speed=1, loop=True, width=200, height=200)
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.title("뉴스 키워드 시각화")


def render_sidebar_api_settings():
    """사이드바 API 설정 입력 UI"""
    st.session_state.setdefault("client_id", "")
    st.session_state.setdefault("client_secret", "")

    with st.sidebar.form("client_settings", clear_on_submit=False):
        st.header("API 설정")
        cid = st.text_input("Client ID:", value=st.session_state["client_id"])
        secret = st.text_input("Client Secret:", type="password", value=st.session_state["client_secret"])
        if st.form_submit_button("OK"):
            st.session_state["client_id"] = (cid or "").strip()
            st.session_state["client_secret"] = (secret or "").strip()
            st.rerun()


def render_sidebar_stopwords() -> set[str]:
    """
    사이드바 불용어 입력 UI
    - 파일 불용어 + 사용자 추가 불용어 합쳐서 반환
    """
    st.sidebar.header("불용어(Stopwords)")
    base_stop = load_stopwords_file(STOPWORDS_PATH)
    extra_stop = st.sidebar.text_area("추가 불용어(줄바꿈으로 입력)", value="", height=120)
    extra_stop_set = {w.strip() for w in extra_stop.splitlines() if w.strip()}
    stopwords = base_stop | extra_stop_set
    st.sidebar.caption(f"현재 불용어 수: {len(stopwords)} (파일 + 추가 입력)")
    return stopwords


def render_main_form():
    """
    메인 입력 폼 UI
    - 체크박스 배치 요구사항 반영
    """
    with st.form("search", clear_on_submit=False):
        category = st.selectbox("분야:", ["경제", "정치", "사회", "국제", "연예", "IT", "문화"])
        user_keyword = st.text_input("검색 키워드(필수):", value="", placeholder="예: 금리, 반도체, AI, 메타버스 ...")
        display = st.select_slider("분량(기사 수):", options=[100, 200, 300, 400, 500], value=100)
        mask = st.radio("백마스크:", ["없음", "타원", "말풍선", "하트"], horizontal=True)

        # 1줄: 기사 목록 보기, 링크 제공, 기사 목록 다운로드(.csv)
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
        with r1c1:
            show_articles = st.checkbox("기사 목록 보기", value=True)
        with r1c2:
            show_links = st.checkbox("링크 제공", value=False)
        with r1c3:
            dl_articles = st.checkbox("기사 목록 다운로드(.csv)", value=False)

        # 2줄: 키워드 표 보기, 키워드 표 다운로드(.csv), 이미지 다운로드(.png)
        r2c1, r2c2, r2c3 = st.columns([1, 1, 1])
        with r2c1:
            show_keywords = st.checkbox("키워드 표 보기", value=True)
        with r2c2:
            dl_keywords = st.checkbox("키워드 표 다운로드(.csv)", value=False)
        with r2c3:
            dl_images = st.checkbox("이미지 다운로드(.png)", value=False)

        submitted = st.form_submit_button("OK")

    return {
        "category": category,
        "user_keyword": user_keyword,
        "display": display,
        "mask": mask,
        "show_articles": show_articles,
        "show_links": show_links,
        "dl_articles": dl_articles,
        "show_keywords": show_keywords,
        "dl_keywords": dl_keywords,
        "dl_images": dl_images,
        "submitted": submitted,
    }


# ===================================================
# 9) 메인 실행 로직(앱의 흐름)
# ===================================================
def run_app():
    # 1) UI 기본
    st.set_page_config(page_title="뉴스 키워드 시각화", layout="wide")
    setup_matplotlib_korean_font()
    render_header_with_lottie()

    # 2) Sidebar
    render_sidebar_api_settings()
    stopwords = render_sidebar_stopwords()

    # 3) Form
    form = render_main_form()
    if not form["submitted"]:
        return

    # 4) 입력 검증(데이터 부족 안내 강화)
    if not form["user_keyword"].strip():
        st.warning("검색 키워드를 입력해 주세요. (예: 금리, 반도체, AI)")
        return

    client_id = st.session_state.get("client_id", "").strip()
    client_secret = st.session_state.get("client_secret", "").strip()
    if not client_id or not client_secret:
        st.error("API 인증 정보(Client ID/Secret)가 설정되지 않았습니다.")
        st.info("사이드바에서 Client ID/Secret을 입력 후 다시 시도하세요.")
        return

    final_keyword = build_final_keyword(form["category"], form["user_keyword"])

    # 5) 뉴스 목록 수집 (API 오류/인증 실패 방어)
    st.info(f"뉴스 목록 수집 중... (검색어: {final_keyword})")
    items = fetch_news_items(final_keyword, form["display"], client_id, client_secret)

    if not items:
        st.warning("뉴스 목록을 가져오지 못했습니다.")
        st.info("가능한 원인: (1) 인증 실패 (2) 네트워크 오류 (3) 검색 결과 없음")
        return

    df_items = build_items_dataframe(items)

    # 6) 기사 목록 표시/링크
    if form["show_articles"]:
        st.subheader("수집된 기사 목록")
        if df_items.empty:
            st.warning("기사 목록이 비어 있습니다.")
        else:
            st.dataframe(df_items[["title", "pubDate"]], use_container_width=True)

            if form["show_links"]:
                st.caption("기사 링크(클릭):")
                # 너무 많으면 부담이 될 수 있어 상위 30개만
                for _, r in df_items.head(30).iterrows():
                    if r["link"]:
                        st.markdown(f"- [🔗 바로가기]({r['link']}) — {r['title']}")

    # 7) 본문 크롤링 (실패 시 skip)
    st.info("뉴스 본문 크롤링 중...")
    docs_clean = collect_corpus_from_items(items)

    if len(docs_clean) < 5:
        st.warning("본문 데이터가 부족하여 분석이 어렵습니다.")
        st.info(
            "개선 팁:\n"
            "- 분량을 200~500으로 늘려보세요.\n"
            "- 키워드를 더 넓게/일반적으로 바꿔보세요.\n"
            "- 링크 제공을 켜서 실제로 네이버 뉴스 링크가 많은지 확인해보세요."
        )
        return

    # 8) 토큰화/필터 + TF-IDF
    st.info("키워드 분석 중(명사 필터 + TF-IDF)...")
    docs_tokens = tokenize_and_filter_docs(docs_clean, stopwords)

    score_dict = compute_tfidf_scores(docs_tokens, top_k=80)
    if not score_dict:
        st.warning("키워드 점수를 계산할 수 없습니다(데이터/필터 조건 부족).")
        st.info(
            "개선 팁:\n"
            "- 불용어가 너무 많으면 단어가 거의 남지 않을 수 있습니다.\n"
            "- 분량을 늘리거나 키워드를 바꿔 다시 시도하세요."
        )
        return

    df_kw, df_kw_top50, df_kw_top20 = build_keyword_tables(score_dict)

    # 9) 키워드 표 표시
    if form["show_keywords"]:
        st.subheader("키워드(TF-IDF) 상위 50")
        st.dataframe(df_kw_top50, use_container_width=True)

    # 10) 시각화
    st.info("워드클라우드 생성 중...")
    wc_fig = make_wordcloud_figure(score_dict, form["mask"])
    if wc_fig is None:
        st.warning("워드클라우드 생성 실패(데이터 부족).")
        return
    st.pyplot(wc_fig)

    st.info("Top20 막대차트 생성 중...")
    top20_fig = make_top20_bar_figure(df_kw_top20)
    if top20_fig is None:
        st.warning("Top20 차트 생성 실패(데이터 부족).")
        return
    st.pyplot(top20_fig)

    # 11) 다운로드(차트 아래, 버튼 한 줄 배치)
    st.markdown("---")
    st.subheader("결과 다운로드")

    can_articles = not df_items.empty
    can_keywords = not df_kw_top50.empty
    can_images = (wc_fig is not None) and (top20_fig is not None)

    base = safe_filename(final_keyword)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            label="기사 목록 다운로드(.csv)",
            data=df_items.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig") if can_articles else b"",
            file_name=f"articles_{base}_{ts}.csv",
            mime="text/csv",
            disabled=not (form["dl_articles"] and can_articles),
        )

    with c2:
        st.download_button(
            label="키워드 표 다운로드(.csv)",
            data=df_kw_top50.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig") if can_keywords else b"",
            file_name=f"keywords_{base}_{ts}.csv",
            mime="text/csv",
            disabled=not (form["dl_keywords"] and can_keywords),
        )

    with c3:
        # 버튼 문구는 .png로 보이지만 2장 동시 다운로드를 위해 ZIP 제공(안정적)
        zip_bytes = make_images_zip_bytes(wc_fig, top20_fig, f"{base}_{ts}") if can_images else b""
        st.download_button(
            label="이미지 다운로드(.png)",
            data=zip_bytes,
            file_name=f"images_{base}_{ts}.zip",
            mime="application/zip",
            disabled=not (form["dl_images"] and can_images),
        )


# ===================================================
# 엔트리포인트
# ===================================================
if __name__ == "__main__":
    run_app()

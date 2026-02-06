#
# ============================================================
#  News Keyword Visualizer V4 (Diagnostics Enhanced)
# ------------------------------------------------------------
#  ✅ V4 기능 유지 + 배포 환경 진단 UI 추가
#
#  [추가된 진단 기능]
#  - API 단계(페이지 단위) 성공/실패, 상태코드, timeout, 평균 응답시간 표시
#  - 크롤링 단계 성공/실패/스킵(짧음/네이버 링크 아님) 카운트 표시
#
#  실행:
#     streamlit run app_v4.py
# ============================================================
#

# ============================================================
# 라이브러리 호출
# ============================================================
import json
import re
import pickle
import html
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from urllib.parse import quote
import zipfile

import requests as rq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


# ============================================================
# 0) 전역 설정(경로/파일)
# ============================================================
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

# API/크롤링 timeout 기본값(배포 환경에서 read_timeout이 더 중요)
API_TIMEOUT = (5, 25)      # (connect_timeout, read_timeout)
CRAWL_TIMEOUT = (5, 15)    # 크롤링은 너무 길게 끌지 않게


# ============================================================
# 1) 테마 친화 CSS (라이트/다크 공용)
# ============================================================
def inject_theme_friendly_css() -> None:
    """
    라이트/다크 모드 모두 자연스럽게 보이도록
    - 카드/배지/하이라이트 스타일을 테마 친화적으로 적용
    """
    st.markdown(
        """
        <style>
        .nk-title-wrap{
            text-align:center;
            margin: 0.25rem 0 1.0rem 0;
        }
        .nk-title-main{
            font-size: 2.0rem;
            font-weight: 850;
            line-height: 1.1;
            margin: 0;
        }
        .nk-title-sub{
            font-size: 1.1rem;
            font-weight: 700;
            opacity: 0.8;
            margin-top: 0.35rem;
        }

        .nk-card{
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            padding: 12px 14px;
            margin: 10px 0;
            background: rgba(128,128,128,0.06);
        }
        @media (prefers-color-scheme: dark) {
          .nk-card{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.18);
          }
        }

        .nk-badge{
            display:inline-block;
            padding:6px 12px;
            margin:4px 6px 4px 0;
            border-radius: 16px;
            font-size: 0.92rem;
            font-weight: 750;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(99,102,241,0.10);
        }
        @media (prefers-color-scheme: dark) {
          .nk-badge{
            background: rgba(99,102,241,0.18);
            border: 1px solid rgba(255,255,255,0.18);
          }
        }

        mark{
            padding: 0.08em 0.18em;
            border-radius: 0.25em;
            background: rgba(245, 158, 11, 0.35);
            color: inherit;
        }
        @media (prefers-color-scheme: dark) {
          mark{ background: rgba(245, 158, 11, 0.28); }
        }

        .nk-link{
            opacity: 0.92;
            font-weight: 650;
        }

        /* 진단 배지 */
        .nk-pill{
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 750;
            border: 1px solid rgba(128,128,128,0.25);
            background: rgba(16,185,129,0.12); /* emerald */
        }
        @media (prefers-color-scheme: dark) {
          .nk-pill{
            background: rgba(16,185,129,0.18);
            border: 1px solid rgba(255,255,255,0.18);
          }
        }

        .nk-pill-warn{
            background: rgba(245,158,11,0.12);
        }
        @media (prefers-color-scheme: dark) {
          .nk-pill-warn{ background: rgba(245,158,11,0.18); }
        }

        .nk-pill-bad{
            background: rgba(239,68,68,0.12);
        }
        @media (prefers-color-scheme: dark) {
          .nk-pill-bad{ background: rgba(239,68,68,0.18); }
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 2) matplotlib 한글 폰트 설정
# ============================================================
def setup_matplotlib_korean_font() -> None:
    """matplotlib 한글 깨짐 방지 설정."""
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    except Exception:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3) requests 세션 + 재시도(배포 환경 안정화)
# ============================================================
@st.cache_resource
def get_http_session() -> rq.Session:
    """
    배포 환경에서 timeout/일시 장애가 종종 발생합니다.
    - Session 재사용(연결/TLS 재사용)
    - Retry + Backoff(429/5xx/timeout 가치 있는 오류 자동 복구)
    """
    session = rq.Session()

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NKVisualizer/1.0)"})
    return session


# ============================================================
# 4) 리소스 로딩(캐시)
# ============================================================
def load_json(path: str) -> dict:
    """JSON 파일 안전 로드(실패 시 빈 dict)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_stopwords_file(path: str) -> set[str]:
    """불용어 파일 -> set (실패 시 빈 set)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}
    except Exception:
        return set()


@st.cache_resource
def load_tokenizer():
    """토크나이저 로드(실패 시 None)."""
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ============================================================
# 5) 텍스트 유틸
# ============================================================
def clean_title(raw_title: str) -> str:
    """네이버 뉴스 title의 HTML 제거 + 공백 정리."""
    t = html.unescape(raw_title or "")
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def format_pubdate(pub_date: str) -> str:
    """pubDate를 사람이 읽기 쉬운 문자열로 변환."""
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pub_date or ""


@st.cache_data(show_spinner=False)
def clean_text_keep_korean(text: str) -> str:
    """숫자/영문/특수문자 제거 + 공백 정리(한글 중심)."""
    text = re.sub(r"\d|[a-zA-Z]|\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_token(t: str) -> str:
    """토큰 정규화(구두점/공백 제거)."""
    if t is None:
        return ""
    t = str(t).strip()
    t = re.sub(r"[\"'“”‘’\(\)\[\]\{\},\.\!\?\:\;]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def build_final_keyword(category: str, user_keyword: str) -> str:
    """분야 + 사용자 키워드 결합."""
    category = (category or "").strip()
    user_keyword = re.sub(r"\s+", " ", (user_keyword or "")).strip()
    return f"{category} {user_keyword}".strip()


def safe_filename(s: str) -> str:
    """파일명 안전화."""
    s = s.strip()
    s = re.sub(r"[^\w\-가-힣]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "result"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV 다운로드용 bytes(utf-8-sig)."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def highlight_keyword(text: str, keyword: str) -> str:
    """제목 내 키워드 하이라이트(<mark>)."""
    if not keyword:
        return text
    try:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    except Exception:
        return text


# ============================================================
# 6) 진단용 메트릭(초기화/저장)
# ============================================================
def init_api_metrics(total_display: int) -> dict:
    """API 단계 진단 메트릭 초기화."""
    page_cnt = max(1, total_display // 100)
    return {
        "pages_planned": page_cnt,
        "pages_attempted": 0,
        "pages_ok": 0,
        "pages_fail": 0,
        "items_total": 0,
        "http_status_counts": {},     # 예: {200: 3, 429: 1}
        "error_counts": {},           # 예: {"ReadTimeout":2, "AuthFail":1}
        "latencies": [],              # 응답시간(초) 리스트
        "last_error": "",
    }


def init_crawl_metrics() -> dict:
    """크롤링 단계 진단 메트릭 초기화."""
    return {
        "candidate_links": 0,         # n.news.naver 링크 수
        "attempted": 0,
        "success": 0,
        "fail": 0,
        "skip_short": 0,              # 본문이 너무 짧아 skip
        "skip_non_naver": 0,          # 네이버 뉴스 링크가 아니라 skip
        "last_error": "",
    }


def inc_dict(d: dict, k, inc: int = 1):
    d[k] = d.get(k, 0) + inc


def save_results_to_session(
    final_keyword: str,
    df_items: pd.DataFrame,
    df_kw_top50: pd.DataFrame,
    df_kw_top20: pd.DataFrame,
    wc_png: bytes,
    top20_png: bytes,
    zip_bytes: bytes,
    api_metrics: dict,
    crawl_metrics: dict,
):
    st.session_state["result_ready"] = True
    st.session_state["final_keyword"] = final_keyword
    st.session_state["df_items"] = df_items
    st.session_state["df_kw_top50"] = df_kw_top50
    st.session_state["df_kw_top20"] = df_kw_top20
    st.session_state["wc_png"] = wc_png
    st.session_state["top20_png"] = top20_png
    st.session_state["images_zip"] = zip_bytes
    st.session_state["api_metrics"] = api_metrics
    st.session_state["crawl_metrics"] = crawl_metrics


def clear_results_session():
    st.session_state["result_ready"] = False
    for k in [
        "final_keyword", "df_items", "df_kw_top50", "df_kw_top20",
        "wc_png", "top20_png", "images_zip", "api_metrics", "crawl_metrics"
    ]:
        if k in st.session_state:
            del st.session_state[k]


# ============================================================
# 7) 네이버 API(방어 + 진단)
# ============================================================
def naver_news_api_request(
    keyword: str,
    display: int,
    start: int,
    client_id: str,
    client_secret: str,
    api_metrics: dict,
) -> list[dict]:
    """
    네이버 뉴스 검색 API(1페이지).
    ✅ 진단 포함:
    - 페이지 시도/성공/실패
    - 상태코드 카운트
    - timeout/exception 카운트
    - 응답시간 기록
    """
    api_metrics["pages_attempted"] += 1

    if not client_id.strip() or not client_secret.strip():
        inc_dict(api_metrics["error_counts"], "MissingKey")
        api_metrics["last_error"] = "MissingKey"
        st.error("API 인증 정보(Client ID/Secret)가 비어 있습니다.")
        return []

    url = (
        "https://openapi.naver.com/v1/search/news.json"
        f"?query={quote(keyword)}&display={display}&start={start}"
    )
    headers = {
        "X-Naver-Client-Id": client_id.strip(),
        "X-Naver-Client-Secret": client_secret.strip(),
    }

    session = get_http_session()

    try:
        t0 = time.perf_counter()
        res = session.get(url, headers=headers, timeout=API_TIMEOUT)
        elapsed = time.perf_counter() - t0
        api_metrics["latencies"].append(elapsed)

    except rq.exceptions.ConnectTimeout:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "ConnectTimeout")
        api_metrics["last_error"] = "ConnectTimeout"
        st.error("네트워크 오류: 서버 연결 시간이 초과되었습니다(ConnectTimeout).")
        return []
    except rq.exceptions.ReadTimeout:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "ReadTimeout")
        api_metrics["last_error"] = "ReadTimeout"
        st.error("네트워크 오류: 응답 대기 시간이 초과되었습니다(ReadTimeout).")
        return []
    except rq.exceptions.Timeout:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "Timeout")
        api_metrics["last_error"] = "Timeout"
        st.error("네트워크 오류: 요청 시간이 초과되었습니다(timeout).")
        return []
    except rq.exceptions.ConnectionError:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "ConnectionError")
        api_metrics["last_error"] = "ConnectionError"
        st.error("네트워크 오류: 서버에 연결할 수 없습니다(ConnectionError).")
        return []
    except rq.exceptions.RequestException as e:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "RequestException")
        api_metrics["last_error"] = f"RequestException: {e}"
        st.error(f"네트워크 오류: {e}")
        return []

    inc_dict(api_metrics["http_status_counts"], res.status_code)

    # 요구사항: 200 아니면 "API 요청 실패"
    if res.status_code != 200:
        api_metrics["pages_fail"] += 1
        st.error("API 요청 실패")

        if res.status_code in (401, 403):
            inc_dict(api_metrics["error_counts"], "AuthFail")
            api_metrics["last_error"] = f"AuthFail({res.status_code})"
            st.warning("API 인증 실패(권한/키 오류). Client ID/Secret을 확인하세요.")
        elif res.status_code == 429:
            inc_dict(api_metrics["error_counts"], "RateLimit(429)")
            api_metrics["last_error"] = "RateLimit(429)"
            st.warning("요청이 너무 많습니다(429). 잠시 후 다시 시도하세요.")
        else:
            inc_dict(api_metrics["error_counts"], f"HTTP({res.status_code})")
            api_metrics["last_error"] = f"HTTP({res.status_code})"
            hint = (res.text or "")[:200].strip()
            if hint:
                st.caption(f"응답 일부: {hint}")
            st.warning(f"HTTP 상태코드: {res.status_code}")

        return []

    try:
        data = res.json()
        items = data.get("items", []) or []
        api_metrics["pages_ok"] += 1
        api_metrics["items_total"] += len(items)
        return items
    except Exception:
        api_metrics["pages_fail"] += 1
        inc_dict(api_metrics["error_counts"], "JSONParseFail")
        api_metrics["last_error"] = "JSONParseFail"
        st.error("API 응답 JSON 파싱 실패")
        return []


def fetch_news_items(final_keyword: str, total_display: int, client_id: str, client_secret: str, api_metrics: dict) -> list[dict]:
    """
    100단위로 페이지 요청 후 items 합치기.
    ✅ 배포 안정화:
    - 일부 페이지 실패해도 계속
    - 연속 실패 누적되면 조기 중단(앱이 오래 멈춘 듯 보이는 문제 방지)
    """
    items: list[dict] = []
    page_count = api_metrics["pages_planned"]

    consecutive_fail = 0
    MAX_CONSEC_FAIL = 2

    for i in range(page_count):
        start = 100 * i + 1
        page_items = naver_news_api_request(
            final_keyword, 100, start,
            client_id, client_secret,
            api_metrics
        )

        if page_items:
            items.extend(page_items)
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            if consecutive_fail >= MAX_CONSEC_FAIL:
                st.warning("API 요청이 연속으로 실패하여 추가 페이지 수집을 중단했습니다.")
                break

    return items


def build_items_dataframe(items: list[dict]) -> pd.DataFrame:
    """items에서 title/pubDate/link만 추출."""
    rows = []
    for it in items:
        rows.append({
            "title": clean_title(it.get("title", "")),
            "pubDate": format_pubdate(it.get("pubDate", "")),
            "link": it.get("link", ""),
        })
    return pd.DataFrame(rows)


# ============================================================
# 8) 크롤링(실패 skip + 진단)
# ============================================================
def crawl_naver_news_body(url: str, crawl_metrics: dict) -> str:
    """
    네이버 뉴스 본문 크롤링(#dic_area).
    ✅ 진단 포함:
    - attempted/success/fail 카운트
    - 예외 발생해도 앱이 죽지 않게 처리
    """
    crawl_metrics["attempted"] += 1
    session = get_http_session()

    try:
        res = session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=CRAWL_TIMEOUT)
        if res.status_code != 200:
            crawl_metrics["fail"] += 1
            crawl_metrics["last_error"] = f"HTTP({res.status_code})"
            return ""

        soup = bs4.BeautifulSoup(res.text, "html.parser")
        tag = soup.select_one("#dic_area")
        if not tag:
            crawl_metrics["fail"] += 1
            crawl_metrics["last_error"] = "NoSelector(#dic_area)"
            return ""

        crawl_metrics["success"] += 1
        return tag.get_text(separator=" ", strip=True)

    except rq.exceptions.Timeout:
        crawl_metrics["fail"] += 1
        crawl_metrics["last_error"] = "Timeout"
        return ""
    except rq.exceptions.RequestException as e:
        crawl_metrics["fail"] += 1
        crawl_metrics["last_error"] = f"RequestException: {e}"
        return ""
    except Exception as e:
        crawl_metrics["fail"] += 1
        crawl_metrics["last_error"] = f"Exception: {e}"
        return ""


def collect_corpus_from_items(items: list[dict], crawl_metrics: dict) -> list[str]:
    """
    네이버 뉴스 링크만 본문 수집.
    - 실패/짧은 본문 skip
    - 진단용 카운트 집계
    """
    docs = []
    for it in items:
        link = it.get("link", "")

        if "n.news.naver" not in link:
            crawl_metrics["skip_non_naver"] += 1
            continue

        crawl_metrics["candidate_links"] += 1

        body = crawl_naver_news_body(link, crawl_metrics)
        if not body:
            continue

        cleaned = clean_text_keep_korean(body)
        if len(cleaned) < 100:
            crawl_metrics["skip_short"] += 1
            continue

        docs.append(cleaned)

    return docs


# ============================================================
# 9) 분석(soynlp 명사 set + TF-IDF)
# ============================================================
@st.cache_data(show_spinner=False)
def build_noun_set(docs_clean: list[str]) -> set[str]:
    """soynlp로 명사 후보 set 생성(데이터 적으면 빈 set)."""
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
    토큰화 -> 명사 필터(가능하면) -> 불용어 제거
    - 토크나이저 로드 실패 시 split fallback
    - 명사 set이 비면(데이터 부족) 명사 필터 완화
    """
    tokenizer = load_tokenizer()
    noun_set = build_noun_set(docs_clean)

    docs_tokens: list[list[str]] = []

    for d in docs_clean:
        if tokenizer is None:
            raw_tokens = d.split()
        else:
            try:
                raw_tokens = [t1 for t1, _ in tokenizer.tokenize(d, flatten=False)]
            except Exception:
                raw_tokens = d.split()

        filtered = []
        for t in raw_tokens:
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
    """TF-IDF 점수 계산(오류/부족 시 빈 dict)."""
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
    except Exception:
        return {}


def build_keyword_tables(score_dict: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """점수 dict -> DataFrame + Top50/Top20."""
    df_kw = (
        pd.DataFrame(list(score_dict.items()), columns=["keyword", "score"])
        .sort_values("score", ascending=False)
    )
    return df_kw, df_kw.head(50).copy(), df_kw.head(20).copy()


# ============================================================
# 10) 시각화(PNG bytes)
# ============================================================
def fig_to_png_bytes(fig) -> bytes:
    """matplotlib figure -> PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_wordcloud_png(freq: dict[str, float], mask_name: str) -> bytes | None:
    """워드클라우드 PNG bytes 생성."""
    if not freq:
        return None

    bg_path = MASK_BG.get(mask_name, MASK_BG["없음"])
    mask = None
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

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def make_top20_bar_png(df_top20: pd.DataFrame) -> bytes | None:
    """Top20 막대차트 PNG bytes 생성."""
    if df_top20.empty:
        return None

    fig = plt.figure(figsize=(10, 5))
    plt.bar(df_top20["keyword"], df_top20["score"])
    plt.xticks(rotation=45, ha="right")
    plt.title("TF-IDF 상위 키워드 (Top 20)")
    plt.tight_layout()

    png = fig_to_png_bytes(fig)
    plt.close(fig)
    return png


def make_images_zip_bytes(wordcloud_png: bytes, top20_png: bytes, base_name: str) -> bytes:
    """워드클라우드 + Top20 이미지를 ZIP으로 묶어서 반환."""
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}_wordcloud.png", wordcloud_png)
        zf.writestr(f"{base_name}_top20.png", top20_png)
    zip_buf.seek(0)
    return zip_buf.getvalue()


# ============================================================
# 11) UI 렌더링
# ============================================================
def render_header_with_lottie_and_center_title():
    """Lottie 좌측 + 2줄 가운데 타이틀"""
    col1, col2 = st.columns([1, 2.2])

    with col1:
        lottie = load_json(LOTTIE_PATH)
        if lottie:
            st_lottie(lottie, speed=1, loop=True, width=200, height=200)

    with col2:
        st.markdown(
            """
            <div class="nk-title-wrap">
                <div class="nk-title-main">뉴스 키워드 어플리케이션</div>
                <div class="nk-title-sub">(분석 &amp; 시각화)</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_sidebar_api_settings():
    """사이드바 API 설정 폼."""
    st.sidebar.header("API Keys :")
    st.session_state.setdefault("client_id", "")
    st.session_state.setdefault("client_secret", "")

    with st.sidebar.form("client_settings", clear_on_submit=False):
        cid = st.text_input("Client ID:", value=st.session_state["client_id"])
        secret = st.text_input("Client Secret:", type="password", value=st.session_state["client_secret"])
        if st.form_submit_button("OK"):
            st.session_state["client_id"] = (cid or "").strip()
            st.session_state["client_secret"] = (secret or "").strip()
            st.rerun()


def render_sidebar_options():
    """
    옵션 체크박스(요구사항):
    - 1줄: 기사 목록 보기, 링크 제공, 기사 목록 다운로드(.csv)
    - 2줄: 키워드 표 보기, 키워드 표 다운로드(.csv), 이미지 다운로드(.png)
    """
    st.sidebar.header("표시/다운로드 옵션 :")
    r1c1, r1c2, r1c3 = st.sidebar.columns(3)
    with r1c1:
        show_articles = st.checkbox("기사 목록 보기", value=True, key="opt_show_articles")
    with r1c2:
        show_links = st.checkbox("링크 제공", value=False, key="opt_show_links")
    with r1c3:
        dl_articles = st.checkbox("기사 목록 다운로드(.csv)", value=False, key="opt_dl_articles")

    r2c1, r2c2, r2c3 = st.sidebar.columns(3)
    with r2c1:
        show_keywords = st.checkbox("키워드 표 보기", value=True, key="opt_show_keywords")
    with r2c2:
        dl_keywords = st.checkbox("키워드 표 다운로드(.csv)", value=False, key="opt_dl_keywords")
    with r2c3:
        dl_images = st.checkbox("이미지 다운로드(.png)", value=False, key="opt_dl_images")

    return {
        "show_articles": show_articles,
        "show_links": show_links,
        "dl_articles": dl_articles,
        "show_keywords": show_keywords,
        "dl_keywords": dl_keywords,
        "dl_images": dl_images,
    }


def render_sidebar_stopwords() -> set[str]:
    """불용어 영역(파일 + 추가 입력)."""
    st.sidebar.header("불용어(Stopwords) :")
    base_stop = load_stopwords_file(STOPWORDS_PATH)
    extra_stop = st.sidebar.text_area("추가 불용어(줄바꿈으로 입력)", value="", height=120)
    stopwords = base_stop | {w.strip() for w in extra_stop.splitlines() if w.strip()}
    st.sidebar.caption(f"현재 불용어 수: {len(stopwords)} (파일 + 추가 입력)")
    return stopwords


def render_search_form():
    """검색 폼(UI 카드)."""
    with st.container(border=True):
        st.subheader("검색 조건")

        with st.form("search", clear_on_submit=False):
            c1, c2, c3 = st.columns([1, 2, 1])

            with c1:
                category = st.selectbox("분야", ["경제", "정치", "사회", "국제", "연예", "IT", "문화"])
            with c2:
                user_keyword = st.text_input(
                    "검색 키워드(필수)",
                    value="",
                    placeholder="예: 금리, 반도체, AI, 메타버스 ..."
                )
            with c3:
                display = st.select_slider("분량", options=[100, 200, 300, 400, 500], value=100)

            mask = st.radio("백마스크", ["없음", "타원", "말풍선", "하트"], horizontal=True)
            submit = st.form_submit_button("검색 실행")

    return {
        "category": category,
        "user_keyword": user_keyword,
        "display": display,
        "mask": mask,
        "submitted": submit,
    }


# ============================================================
# 12) 파이프라인 실행(상태박스+진행바)
# ============================================================
def run_pipeline(form: dict, stopwords: set[str], status_box, progress_bar):
    """
    파이프라인:
    1) API 수집(+진단)
    2) 크롤링(+진단)
    3) 분석
    4) 시각화
    """
    if not form["user_keyword"].strip():
        st.warning("검색 키워드를 입력해 주세요. (예: 금리, 반도체, AI)")
        return

    client_id = st.session_state.get("client_id", "").strip()
    client_secret = st.session_state.get("client_secret", "").strip()
    if not client_id or not client_secret:
        st.error("API 인증 정보(Client ID/Secret)가 설정되지 않았습니다. 사이드바에서 입력 후 다시 시도하세요.")
        return

    final_keyword = build_final_keyword(form["category"], form["user_keyword"])

    # ✅ 진단 메트릭 초기화
    api_metrics = init_api_metrics(form["display"])
    crawl_metrics = init_crawl_metrics()

    # 1) API 수집
    status_box.info(f"1/4 뉴스 목록 수집 중... (검색어: {final_keyword})")
    progress_bar.progress(0.2)

    items = fetch_news_items(final_keyword, form["display"], client_id, client_secret, api_metrics)

    # API 단계가 실패하면(=items가 없다면) 진단 정보를 저장해두고 종료
    if not items:
        status_box.error("뉴스 목록을 가져오지 못했습니다.")
        st.info("가능한 원인: (1) 인증 실패 (2) 네트워크 오류 (3) 검색 결과 없음")

        # ✅ 결과 세션에도 진단 정보만 저장(요약 탭에서 확인 가능하게)
        df_items = pd.DataFrame(columns=["title", "pubDate", "link"])
        save_results_to_session(
            final_keyword=final_keyword,
            df_items=df_items,
            df_kw_top50=pd.DataFrame(columns=["keyword", "score"]),
            df_kw_top20=pd.DataFrame(columns=["keyword", "score"]),
            wc_png=b"",
            top20_png=b"",
            zip_bytes=b"",
            api_metrics=api_metrics,
            crawl_metrics=crawl_metrics,
        )
        st.session_state["result_ready"] = True
        return

    df_items = build_items_dataframe(items)

    # 2) 크롤링
    status_box.info("2/4 뉴스 본문 크롤링 중...")
    progress_bar.progress(0.45)

    docs_clean = collect_corpus_from_items(items, crawl_metrics)

    if len(docs_clean) < 5:
        status_box.warning("본문 데이터가 부족하여 분석이 어렵습니다.")
        st.info(
            "개선 팁:\n"
            "- 분량을 300~500으로 늘려보세요.\n"
            "- 키워드를 더 일반적으로 바꿔보세요.\n"
            "- 기사 목록에서 네이버 뉴스 링크가 충분한지 확인해보세요.\n"
            "- (배포 환경) 크롤링 성공률이 낮다면 차단 가능성이 큽니다."
        )

        save_results_to_session(
            final_keyword=final_keyword,
            df_items=df_items,
            df_kw_top50=pd.DataFrame(columns=["keyword", "score"]),
            df_kw_top20=pd.DataFrame(columns=["keyword", "score"]),
            wc_png=b"",
            top20_png=b"",
            zip_bytes=b"",
            api_metrics=api_metrics,
            crawl_metrics=crawl_metrics,
        )
        st.session_state["result_ready"] = True
        return

    # 3) 분석
    status_box.info("3/4 키워드 분석 중(명사 필터 + TF-IDF)...")
    progress_bar.progress(0.7)

    docs_tokens = tokenize_and_filter_docs(docs_clean, stopwords)

    score_dict = compute_tfidf_scores(docs_tokens, top_k=80)
    if not score_dict:
        status_box.warning("키워드 점수를 계산할 수 없습니다(데이터/필터 조건 부족).")
        st.info("개선 팁: 분량을 늘리거나 불용어를 과도하게 추가하지 않았는지 확인하세요.")

        save_results_to_session(
            final_keyword=final_keyword,
            df_items=df_items,
            df_kw_top50=pd.DataFrame(columns=["keyword", "score"]),
            df_kw_top20=pd.DataFrame(columns=["keyword", "score"]),
            wc_png=b"",
            top20_png=b"",
            zip_bytes=b"",
            api_metrics=api_metrics,
            crawl_metrics=crawl_metrics,
        )
        st.session_state["result_ready"] = True
        return

    _, df_kw_top50, df_kw_top20 = build_keyword_tables(score_dict)

    # 4) 시각화 생성
    status_box.info("4/4 시각화 생성 중...")
    progress_bar.progress(0.9)

    wc_png = make_wordcloud_png(score_dict, form["mask"])
    top20_png = make_top20_bar_png(df_kw_top20)

    if not wc_png or not top20_png:
        status_box.error("시각화 생성에 실패했습니다(데이터 부족/렌더링 오류).")

        save_results_to_session(
            final_keyword=final_keyword,
            df_items=df_items,
            df_kw_top50=df_kw_top50,
            df_kw_top20=df_kw_top20,
            wc_png=b"",
            top20_png=b"",
            zip_bytes=b"",
            api_metrics=api_metrics,
            crawl_metrics=crawl_metrics,
        )
        st.session_state["result_ready"] = True
        return

    base = safe_filename(final_keyword)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_bytes = make_images_zip_bytes(wc_png, top20_png, f"{base}_{ts}")

    progress_bar.progress(1.0)
    status_box.success("완료! 결과를 확인하세요.")

    save_results_to_session(
        final_keyword=final_keyword,
        df_items=df_items,
        df_kw_top50=df_kw_top50,
        df_kw_top20=df_kw_top20,
        wc_png=wc_png,
        top20_png=top20_png,
        zip_bytes=zip_bytes,
        api_metrics=api_metrics,
        crawl_metrics=crawl_metrics,
    )


# ============================================================
# 13) 결과 탭 UI
# ============================================================
def render_top5_badges(df_kw_top50: pd.DataFrame) -> None:
    """요약 탭에 Top5 키워드를 배지로 렌더링."""
    top5 = df_kw_top50.head(5)["keyword"].tolist()
    badges = "".join([f'<span class="nk-badge">#{kw}</span>' for kw in top5])

    st.markdown(
        f"""
        <div style="margin:6px 0 14px 0;">
            <div style="font-weight:800; margin-bottom:6px;">핵심 키워드 요약</div>
            <div>{badges}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_diagnostics_panel(api_metrics: dict, crawl_metrics: dict):
    """
    ✅ 배포 진단용 패널
    - API: 페이지 성공률, 상태코드 분포, timeout/에러, 평균 응답시간
    - Crawl: 후보 링크/시도/성공/실패/스킵 카운트
    """
    st.markdown('<div class="nk-card">', unsafe_allow_html=True)
    st.markdown("### 배포 환경 진단(원인 빠른 판별)")

    # ---------- API 요약 ----------
    pages_planned = api_metrics.get("pages_planned", 0)
    pages_attempted = api_metrics.get("pages_attempted", 0)
    pages_ok = api_metrics.get("pages_ok", 0)
    pages_fail = api_metrics.get("pages_fail", 0)
    items_total = api_metrics.get("items_total", 0)

    lat = api_metrics.get("latencies", [])
    avg_latency = float(np.mean(lat)) if lat else 0.0
    p95_latency = float(np.percentile(lat, 95)) if len(lat) >= 3 else (max(lat) if lat else 0.0)

    http_counts = api_metrics.get("http_status_counts", {})
    err_counts = api_metrics.get("error_counts", {})
    last_err = api_metrics.get("last_error", "")

    # 성공률
    api_success_rate = (pages_ok / pages_attempted * 100) if pages_attempted else 0.0

    # 상태 판단(대충 3단계)
    # - ok: 성공률 70% 이상
    # - warn: 30~70%
    # - bad: 30% 미만
    if api_success_rate >= 70:
        api_pill_class = "nk-pill"
        api_level = "양호"
    elif api_success_rate >= 30:
        api_pill_class = "nk-pill nk-pill-warn"
        api_level = "주의"
    else:
        api_pill_class = "nk-pill nk-pill-bad"
        api_level = "위험"

    st.markdown("**1) API 단계(네이버 뉴스 검색 API)**")
    st.markdown(
        f"""
        <span class="{api_pill_class}">성공률 {api_success_rate:.0f}% ({api_level})</span>
        <span class="nk-pill">페이지 OK {pages_ok}</span>
        <span class="nk-pill nk-pill-warn">페이지 FAIL {pages_fail}</span>
        <span class="nk-pill">items {items_total}</span>
        <span class="nk-pill">평균응답 {avg_latency:.2f}s</span>
        <span class="nk-pill">p95 {p95_latency:.2f}s</span>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"계획 페이지: {pages_planned} / 시도 페이지: {pages_attempted} / 마지막 에러: {last_err or '-'}")

    # 상태코드/에러 테이블(간단)
    colA, colB = st.columns(2)
    with colA:
        if http_counts:
            df_http = pd.DataFrame(sorted(http_counts.items()), columns=["status_code", "count"])
            st.dataframe(df_http, use_container_width=True, height=150)
        else:
            st.info("상태코드 기록이 없습니다.")
    with colB:
        if err_counts:
            df_err = pd.DataFrame(sorted(err_counts.items(), key=lambda x: x[1], reverse=True), columns=["error", "count"])
            st.dataframe(df_err, use_container_width=True, height=150)
        else:
            st.info("에러 기록이 없습니다.")

    st.divider()

    # ---------- Crawl 요약 ----------
    st.markdown("**2) 크롤링 단계(네이버 뉴스 본문 수집)**")
    candidate = crawl_metrics.get("candidate_links", 0)
    attempted = crawl_metrics.get("attempted", 0)
    success = crawl_metrics.get("success", 0)
    fail = crawl_metrics.get("fail", 0)
    skip_short = crawl_metrics.get("skip_short", 0)
    skip_non = crawl_metrics.get("skip_non_naver", 0)
    last_crawl_err = crawl_metrics.get("last_error", "")

    crawl_success_rate = (success / attempted * 100) if attempted else 0.0

    if crawl_success_rate >= 70:
        crawl_pill_class = "nk-pill"
        crawl_level = "양호"
    elif crawl_success_rate >= 30:
        crawl_pill_class = "nk-pill nk-pill-warn"
        crawl_level = "주의"
    else:
        crawl_pill_class = "nk-pill nk-pill-bad"
        crawl_level = "위험"

    st.markdown(
        f"""
        <span class="{crawl_pill_class}">성공률 {crawl_success_rate:.0f}% ({crawl_level})</span>
        <span class="nk-pill">후보링크 {candidate}</span>
        <span class="nk-pill">시도 {attempted}</span>
        <span class="nk-pill">성공 {success}</span>
        <span class="nk-pill nk-pill-warn">실패 {fail}</span>
        <span class="nk-pill nk-pill-warn">짧아서 스킵 {skip_short}</span>
        <span class="nk-pill">비네이버 스킵 {skip_non}</span>
        """,
        unsafe_allow_html=True
    )
    st.caption(f"마지막 크롤링 에러: {last_crawl_err or '-'}")

    # 원인 힌트
    st.markdown("**원인 힌트(빠른 판단)**")
    hints = []
    if api_success_rate < 30:
        hints.append("- API 성공률이 낮습니다 → 배포 네트워크/레이트리밋/인증 문제 가능성이 큽니다.")
    if api_success_rate >= 70 and crawl_success_rate < 30:
        hints.append("- API는 정상인데 크롤링 성공률이 낮습니다 → 크롤링 차단(봇 차단) 또는 본문 선택자 변화 가능성이 큽니다.")
    if candidate == 0:
        hints.append("- 네이버 뉴스 링크(n.news.naver)가 거의 없습니다 → 검색 결과 링크가 다른 도메인 위주일 수 있습니다.")
    if not hints:
        hints.append("- 큰 이상 징후가 없습니다. 분량을 늘리거나 키워드를 조정해 보세요.")
    st.markdown("\n".join(hints))

    st.markdown("</div>", unsafe_allow_html=True)


def render_results_tabs(options: dict, user_keyword: str) -> None:
    """탭 기반 결과 UI: 요약 / 기사 목록 / 키워드 표"""
    if not st.session_state.get("result_ready", False):
        st.info("검색 실행 후 결과가 여기에 표시됩니다.")
        return

    final_keyword = st.session_state.get("final_keyword", "")
    df_items: pd.DataFrame = st.session_state.get("df_items", pd.DataFrame())
    df_kw_top50: pd.DataFrame = st.session_state.get("df_kw_top50", pd.DataFrame())
    df_kw_top20: pd.DataFrame = st.session_state.get("df_kw_top20", pd.DataFrame())

    wc_png: bytes = st.session_state.get("wc_png", b"")
    top20_png: bytes = st.session_state.get("top20_png", b"")
    images_zip: bytes = st.session_state.get("images_zip", b"")

    api_metrics: dict = st.session_state.get("api_metrics", {})
    crawl_metrics: dict = st.session_state.get("crawl_metrics", {})

    tab_summary, tab_articles, tab_keywords = st.tabs(["요약", "기사 목록", "키워드 표"])

    # ---------------------------
    # 요약 탭
    # ---------------------------
    with tab_summary:
        st.subheader(f"분석 요약: {final_keyword}")

        # ✅ 진단 패널(항상 표시)
        if api_metrics or crawl_metrics:
            render_diagnostics_panel(api_metrics, crawl_metrics)

        # 키워드가 없을 수도 있으니 방어
        if not df_kw_top50.empty:
            render_top5_badges(df_kw_top50)

        # 시각화는 생성된 경우만
        if wc_png and top20_png:
            left, right = st.columns(2)
            with left:
                st.caption("워드클라우드")
                st.image(wc_png, use_container_width=True)
            with right:
                st.caption("Top20 막대차트")
                st.image(top20_png, use_container_width=True)
        else:
            st.info("시각화 결과가 없습니다. (API/크롤링/분석 단계에서 실패했을 수 있습니다)")

        # 다운로드 액션바(1줄 3버튼)
        with st.container(border=True):
            st.subheader("결과 다운로드")

            can_articles = not df_items.empty
            can_keywords = not df_kw_top50.empty
            can_images = bool(images_zip)

            base = safe_filename(final_keyword or "result")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            b1, b2, b3 = st.columns(3)
            with b1:
                st.download_button(
                    label="기사 목록 다운로드(.csv)",
                    data=df_to_csv_bytes(df_items) if can_articles else b"",
                    file_name=f"articles_{base}_{ts}.csv",
                    mime="text/csv",
                    disabled=not (options["dl_articles"] and can_articles),
                )
            with b2:
                st.download_button(
                    label="키워드 표 다운로드(.csv)",
                    data=df_to_csv_bytes(df_kw_top50) if can_keywords else b"",
                    file_name=f"keywords_{base}_{ts}.csv",
                    mime="text/csv",
                    disabled=not (options["dl_keywords"] and can_keywords),
                )
            with b3:
                st.download_button(
                    label="이미지 다운로드(.png)",
                    data=images_zip if can_images else b"",
                    file_name=f"images_{base}_{ts}.zip",
                    mime="application/zip",
                    disabled=not (options["dl_images"] and can_images),
                )

            st.caption("※ 이미지 다운로드는 워드클라우드+Top20 PNG를 ZIP으로 함께 제공합니다.")

    # ---------------------------
    # 기사 목록 탭
    # ---------------------------
    with tab_articles:
        st.subheader("수집된 기사 목록")

        if df_items.empty:
            st.warning("기사 목록이 비어 있습니다.")
            return

        # 정렬/필터 UI
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            sort_order = st.selectbox("정렬", ["최신순", "오래된순"], index=0)
        with fcol2:
            title_filter = st.text_input("제목에 포함된 단어 필터", value="")

        df_view = df_items.copy()

        # 날짜 정렬
        df_view["__dt"] = pd.to_datetime(df_view["pubDate"], errors="coerce")
        df_view = df_view.sort_values("__dt", ascending=(sort_order == "오래된순"))
        if title_filter.strip():
            df_view = df_view[df_view["title"].str.contains(title_filter, case=False, na=False)]
        df_view = df_view.drop(columns="__dt")

        st.divider()

        if not options["show_articles"]:
            st.info("사이드바에서 '기사 목록 보기'를 켜면 표시됩니다.")
            return

        highlight_key = user_keyword.strip()

        MAX_SHOW = 60
        df_show = df_view.head(MAX_SHOW)

        st.caption(f"표시 기사 수: {len(df_show)} / 필터링 후 전체: {len(df_view)}")

        for _, row in df_show.iterrows():
            title = row.get("title", "")
            pub = row.get("pubDate", "")
            link = row.get("link", "")

            title_html = highlight_keyword(title, highlight_key)

            st.markdown(
                f"""
                <div class="nk-card">
                    <div style="font-weight:800; font-size:16px; line-height:1.35;">
                        {title_html}
                    </div>
                    <div style="opacity:0.75; font-size:13px; margin-top:4px;">
                        {pub}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if options["show_links"] and link:
                st.markdown(
                    f'- <a class="nk-link" href="{link}" target="_blank">🔗 기사 바로가기</a>',
                    unsafe_allow_html=True
                )

        if len(df_view) > MAX_SHOW:
            st.info(f"기사 목록이 많아 상위 {MAX_SHOW}개만 표시했습니다. (필터를 더 걸어보세요)")

    # ---------------------------
    # 키워드 표 탭
    # ---------------------------
    with tab_keywords:
        st.subheader("키워드(TF-IDF) 상위 50")

        if df_kw_top50.empty:
            st.warning("키워드 표가 비어 있습니다.")
            return

        if options["show_keywords"]:
            st.dataframe(df_kw_top50, use_container_width=True)
        else:
            st.info("사이드바에서 '키워드 표 보기'를 켜면 표시됩니다.")


# ============================================================
# 14) 앱 실행(메인)
# ============================================================
def run_app():
    st.set_page_config(page_title="뉴스 키워드 어플리케이션", layout="wide")

    inject_theme_friendly_css()
    setup_matplotlib_korean_font()

    render_header_with_lottie_and_center_title()

    render_sidebar_api_settings()

    # ✅ 사이드바 순서: 옵션 -> 불용어
    options = render_sidebar_options()
    stopwords = render_sidebar_stopwords()

    form = render_search_form()

    status_box = st.empty()
    progress_bar = st.progress(0)

    st.session_state.setdefault("result_ready", False)

    if form["submitted"]:
        clear_results_session()
        progress_bar.progress(0)
        run_pipeline(form, stopwords, status_box, progress_bar)

    render_results_tabs(options, user_keyword=form["user_keyword"])


if __name__ == "__main__":
    run_app()

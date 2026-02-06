# app_v4_with_crawl_stats.py
#
# ============================================================
#  News Keyword Visualizer V4
# ------------------------------------------------------------
#  ✅ V3 기능은 유지하면서 UI만 고도화한 버전
#     (UI Improved + Safe Guards)
#
#  + [추가 개선]
#    ✅ 배포 환경에서 크롤링 성공률(성공/실패/실패원인)을 UI에 표시
#       - 실패 원인 분류:
#         * timeout / connection / request_exception
#         * http_403 / http_429 / http_other
#         * no_selector(#dic_area 없음)
#         * too_short(정제 후 너무 짧음)
#         * not_naver(네이버 뉴스 링크 아님)
#
#  ------------------------------------------------------------
#  실행:
#     streamlit run app_v4_with_crawl_stats.py
# ============================================================

import json
import re
import pickle
import html
from datetime import datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from urllib.parse import quote
import zipfile
from collections import Counter

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

# 크롤링 판단을 위한 임계값
MIN_BODY_LEN = 100  # clean_text_keep_korean 이후 최소 길이
CRAWL_TIMEOUT = 10  # requests timeout(초)


# ============================================================
# 1) 테마 친화 CSS (라이트/다크 공용)
# ============================================================
def inject_theme_friendly_css() -> None:
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
          mark{
            background: rgba(245, 158, 11, 0.28);
          }
        }

        .nk-link{
            opacity: 0.92;
            font-weight: 650;
        }

        /* 크롤링 상태 박스 */
        .nk-stat{
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top: 6px;
        }
        .nk-pill{
            padding:6px 10px;
            border-radius:999px;
            border:1px solid rgba(128,128,128,0.25);
            background: rgba(34,197,94,0.10);
            font-weight:750;
            font-size:0.9rem;
        }
        .nk-pill-warn{
            background: rgba(245,158,11,0.12);
        }
        .nk-pill-bad{
            background: rgba(239,68,68,0.12);
        }
        @media (prefers-color-scheme: dark) {
          .nk-pill{
            border:1px solid rgba(255,255,255,0.18);
          }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 2) matplotlib 한글 폰트 설정
# ============================================================
def setup_matplotlib_korean_font() -> None:
    try:
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    except Exception:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3) 리소스 로딩(캐시)
# ============================================================
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_stopwords_file(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {w.strip() for w in f if w.strip()}
    except Exception:
        return set()


@st.cache_resource
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ============================================================
# 4) 텍스트 유틸
# ============================================================
def clean_title(raw_title: str) -> str:
    t = html.unescape(raw_title or "")
    t = re.sub(r"<.*?>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def format_pubdate(pub_date: str) -> str:
    try:
        dt = parsedate_to_datetime(pub_date)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return pub_date or ""


@st.cache_data(show_spinner=False)
def clean_text_keep_korean(text: str) -> str:
    text = re.sub(r"\d|[a-zA-Z]|\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_token(t: str) -> str:
    if t is None:
        return ""
    t = str(t).strip()
    t = re.sub(r"[\"'“”‘’\(\)\[\]\{\},\.\!\?\:\;]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def build_final_keyword(category: str, user_keyword: str) -> str:
    category = (category or "").strip()
    user_keyword = re.sub(r"\s+", " ", (user_keyword or "")).strip()
    return f"{category} {user_keyword}".strip()


def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-가-힣]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "result"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def highlight_keyword(text: str, keyword: str) -> str:
    if not keyword:
        return text
    try:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)
    except Exception:
        return text


# ============================================================
# 5) 네이버 API(방어 코드)
# ============================================================
def naver_news_api_request(keyword: str, display: int, start: int, client_id: str, client_secret: str) -> list[dict]:
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

    if res.status_code != 200:
        st.error("API 요청 실패")
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
    items: list[dict] = []
    page_count = max(1, total_display // 100)

    for i in range(page_count):
        start = 100 * i + 1
        page_items = naver_news_api_request(final_keyword, 100, start, client_id, client_secret)
        if page_items:
            items.extend(page_items)

    return items


def build_items_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for it in items:
        rows.append({
            "title": clean_title(it.get("title", "")),
            "pubDate": format_pubdate(it.get("pubDate", "")),
            "link": it.get("link", ""),
        })
    return pd.DataFrame(rows)


# ============================================================
# 6) 크롤링(실패 skip) + ✅ 성공률/원인 집계용 상태 반환
# ============================================================
def crawl_naver_news_body_with_status(url: str) -> tuple[str, str, int | None]:
    """
    본문 크롤링 결과를 (body_text, status_label, http_code)로 반환합니다.

    status_label 예시:
    - ok
    - timeout
    - connection
    - request_exception
    - http_403
    - http_429
    - http_other
    - no_selector
    """
    try:
        res = rq.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=CRAWL_TIMEOUT)

        if res.status_code == 403:
            return "", "http_403", 403
        if res.status_code == 429:
            return "", "http_429", 429
        if res.status_code != 200:
            return "", "http_other", res.status_code

        soup = bs4.BeautifulSoup(res.text, "html.parser")
        tag = soup.select_one("#dic_area")
        if not tag:
            return "", "no_selector", res.status_code

        body = tag.get_text(separator=" ", strip=True)
        return body, "ok", res.status_code

    except rq.exceptions.Timeout:
        return "", "timeout", None
    except rq.exceptions.ConnectionError:
        return "", "connection", None
    except rq.exceptions.RequestException:
        return "", "request_exception", None
    except Exception:
        return "", "request_exception", None


def collect_corpus_from_items_with_stats(items: list[dict]) -> tuple[list[str], dict]:
    """
    ✅ (개선) 본문 수집 + 성공/실패 통계를 함께 반환

    반환:
    - docs_clean: 분석 가능한 본문 리스트
    - stats: 크롤링 성공률 판단용 dict
      {
        "total_items": int,
        "naver_links": int,
        "ok": int,
        "failed": int,
        "too_short": int,
        "no_selector": int,
        "timeout": int,
        "connection": int,
        "request_exception": int,
        "http_403": int,
        "http_429": int,
        "http_other": int,
        "by_reason": {reason: count, ...}
      }
    """
    docs_clean: list[str] = []
    reason_counter = Counter()

    total_items = len(items)
    naver_links = 0

    for it in items:
        link = it.get("link", "")
        if "n.news.naver" not in link:
            reason_counter["not_naver"] += 1
            continue

        naver_links += 1

        body, status, _http = crawl_naver_news_body_with_status(link)
        if status != "ok":
            reason_counter[status] += 1
            continue

        cleaned = clean_text_keep_korean(body)
        if len(cleaned) < MIN_BODY_LEN:
            reason_counter["too_short"] += 1
            continue

        reason_counter["ok"] += 1
        docs_clean.append(cleaned)

    ok = int(reason_counter.get("ok", 0))
    failed = naver_links - ok  # 네이버 링크 중 성공(ok) 아닌 것

    stats = {
        "total_items": total_items,
        "naver_links": naver_links,
        "ok": ok,
        "failed": max(0, failed),
        "too_short": int(reason_counter.get("too_short", 0)),
        "no_selector": int(reason_counter.get("no_selector", 0)),
        "timeout": int(reason_counter.get("timeout", 0)),
        "connection": int(reason_counter.get("connection", 0)),
        "request_exception": int(reason_counter.get("request_exception", 0)),
        "http_403": int(reason_counter.get("http_403", 0)),
        "http_429": int(reason_counter.get("http_429", 0)),
        "http_other": int(reason_counter.get("http_other", 0)),
        "not_naver": int(reason_counter.get("not_naver", 0)),
        "by_reason": dict(reason_counter),
    }

    return docs_clean, stats


def infer_crawl_root_cause(stats: dict) -> str:
    """
    통계로부터 "가장 가능성 높은 원인"을 간단히 추론합니다.
    (정답 판정은 불가능하지만, 배포 환경에서 디버깅 방향을 잡는 용도)
    """
    naver_links = stats.get("naver_links", 0)
    if naver_links <= 0:
        return "네이버 뉴스 링크가 거의 없습니다(검색 결과가 다른 언론사 링크 위주)."

    ok = stats.get("ok", 0)
    if ok == 0:
        # 전부 실패일 때: 어떤 실패가 지배적인지
        if stats.get("http_403", 0) + stats.get("http_429", 0) >= max(1, naver_links // 2):
            return "차단/레이트리밋(403/429) 의심: 봇 차단 또는 요청 과다 가능성이 큽니다."
        if stats.get("timeout", 0) >= max(1, naver_links // 2):
            return "네트워크 지연/타임아웃 우세: 배포 환경 네트워크 또는 대상 서버 응답 지연 가능성이 큽니다."
        if stats.get("no_selector", 0) >= max(1, naver_links // 2):
            return "본문 셀렉터(#dic_area) 미검출 우세: 기사 DOM 구조 변경/다른 페이지 유형 가능성이 큽니다."
        return "복합 원인(네트워크/차단/파싱) 가능성: 상세 실패 유형을 확인하세요."

    # 부분 성공일 때
    success_rate = ok / max(1, naver_links)
    if success_rate < 0.3:
        if stats.get("http_403", 0) + stats.get("http_429", 0) > stats.get("timeout", 0):
            return "성공률이 낮고 403/429가 많습니다: 차단/요청 과다 가능성이 큽니다."
        if stats.get("timeout", 0) > stats.get("http_403", 0) + stats.get("http_429", 0):
            return "성공률이 낮고 timeout이 많습니다: 네트워크 지연/응답 지연 가능성이 큽니다."
        return "성공률이 낮습니다: 실패 유형 비율을 보고 원인을 좁혀보세요."

    return "크롤링은 대체로 정상입니다(필터/불용어/분량 설정이 결과에 더 큰 영향을 줄 수 있습니다)."


# ============================================================
# 7) 분석(soynlp 명사 set + TF-IDF)
# ============================================================
@st.cache_data(show_spinner=False)
def build_noun_set(docs_clean: list[str]) -> set[str]:
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
    df_kw = (
        pd.DataFrame(list(score_dict.items()), columns=["keyword", "score"])
        .sort_values("score", ascending=False)
    )
    return df_kw, df_kw.head(50).copy(), df_kw.head(20).copy()


# ============================================================
# 8) 시각화(이미지 bytes)
# ============================================================
def fig_to_png_bytes(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def make_wordcloud_png(freq: dict[str, float], mask_name: str) -> bytes | None:
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
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}_wordcloud.png", wordcloud_png)
        zf.writestr(f"{base_name}_top20.png", top20_png)
    zip_buf.seek(0)
    return zip_buf.getvalue()


# ============================================================
# 9) UI 렌더링
# ============================================================
def render_header_with_lottie_and_center_title():
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
    st.sidebar.header("불용어(Stopwords) :")
    base_stop = load_stopwords_file(STOPWORDS_PATH)
    extra_stop = st.sidebar.text_area("추가 불용어(줄바꿈으로 입력)", value="", height=120)
    stopwords = base_stop | {w.strip() for w in extra_stop.splitlines() if w.strip()}
    st.sidebar.caption(f"현재 불용어 수: {len(stopwords)} (파일 + 추가 입력)")
    return stopwords


def render_search_form():
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
# 10) 결과 세션 저장/초기화
# ============================================================
def save_results_to_session(
    final_keyword: str,
    df_items: pd.DataFrame,
    df_kw_top50: pd.DataFrame,
    df_kw_top20: pd.DataFrame,
    wc_png: bytes,
    top20_png: bytes,
    zip_bytes: bytes,
    crawl_stats: dict,
):
    st.session_state["result_ready"] = True
    st.session_state["final_keyword"] = final_keyword
    st.session_state["df_items"] = df_items
    st.session_state["df_kw_top50"] = df_kw_top50
    st.session_state["df_kw_top20"] = df_kw_top20
    st.session_state["wc_png"] = wc_png
    st.session_state["top20_png"] = top20_png
    st.session_state["images_zip"] = zip_bytes
    st.session_state["crawl_stats"] = crawl_stats  # ✅ 추가


def clear_results_session():
    st.session_state["result_ready"] = False
    for k in ["final_keyword", "df_items", "df_kw_top50", "df_kw_top20", "wc_png", "top20_png", "images_zip", "crawl_stats"]:
        if k in st.session_state:
            del st.session_state[k]


# ============================================================
# 11) 파이프라인 실행(상태박스+진행바)
# ============================================================
def run_pipeline(form: dict, stopwords: set[str], status_box, progress_bar):
    if not form["user_keyword"].strip():
        st.warning("검색 키워드를 입력해 주세요. (예: 금리, 반도체, AI)")
        return

    client_id = st.session_state.get("client_id", "").strip()
    client_secret = st.session_state.get("client_secret", "").strip()
    if not client_id or not client_secret:
        st.error("API 인증 정보(Client ID/Secret)가 설정되지 않았습니다. 사이드바에서 입력 후 다시 시도하세요.")
        return

    final_keyword = build_final_keyword(form["category"], form["user_keyword"])

    # 1) API 수집
    status_box.info(f"1/4 뉴스 목록 수집 중... (검색어: {final_keyword})")
    progress_bar.progress(0.2)
    items = fetch_news_items(final_keyword, form["display"], client_id, client_secret)

    if not items:
        status_box.error("뉴스 목록을 가져오지 못했습니다.")
        st.info("가능한 원인: (1) 인증 실패 (2) 네트워크 오류 (3) 검색 결과 없음")
        return

    df_items = build_items_dataframe(items)

    # 2) 크롤링 + ✅ 통계 수집
    status_box.info("2/4 뉴스 본문 크롤링 중...")
    progress_bar.progress(0.45)
    docs_clean, crawl_stats = collect_corpus_from_items_with_stats(items)

    # ✅ 통계/추론을 상태박스 아래 즉시 보여주면 배포 디버깅이 쉬움
    ok = crawl_stats.get("ok", 0)
    naver_links = max(1, crawl_stats.get("naver_links", 0))
    success_rate = ok / naver_links * 100

    status_box.info(f"크롤링 성공률: {success_rate:.1f}% (성공 {ok} / 네이버링크 {crawl_stats.get('naver_links', 0)})")

    # 2-1) 데이터 부족
    if len(docs_clean) < 5:
        status_box.warning("본문 데이터가 부족하여 분석이 어렵습니다.")
        st.info(
            "개선 팁:\n"
            "- 분량을 300~500으로 늘려보세요.\n"
            "- 키워드를 더 일반적으로 바꿔보세요.\n"
            "- 기사 목록에서 네이버 뉴스 링크가 충분한지 확인해보세요."
        )
        # ✅ 부족한 이유도 같이 표시
        st.warning(f"원인 추정: {infer_crawl_root_cause(crawl_stats)}")
        st.session_state["crawl_stats_preview"] = crawl_stats
        return

    # 3) 분석
    status_box.info("3/4 키워드 분석 중(명사 필터 + TF-IDF)...")
    progress_bar.progress(0.7)
    docs_tokens = tokenize_and_filter_docs(docs_clean, stopwords)

    score_dict = compute_tfidf_scores(docs_tokens, top_k=80)
    if not score_dict:
        status_box.warning("키워드 점수를 계산할 수 없습니다(데이터/필터 조건 부족).")
        st.info("개선 팁: 분량을 늘리거나 불용어를 과도하게 추가하지 않았는지 확인하세요.")
        st.warning(f"원인 추정(크롤링 관점): {infer_crawl_root_cause(crawl_stats)}")
        st.session_state["crawl_stats_preview"] = crawl_stats
        return

    _, df_kw_top50, df_kw_top20 = build_keyword_tables(score_dict)

    # 4) 시각화 생성
    status_box.info("4/4 시각화 생성 중...")
    progress_bar.progress(0.9)

    wc_png = make_wordcloud_png(score_dict, form["mask"])
    top20_png = make_top20_bar_png(df_kw_top20)

    if not wc_png or not top20_png:
        status_box.error("시각화 생성에 실패했습니다(데이터 부족/렌더링 오류).")
        st.warning(f"원인 추정(크롤링 관점): {infer_crawl_root_cause(crawl_stats)}")
        st.session_state["crawl_stats_preview"] = crawl_stats
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
        crawl_stats=crawl_stats,
    )


# ============================================================
# 12) 결과 탭 UI
# ============================================================
def render_top5_badges(df_kw_top50: pd.DataFrame) -> None:
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


def render_crawl_stats_panel(crawl_stats: dict) -> None:
    """
    ✅ 크롤링 성공률/실패 원인을 UI로 보여주는 패널
    - 한 줄 요약 + 실패유형별 카운트 + 원인 추정
    """
    if not crawl_stats:
        return

    total_items = crawl_stats.get("total_items", 0)
    naver_links = crawl_stats.get("naver_links", 0)
    ok = crawl_stats.get("ok", 0)
    failed = crawl_stats.get("failed", 0)

    success_rate = (ok / max(1, naver_links)) * 100

    # 빠른 진단용 핵심 카운트
    http_block = crawl_stats.get("http_403", 0) + crawl_stats.get("http_429", 0)
    timeout_cnt = crawl_stats.get("timeout", 0)
    selector_cnt = crawl_stats.get("no_selector", 0)
    too_short = crawl_stats.get("too_short", 0)

    st.markdown(
        f"""
        <div class="nk-card">
          <div style="font-weight:850; font-size:1.05rem;">크롤링 성공률</div>
          <div class="nk-stat">
            <span class="nk-pill">성공률 {success_rate:.1f}%</span>
            <span class="nk-pill">성공 {ok}</span>
            <span class="nk-pill nk-pill-bad">실패 {failed}</span>
            <span class="nk-pill nk-pill-warn">네이버링크 {naver_links}</span>
            <span class="nk-pill">전체 items {total_items}</span>
          </div>
          <div style="margin-top:10px; opacity:0.85; font-weight:750;">
            원인 추정: {infer_crawl_root_cause(crawl_stats)}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("상세 실패 유형 보기(디버깅용)"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("403/429(차단 의심)", http_block)
        with c2:
            st.metric("timeout", timeout_cnt)
        with c3:
            st.metric("#dic_area 없음", selector_cnt)
        with c4:
            st.metric("너무 짧은 본문", too_short)

        detail_rows = []
        by_reason = crawl_stats.get("by_reason", {})
        for k, v in sorted(by_reason.items(), key=lambda x: x[1], reverse=True):
            detail_rows.append({"reason": k, "count": v})

        if detail_rows:
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


def render_results_tabs(options: dict, user_keyword: str) -> None:
    if not st.session_state.get("result_ready", False):
        # 파이프라인 중간 실패 시에도 preview 통계만 보여주고 싶다면
        preview = st.session_state.get("crawl_stats_preview")
        if preview:
            st.info("이전 실행에서 크롤링 통계가 남아있습니다.")
            render_crawl_stats_panel(preview)
        else:
            st.info("검색 실행 후 결과가 여기에 표시됩니다.")
        return

    final_keyword = st.session_state["final_keyword"]
    df_items: pd.DataFrame = st.session_state["df_items"]
    df_kw_top50: pd.DataFrame = st.session_state["df_kw_top50"]
    df_kw_top20: pd.DataFrame = st.session_state["df_kw_top20"]

    wc_png: bytes = st.session_state["wc_png"]
    top20_png: bytes = st.session_state["top20_png"]
    images_zip: bytes = st.session_state["images_zip"]
    crawl_stats: dict = st.session_state.get("crawl_stats", {})

    tab_summary, tab_articles, tab_keywords = st.tabs(["요약", "기사 목록", "키워드 표"])

    # ---------------------------
    # 요약 탭
    # ---------------------------
    with tab_summary:
        st.subheader(f"분석 요약: {final_keyword}")

        # ✅ 크롤링 성공률 패널
        render_crawl_stats_panel(crawl_stats)

        if not df_kw_top50.empty:
            render_top5_badges(df_kw_top50)

        left, right = st.columns(2)
        with left:
            st.caption("워드클라우드")
            st.image(wc_png, use_container_width=True)
        with right:
            st.caption("Top20 막대차트")
            st.image(top20_png, use_container_width=True)

        with st.container(border=True):
            st.subheader("결과 다운로드")

            can_articles = not df_items.empty
            can_keywords = not df_kw_top50.empty
            can_images = bool(images_zip)

            base = safe_filename(final_keyword)
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

        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            sort_order = st.selectbox("정렬", ["최신순", "오래된순"], index=0)
        with fcol2:
            title_filter = st.text_input("제목에 포함된 단어 필터", value="")

        df_view = df_items.copy()
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
                st.markdown(f'- <a class="nk-link" href="{link}" target="_blank">🔗 기사 바로가기</a>', unsafe_allow_html=True)

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
# 13) 앱 실행(메인)
# ============================================================
def run_app():
    st.set_page_config(page_title="뉴스 키워드 어플리케이션", layout="wide")

    inject_theme_friendly_css()
    setup_matplotlib_korean_font()
    render_header_with_lottie_and_center_title()

    render_sidebar_api_settings()

    # ✅ 사이드바: 옵션 -> 불용어
    options = render_sidebar_options()
    stopwords = render_sidebar_stopwords()

    form = render_search_form()

    status_box = st.empty()
    progress_bar = st.progress(0)

    st.session_state.setdefault("result_ready", False)

    if form["submitted"]:
        clear_results_session()
        # 이전 실행에서 남아있을 수 있는 preview 제거
        if "crawl_stats_preview" in st.session_state:
            del st.session_state["crawl_stats_preview"]

        progress_bar.progress(0)
        run_pipeline(form, stopwords, status_box, progress_bar)

    render_results_tabs(options, user_keyword=form["user_keyword"])


if __name__ == "__main__":
    run_app()

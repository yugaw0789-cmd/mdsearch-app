import os, sqlite3, numpy as np, re, datetime
import streamlit as st
from openai import OpenAI

DB_PATH_DEFAULT = "mdindex.sqlite"
EXPORT_DIR = "./exports"
EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o"

# ---------- DB helpers ----------
@st.cache_resource
def get_conn(db_path: str):
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def exact_fts(con, query: str, k: int, phrase: bool):
    match = f'"{query}"' if phrase else query
    sql = """
    SELECT p.id, p.file, p.section, p.start, p.text,
           snippet(passages_fts,-1,'[',']',' … ',32) AS snippet
    FROM passages_fts
    JOIN passages p ON passages_fts.rowid = p.id
    WHERE passages_fts MATCH ?
    LIMIT ?
    """
    cur = con.cursor()
    cur.execute(sql, (match, k))
    return cur.fetchall()

def substring_like(con, query: str, k: int):
    sql = """
    SELECT id, file, section, substr(text, max(1, instr(text, ?)-40), 160) AS snippet
    FROM passages
    WHERE text LIKE ?
    LIMIT ?
    """
    cur = con.cursor()
    cur.execute(sql, (query, f"%{query}%", k))
    return cur.fetchall()

def fetch_vectors(con):
    cur = con.cursor()
    cur.execute("SELECT passage_id, dim, vec FROM vectors")
    rows = cur.fetchall()
    ids, vecs = [], []
    for pid, dim, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.shape[0] == dim:
            ids.append(pid)
            vecs.append(v)
    return (np.vstack(vecs) if vecs else np.zeros((0,1536),dtype=np.float32)), ids

def fetch_passages_by_ids(con, ids):
    if not ids:
        return []
    cur = con.cursor()
    q = "SELECT id, file, section, text FROM passages WHERE id IN (%s)" % ",".join("?"*len(ids))
    cur.execute(q, ids)
    rows = {r["id"]: r for r in cur.fetchall()}
    return [rows[i] for i in ids if i in rows]

def fetch_section_text(con, file: str, section: str) -> str:
    cur = con.cursor()
    cur.execute(
        """
        SELECT text FROM passages
        WHERE file = ? AND section = ?
        ORDER BY start ASC
        """,
        (file, section),
    )
    rows = cur.fetchall()
    return "\n\n".join(r[0] for r in rows) if rows else ""

# ---------- 見出し評価（優先度付け） ----------
_heading_pat = re.compile(r"^(#+)\s*(.+?)\s*$")

def parse_section(section: str):
    m = _heading_pat.match(section.strip())
    if not m:
        return 6, section.strip()
    level = len(m.group(1))  # #数。少ないほど上位
    title = m.group(2).strip()
    return level, title

def section_rank(section: str, query: str) -> float:
    level, title = parse_section(section)
    q = query.lower()
    t = title.lower()
    hit = (q in t)
    starts = t.startswith(q)
    exact = (t == q)
    score = 0.0
    if hit:    score += 50
    if starts: score += 20
    if exact:  score += 50
    score += max(0, 30 - 5*max(1, level))  # 階層浅いほど加点
    return score

def ranked_exact_hits(con, query: str, k: int, phrase: bool, pool_factor: int = 4):
    pool = exact_fts(con, query, k*pool_factor, phrase)
    uniq = {}
    for r in pool:
        key = (r["file"], r["section"])
        if key not in uniq:
            uniq[key] = r
    rows = list(uniq.values())
    rows.sort(key=lambda r: section_rank(r["section"], query), reverse=True)
    return rows[:k]

# ヒット位置から次の ### 直前までをファイル横断で連結（連続ページのみ）
def collect_until_next_heading_across_files(
    con, start_file: str, start_offset: int, max_files: int = 5, max_chars: int = 0
) -> str:
    cur = con.cursor()
    cur.execute("SELECT DISTINCT file FROM passages ORDER BY file ASC")
    files = [r[0] for r in cur.fetchall()]
    try:
        i0 = files.index(start_file)
    except ValueError:
        return ""

    h_re = re.compile(r"(?m)^\s*###\s+")
    fence_re = re.compile(r"(?s)```.*?```")

    out_parts, total = [], 0
    files_used, stop, first_chunk_handled = 0, False, False

    for fi in range(i0, len(files)):
        if files_used >= max_files or stop:
            break
        f = files[fi]

        if fi == i0:
            cur.execute(
                "SELECT start, text FROM passages WHERE file=? AND start>=? ORDER BY start ASC",
                (f, start_offset),
            )
        else:
            cur.execute("SELECT start, text FROM passages WHERE file=? ORDER BY start ASC", (f,))
        rows = cur.fetchall()

        for start, text in rows:
            if fi == i0 and start_offset > start:
                cut = start_offset - start
                if 0 < cut < len(text):
                    text = text[cut:]
                elif cut >= len(text):
                    continue

            masked = fence_re.sub(lambda m: " " * (m.end() - m.start()), text)
            search_from = 0
            if not first_chunk_handled:
                m0 = h_re.match(masked)
                if m0:
                    search_from = m0.end()  # 先頭の ### はスキップ
                first_chunk_handled = True

            m = h_re.search(masked, search_from)
            if m:
                seg = text[: m.start()]
                out_parts.append(seg)
                total += len(seg)
                stop = True
                break
            else:
                out_parts.append(text)
                total += len(text)

            if max_chars and total >= max_chars:
                return "".join(out_parts)[:max_chars]

        files_used += 1

    joined = "".join(out_parts)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined

# ---------- OpenAI helpers ----------
@st.cache_resource
def get_client():
    return OpenAI()

def embed(client: OpenAI, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.vstack([np.array(d.embedding, dtype=np.float32) for d in resp.data])

def ai_summarize_from_exact(client: OpenAI, query: str, full_text: str, user_prompt: str, temperature: float = 0.2):
    system = (
        "あなたは技術文書を正確に整理するアシスタントです。"
        "入力された原文からのみ情報を抜き出し、推測は最小限にする。"
        "出力はMarkdownで、必ず以下の順序で構成:\n"
        "## 結論\n- 要点を簡潔に\n"
        "## 詳細整理\n- 表や小見出しで論理的にまとめる\n"
        "## 注意・不明点\n- 原文に無い事項は『原文に記載なし』と明記"
    )
    user = f"検索ワード: {query}\n\nユーザープロンプト:\n{user_prompt}\n\n[原文]\n{full_text}"
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

# ---------- 人手用：根拠選択UIのための整形 ----------
def build_evidences(con, rows, query: str, max_files: int, max_chars: int):
    """rank済み rows から (dict) evidences を作る"""
    evs = []
    for r in rows:
        joined = collect_until_next_heading_across_files(
            con, r["file"], r["start"], max_files=max_files, max_chars=max_chars
        )
        if not joined:
            continue
        level, title = parse_section(r["section"])
        score = section_rank(r["section"], query)
        key = f"{r['file']}|{r['section']}|{r['start']}"
        default_selected = (query.lower() in title.lower()) or (joined.lower().count(query.lower()) >= 1)
        evs.append({
            "key": key,
            "file": r["file"],
            "section": r["section"],
            "title": title,
            "level": level,
            "score": score,
            "text": joined,
            "selected": default_selected
        })
    # score降順で見やすく
    evs.sort(key=lambda x: x["score"], reverse=True)
    return evs

def combined_text_from_selected(evs):
    parts = [e["text"] for e in evs if e["selected"]]
    return "\n\n------\n\n".join(parts)

# ---------- UI ----------
st.set_page_config(page_title="Markdown 検索ツール", layout="wide")
st.title("🔎 Markdown 検索ツール")

with st.sidebar:
    st.header("設定")
    db_path = st.text_input("DBファイル", value=DB_PATH_DEFAULT)
    k = st.slider("Top-K", 1, 30, 10)
    mode = st.radio("検索モード", ["完全一致（FTS）", "部分一致（LIKE）", "意味検索（Embedding）", "抽出回答（RAG）", "完全一致＋AI整形"])
    phrase = st.checkbox("フレーズ完全一致（”…”）", value=False, disabled=(mode!="完全一致（FTS）"))

    # 完全一致（FTS）用
    section_full = st.checkbox("（完全一致）項目全文を表示（###見出し単位）", value=False, disabled=(mode!="完全一致（FTS）"))
    span_pages   = st.checkbox("（完全一致）ヒットページから次見出しまでファイル横断で連結", value=True, disabled=(mode!="完全一致（FTS）"))
    max_files    = st.number_input("（完全一致）連結ページ上限", min_value=1, value=10, step=1, disabled=(mode!="完全一致（FTS）"))
    max_chars    = st.number_input("（完全一致）最大表示文字数（0=無制限）", min_value=0, value=0, step=1000, disabled=(mode!="完全一致（FTS）"))

    # AI整形用
    if mode == "完全一致＋AI整形":
        ai_exact_k   = st.number_input("AI整形: 完全一致のTop-K", min_value=1, value=3, step=1)
        ai_max_files = st.number_input("AI整形: 連結ページ上限", min_value=1, value=10, step=1)
        ai_max_chars = st.number_input("AI整形: 最大文字数（0=無制限）", min_value=0, value=8000, step=1000)
        ai_prompt    = st.text_area("AI整形プロンプト", "製造会社・用途・規制・危険性を整理してください。", height=120)
        ai_temp      = st.slider("AI温度", 0.0, 1.0, 0.2, 0.1)
        save_flag    = st.checkbox("結果をMarkdownで保存", value=True)
        save_dir     = st.text_input("保存先フォルダ", EXPORT_DIR)
        st.caption("⚠️『根拠一覧』でチェックを外すと、除外して再要約できます。")

query = st.text_input("検索ワード（完全一致用）", value="", placeholder="例：酢酸ブチル / 光触媒酸化チタン / 主な油脂の構成表")
run = st.button("検索 / 実行")

# セッション状態
if "evidences" not in st.session_state:
    st.session_state.evidences = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

def render_evidence_selector():
    st.subheader("根拠一覧（人手で選択可能）")
    if not st.session_state.evidences:
        st.info("根拠がありません。")
        return
    # 行ごとにチェック＋メタ情報＋本文
    for i, e in enumerate(st.session_state.evidences):
        cols = st.columns([0.1, 0.55, 0.15, 0.2])
        with cols[0]:
            st.session_state.evidences[i]["selected"] = st.checkbox(
                "", value=e["selected"], key=f"sel_{e['key']}")
        with cols[1]:
            st.markdown(f"**{e['title']}**  \n`{e['file']}`")
        with cols[2]:
            st.caption(f"優先度: {e['score']:.1f}  （階層 #{e['level']}）")
        with cols[3]:
            st.caption(f"key: {e['key']}")
        with st.expander("原文を表示 / 人間による確認", expanded=False):
            st.markdown(e["text"])
        st.divider()

if run and query.strip():
    con = get_conn(db_path)

    if mode == "完全一致＋AI整形":
        # クエリが変わったらエビデンスを作り直し
        if st.session_state.last_query != query:
            st.session_state.evidences = []
            st.session_state.last_query = query

        rows = ranked_exact_hits(con, query.strip(), int(ai_exact_k), phrase)
        if not rows:
            st.warning("完全一致のヒットがありません。")
        else:
            evs = build_evidences(con, rows, query, int(ai_max_files), int(ai_max_chars))
            st.session_state.evidences = evs  # 保持

            # 1) 初回要約（選択状態に従う）
            client = get_client()
            combined_text = combined_text_from_selected(st.session_state.evidences)
            if not combined_text:
                st.warning("選択済みの根拠がありません。少なくとも1つ選択してください。")
            else:
                st.subheader("AI整形結果")
                md = ai_summarize_from_exact(client, query, combined_text, ai_prompt, ai_temp)
                st.markdown(md)

            # 2) 人手で根拠を確認・選択
            render_evidence_selector()

            # 3) 人手選択に基づく再要約
            if st.button("選択した根拠のみで再要約する"):
                combined_text2 = combined_text_from_selected(st.session_state.evidences)
                if not combined_text2:
                    st.warning("選択済みの根拠がありません。少なくとも1つ選択してください。")
                else:
                    st.subheader("AI整形結果（再要約）")
                    md = ai_summarize_from_exact(client, query, combined_text2, ai_prompt, ai_temp)
                    st.markdown(md)

            # 4) 保存（選択済みの根拠のみ）
            if save_flag:
                os.makedirs(save_dir, exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                safe_q = re.sub(r"[^0-9A-Za-z一-龥ぁ-んァ-ヴー]", "_", query)[:50]
                path = os.path.join(save_dir, f"{ts}_{safe_q}_ai-format.md")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("# AI整形結果（最新の表示内容を手動で保存）\n\n")
                    # 現画面に出ている整形結果は直上の実行結果。必要に応じて再要約後に押してください
                    f.write("(※このファイルは選択済み根拠に基づく最新の要約を書き出す設計です。)\n\n")
                    # 直近の選択状態で再生成して保存
                    combined_text3 = combined_text_from_selected(st.session_state.evidences)
                    if combined_text3:
                        md3 = ai_summarize_from_exact(get_client(), query, combined_text3, ai_prompt, ai_temp)
                        f.write(md3 + "\n\n---\n\n")
                    f.write(f"## メタ情報\n- 検索ワード: {query}\n- モデル: {ANSWER_MODEL}\n- 保存日時: {ts}\n\n")
                    f.write("## 根拠（選択済みのみ）\n")
                    for e in st.session_state.evidences:
                        if not e["selected"]:
                            continue
                        f.write(f"### {e['title']}  ({e['file']})\n\n{e['text']}\n\n---\n\n")
                st.success(f"保存しました: {path}")

    elif mode == "完全一致（FTS）":
        rows = ranked_exact_hits(con, query.strip(), int(k), phrase)
        st.subheader(f"完全一致（{len(rows)}件）")
        for r in rows:
            st.markdown(f"**[{r['file']}] {r['section']}**")
            if span_pages:
                joined = collect_until_next_heading_across_files(
                    con, r["file"], r["start"], max_files=int(max_files), max_chars=int(max_chars)
                )
                st.markdown(joined if joined else r["snippet"])
            elif section_full:
                full = fetch_section_text(con, r["file"], r["section"])
                if int(max_chars) and len(full) > int(max_chars):
                    full = full[:int(max_chars)] + " …"
                st.markdown(full if full else r["snippet"])
            else:
                st.markdown(r["snippet"])
            st.divider()

    elif mode == "部分一致（LIKE）":
        rows = substring_like(con, query.strip(), int(k))
        st.subheader(f"部分一致（{len(rows)}件）")
        for r in rows:
            st.markdown(f"**[{r['file']}] {r['section']}**  \n{r['snippet']}")
            st.divider()

    elif mode == "意味検索（Embedding）":
        vecs, ids = fetch_vectors(con)
        if vecs.shape[0] == 0:
            st.error("ベクトルが未作成です。先に CLI で `index --embed` を実行してください。")
        else:
            client = get_client()
            qv = embed(client, [query])[0]
            sims = vecs @ qv / (np.linalg.norm(vecs,axis=1)*np.linalg.norm(qv) + 1e-9)
            top = np.argsort(-sims)[:int(k)]
            sel_ids = [ids[i] for i in top]
            rows = fetch_passages_by_ids(con, sel_ids)
            st.subheader(f"意味検索（{len(rows)}件）")
            for r,score in zip(rows, sims[top]):
                preview = r["text"].replace("\n"," ")
                if len(preview) > 180: preview = preview[:180]+" …"
                st.markdown(f"**[{r['file']}] {r['section']}**  \nscore={score:.3f}  \n{preview}")
                st.divider()

    else:  # 抽出回答（RAG）簡易
        st.info("RAGは簡易表示です。まず『完全一致＋AI整形』で人手確認→再要約をおすすめします。")
        vecs, ids = fetch_vectors(con)
        if vecs.shape[0] == 0:
            st.error("ベクトルが未作成です。先に CLI で `index --embed` を実行してください。")
        else:
            client = get_client()
            qv = embed(client, [query])[0]
            sims = vecs @ qv / (np.linalg.norm(vecs,axis=1)*np.linalg.norm(qv) + 1e-9)
            top = np.argsort(-sims)[:int(k)]
            sel_ids = [ids[i] for i in top]
            rows = fetch_passages_by_ids(con, sel_ids)
            st.subheader(f"RAG候補（意味検索 Top-K）")
            for r,score in zip(rows, sims[top]):
                st.markdown(f"> **[id:{r['id']} | {r['file']} | {r['section']}]** (score={score:.3f})\n\n{r['text']}")
                st.divider()

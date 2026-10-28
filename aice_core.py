# AICE Associate 독학 키트 - 브라우저(Pyodide)에서 실행되는 데이터 생성·실행·채점 코어
# index.html이 이 파일 내용을 그대로 포함해 사용한다. (수정 후 build.py 실행)
import ast, io, json, base64, math, warnings, traceback, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- 합성 데이터
def make_churn(path="churn.csv", n=2400, seed=2026):
    """통신사 고객 이탈(churn) 합성 데이터. 이상치·결측치·지저분한 범주값을 일부러 넣는다."""
    rng = np.random.default_rng(seed)
    plan = rng.choice(["Basic", "Standard", "Premium", "Basic|Promo", "Standard|Promo"],
                      n, p=[0.28, 0.27, 0.2, 0.13, 0.12])
    base_fee = pd.Series(plan).str.split("|").str[0].map({"Basic": 33, "Standard": 55, "Premium": 89}).to_numpy()
    fee = base_fee + rng.normal(0, 6, n)
    promo = np.char.find(plan.astype(str), "Promo") >= 0
    fee = np.where(promo, fee * 0.85, fee)
    out = rng.choice(n, 28, replace=False)
    fee[out] = fee[out] * rng.uniform(3.0, 5.0, 28)          # 요금 이상치
    contract = rng.choice(["Monthly", "1Year", "2Year"], n, p=[0.5, 0.3, 0.2])
    tenure = rng.integers(1, 73, n)
    age = rng.integers(19, 76, n).astype(float)
    complaints = rng.poisson(0.7, n)
    data_gb = np.round(rng.gamma(2.0, 9.0, n), 1)
    calls = np.round(np.clip(rng.normal(280, 90, n), 0, None)).astype(int)
    logit = (-1.1 + 1.4 * (contract == "Monthly") - 0.6 * (contract == "2Year")
             - 0.035 * tenure + 0.65 * complaints + 0.018 * (base_fee - 55) - 0.4 * promo)
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
        "gender": rng.choice(["M", "F"], n),
        "age": age,
        "region": rng.choice(["Seoul", "Busan", "Daegu", "Gwangju", "Daejeon"], n),
        "plan": plan,
        "contract": contract,
        "tenure_months": tenure,
        "monthly_fee": np.round(fee, 1),
        "data_gb": data_gb,
        "call_minutes": calls,
        "complaints": complaints,
        "churn": churn,
    })
    df.loc[rng.choice(n, 37, replace=False), "data_gb"] = np.nan   # 결측치
    df.loc[rng.choice(n, 22, replace=False), "age"] = np.nan
    df.to_csv(path, index=False)
    return df.shape


# ---------------------------------------------------------------- 셀 실행
class _History:
    """딥러닝 학습 결과를 흉내 낸 객체 (14번 문항용, TensorFlow 없이 학습 곡선 실습)"""
    def __init__(self):
        e = np.arange(1, 24)
        loss = 0.62 * np.exp(-e / 6) + 0.38
        val = 0.6 * np.exp(-e / 7) + 0.40 + 0.0025 * np.clip(e - 12, 0, None) ** 1.5
        self.history = {
            "loss": list(np.round(loss, 4)), "val_loss": list(np.round(val, 4)),
            "accuracy": list(np.round(1 - loss * 0.5, 4)), "val_accuracy": list(np.round(1 - val * 0.52, 4)),
        }


def _fig_payload():
    imgs, info = [], []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        imgs.append(base64.b64encode(buf.getvalue()).decode())
        info.append([{"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                      "lines": len(ax.get_lines()), "patches": len(ax.patches),
                      "legend": ax.get_legend() is not None} for ax in fig.axes])
    plt.close("all")
    return imgs, info


def _exec(code, ns):
    """주피터 셀처럼 실행: 마지막 줄이 식이면 그 값을 출력한다."""
    out = io.StringIO()
    error, shown = None, None
    plt.close("all")
    try:
        tree = ast.parse(code)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        if last is not None:
            tree.body = tree.body[:-1]
        with contextlib.redirect_stdout(out):
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
                if val is not None and not (hasattr(val, "__module__") and "matplotlib" in str(type(val))):
                    shown = val.to_string(max_rows=12, max_cols=14) if isinstance(val, pd.DataFrame) else repr(val)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<cell>"), None)
        if isinstance(e, SyntaxError):
            line = e.lineno
        error = f"{type(e).__name__}: {e}" + (f"  (셀 {line}번째 줄)" if line else "")
    imgs, info = _fig_payload()
    return {"stdout": out.getvalue(), "shown": shown, "error": error, "figs": imgs}, info


# ---------------------------------------------------------------- 채점용 사실(facts) 추출
def _mod(ns, name):
    return getattr(ns.get(name), "__name__", None)


def _blank(v):
    return str(v).strip().strip("'\"").strip() if v is not None else None


def _shape(ns, name):
    v = ns.get(name)
    return list(v.shape) if hasattr(v, "shape") else None


def _calls(code):
    names = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call):
            args = [ast.unparse(a) for a in node.args]
            names.append((ast.unparse(node.func), args))
    return names


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def keras_facts(code):
    tree = ast.parse(code)
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset))
    layers, f = [], {}
    for c in calls:
        fn = ast.unparse(c.func).split(".")[-1]
        kw = {k.arg: _lit(k.value) for k in c.keywords if k.arg}
        first = _lit(c.args[0]) if c.args else None
        if fn == "Dense":
            layers.append(f"Dense({first if first is not None else kw.get('units')}, {kw.get('activation')})")
        elif fn == "Dropout":
            layers.append(f"Dropout({first if first is not None else kw.get('rate')})")
        elif fn == "compile":
            f["compile: loss"] = kw.get("loss")
            f["compile: optimizer"] = kw.get("optimizer")
            f["compile: metrics"] = kw.get("metrics")
        elif fn == "EarlyStopping":
            f["EarlyStopping: monitor"] = kw.get("monitor")
            f["EarlyStopping: patience"] = kw.get("patience")
        elif fn == "fit":
            f["fit: epochs"] = kw.get("epochs")
            f["fit: batch_size"] = kw.get("batch_size")
            f["fit: validation_data"] = ast.unparse(c.keywords[[k.arg for k in c.keywords].index("validation_data")].value).replace(" ", "") if "validation_data" in kw else None
            f["fit: callbacks 사용"] = "callbacks" in kw
    f["레이어 구성"] = " → ".join(layers)
    f["fit 결과를 history에 저장"] = any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "history" for t in n.targets)
        and isinstance(n.value, ast.Call) and ast.unparse(n.value.func).endswith(".fit")
        for n in ast.walk(tree))
    return f


def facts(qid, ns, code, figs):
    g = ns.get
    if qid == 1:
        return {"pd 별칭": _mod(ns, "pd"), "np 별칭": _mod(ns, "np"), "sns 별칭": _mod(ns, "sns")}
    if qid == 2:
        return {"df 크기": _shape(ns, "df"), "답안2": g("답안2")}
    if qid == 3:
        return {"답안3": _blank(g("답안3")), "그래프 개수": len(figs)}
    if qid == 4:
        axes = figs[0] if figs else []
        return {"답안4_1": _blank(g("답안4_1")), "답안4_2": _blank(g("답안4_2")),
                "서브플롯 개수": len(axes), "제목": [a["title"] for a in axes]}
    if qid == 5:
        d = g("df_temp")
        return {"df_temp 크기": _shape(ns, "df_temp"), "답안5": g("답안5"),
                "인덱스 재정렬(0부터 연속)": bool(d is not None and list(d.index[:3]) == [0, 1, 2] and d.index[-1] == len(d) - 1)}
    if qid == 6:
        d = g("df_na")
        return {"df_na 크기": _shape(ns, "df_na"), "customer_id 삭제": d is not None and "customer_id" not in d.columns,
                "결측치 0개": d is not None and int(d.isnull().sum().sum()) == 0, "답안6": g("답안6")}
    if qid == 7:
        d = g("df_preset")
        return {"df_preset 크기": _shape(ns, "df_preset"),
                "plan 관련 컬럼": sorted(c for c in d.columns if c.startswith("plan")) if d is not None else None,
                "답안7": g("답안7")}
    if qid == 8:
        yt = g("y_train")
        return {"X_train 크기": _shape(ns, "X_train"), "X_valid 크기": _shape(ns, "X_valid"),
                "y_train 이탈 고객 수": int(yt.sum()) if yt is not None else None,
                "X_train 첫 행 인덱스(분리 시드 확인)": [int(i) for i in g("X_train").index[:3]] if hasattr(g("X_train"), "index") else None}
    if qid == 9:
        bad = any(f.endswith("fit_transform") and a[:1] == ["X_valid"] for f, a in _calls(code))
        return {"검증 데이터에 fit_transform 사용 안 함": not bad, "X_valid 크기": _shape(ns, "X_valid"), "답안9": g("답안9")}
    if qid == 10:
        gs = g("gs")
        ok = gs is not None and hasattr(gs, "best_params_")
        return {"모델 종류": type(gs.estimator).__name__ if gs is not None else None,
                "random_state": gs.estimator.get_params().get("random_state") if gs is not None else None,
                "cv": getattr(gs, "cv", None), "scoring": getattr(gs, "scoring", None),
                "best_params_": dict(sorted(gs.best_params_.items())) if ok else "학습(fit) 안 됨"}
    if qid == 11:
        t = g("top5")
        return {"top5 변수": [str(i) for i in t.index] if hasattr(t, "index") else None,
                "답안11": _blank(g("답안11")), "그래프 개수": len(figs)}
    if qid == 12:
        return {"답안12": g("답안12")}
    if qid == 13:
        return keras_facts(code)
    if qid == 14:
        ax = figs[0][0] if figs and figs[0] else {}
        return {"답안14_1": _blank(g("답안14_1")), "답안14_2": _blank(g("답안14_2")),
                "그래프 제목": ax.get("title"), "선 개수": ax.get("lines"), "범례 표시": ax.get("legend")}
    return {}


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return repr(v)


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-4)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


# ---------------------------------------------------------------- 상태 관리 / JS용 API
NO_EXEC = {13}           # TensorFlow 문항: 실행하지 않고 코드 구조만 채점
USER, REF = {}, {}
REF_FACTS = {}
REF_CODES = {}


def _new_ns(qid=None):
    return {"__name__": "__main__"}


def _prepare(ns, qid):
    if qid == 14 and "history" not in ns:
        ns["history"] = _History()


def api_init(ref_codes_json):
    global USER, REF, REF_FACTS, REF_CODES
    make_churn()
    REF_CODES = {int(k): v for k, v in json.loads(ref_codes_json).items()}
    USER, REF, REF_FACTS = _new_ns(), _new_ns(), {}
    return "ok"


def _ensure_ref(qid):
    for q in range(1, qid + 1):
        if q in REF_FACTS:
            continue
        code = REF_CODES[q]
        _prepare(REF, q)
        figs = []
        if q not in NO_EXEC:
            res, figs = _exec(code, REF)
            if res["error"]:
                raise RuntimeError(f"모범답안 {q}번 실행 오류: {res['error']}")
        REF_FACTS[q] = _jsonable(facts(q, REF, code, figs))


def api_run(qid, code):
    qid = int(qid)
    _ensure_ref(qid)
    _prepare(USER, qid)
    if qid in NO_EXEC:
        try:
            ast.parse(code)
            res = {"stdout": "", "shown": None, "figs": [], "error": None,
                   "note": "브라우저에서는 TensorFlow를 실행할 수 없어 코드 구조(레이어·compile·EarlyStopping·fit 인자)만 채점합니다."}
        except SyntaxError as e:
            res = {"stdout": "", "shown": None, "figs": [], "error": f"SyntaxError: {e.msg} (셀 {e.lineno}번째 줄)"}
        figs = []
    else:
        res, figs = _exec(code, USER)
    try:
        mine = _jsonable(facts(qid, USER, code, figs))
    except Exception as e:
        mine = {"채점 오류": f"{type(e).__name__}: {e}"}
    ref = REF_FACTS[qid]
    diffs = [{"key": k, "mine": mine.get(k), "expected": v} for k, v in ref.items() if not _same(mine.get(k), v)]
    if res["error"]:
        diffs.insert(0, {"key": "실행 오류", "mine": res["error"], "expected": "오류 없이 실행"})
    res["correct"] = not diffs
    res["diffs"] = diffs
    return json.dumps(res, ensure_ascii=False)


def api_sync(qid):
    """앞 문항(1 ~ qid-1)의 모범답안을 실행한 상태로 사용자 변수 공간을 맞춘다."""
    global USER
    qid = int(qid)
    USER = _new_ns()
    for q in range(1, qid):
        _prepare(USER, q)
        if q not in NO_EXEC:
            _exec(REF_CODES[q], USER)
    return "ok"


def api_reset():
    global USER
    USER = _new_ns()
    return "ok"

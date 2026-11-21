# Console Project Agent

一個獨立的主控台應用程式，用 LLM 理解自然語言並管理簡單的專案清單。程式把請求分類成意圖（intent），透過 LangGraph工作流程執行對應動作，並把專案資料持久化到JSON檔。

使用LangChain + LangGraph，搭配Agnes AI相容的LLM端點。

---

## 功能

### 必要意圖
- **create** — 從自然語言建立專案，至少抓取 `project_name` 與 `customer`。若必要欄位缺漏，會持續追問直到能存檔。
- **list** — 以可讀格式列出所有專案。沒有任何專案時會明確告知使用者。
- **get** — 顯示單一專案詳情。
- **update** — 更新既有專案的欄位（`customer`、`start_date`、`location`、`status`、`notes`）。
- **delete** — 刪除既有專案。
- **clear** — 清空所有專案，**強制二次確認**（`yes` / `no`）。

### 選填欄位
`start_date`、`location`、`status`、`notes`。`status` 未提供時預設為 `new`。

### 一般行為
- 持續執行，直到使用者輸入 `quit` 或 `exit`。
- 優雅處理不清楚或無效的輸入，不會當機（磁碟上的壞 JSON、空輸入、Ctrl+C、編號越界、同名專案等）。
- 專案可用**清單編號**（1-based，來自 `list projects`）或**唯一的專案名稱**定位。若多個專案同名，程式會列出候選並要求使用者改用編號重試。

---

## 環境需求

- Python 3.10+(本次測試使用Python 3.12)
- 其餘配置見requirement.txt

---

## 安裝

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`：

```txt
langchain>=0.3
langchain-openai>=0.2
langgraph>=0.2
python-dotenv>=1.0
```

---

## 設定 LLM

在專案根目錄`.env`中調整：

```env
LLM_API_KEY = 【放置api_key】
LLM_BASE_URL = https://apihub.agnes-ai.com/v1【模型供應商】
LLM_MODEL = agnes-2.5-flash【模型名稱】
```

平台：https://platform.agnes-ai.com/settings/apiKeys

---

## 執行應用程式

```bash
python main.py
```

應該會看到：

```
Project Agent ready.
Try commands like:
  create a project called Alpha for customer Acme
  create project Beta for Globex, starts 2026-01-15, location Taipei, status active, notes "kickoff next week"
    (optional fields: start_date, location, status, notes; status defaults to 'new')
  list projects
  get 1 / update 1 status to active / delete 1
  update 1 location Taipei, notes "moved to new office"
    (updatable fields: customer, start_date, location, status, notes)
  clear            (deletes ALL projects, asks for confirmation)
  quit             (or exit / Ctrl+C)

>
```

輸入自然語言指令即可。輸入 `quit` 或 `exit`（或按 Ctrl+C）離開。

---

## LangGraph 工作流程

```
START
  │
  ▼
router ──(條件邊：route)──┬──▶ create  ──▶ END
                          ├──▶ list    ──▶ END
                          ├──▶ get     ──▶ END
                          ├──▶ update  ──▶ END
                          ├──▶ delete  ──▶ END
                          ├──▶ clear   ──▶ END
                          ├──▶ confirm ──▶ END
                          ├──▶ quit    ──▶ END
                          └──▶ unknown ──▶ END
```

### 節點說明

| 節點 | 角色 |
|------|------|
| `router` | 把最新使用者訊息連同嚴格「只回 JSON」的 prompt 送給 LLM，解析成 `intent` 與抽出的欄位（`payload`）。LLM 輸出不是合法 JSON 時退回 `unknown`。 |
| `create` | 檢查必要欄位（`project_name`、`customer`）。缺漏時存進 `pending` 並反問。後續回合會再用 LLM 解析使用者回覆以補齊。齊全後追加到 `projects.json`。 |
| `list` | 讀取 `projects.json` 並印出編號清單，或明確的「沒有專案」訊息。 |
| `get` / `update` / `delete` | 以清單編號或唯一名稱定位目標專案，再讀取／修改／移除。 |
| `clear` | 第一次呼叫設 `pending_confirm=True` 並要求確認。之後非確認的輸入會被重新提醒。 |
| `confirm` | 在 `pending_confirm` 為真時處理 `yes` / `no`。`yes` 清空儲存；`no` 取消。 |
| `quit` | 印出 `Bye.` 並結束迴圈。 |
| `unknown` | 無法辨識輸入時的友善回覆。 |

### 條件路由（`route`）

優先順序：
1. `intent == "quit"` → `quit`（讓使用者即使對話中途也能退出）。
2. `pending_confirm` 為真 → 只接受 `confirm`；其他輸入回到 `clear` 並提醒。
3. `pending`（create 缺欄位）非空 → `create`（繼續追問迴圈）。
4. 其他 → 依 `intent` 分派。

### 共享狀態（`AgentState`）

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    payload: dict          # 本輪抽出的欄位
    pending: list[str]     # create 還缺的必要欄位
    pending_confirm: bool  # 是否正在等待 clear 確認
    reply: str             # 要印給使用者的文字
```

---

## 儲存

專案存在工作目錄下的 `projects.json`。每筆記錄長這樣：

```json
{
  "project_name": "Alpha",
  "customer": "Acme",
  "start_date": null,
  "location": null,
  "status": "new",
  "notes": null
}
```

若檔案不存在或損壞，程式會把儲存視為空的，不會當機。

---

## 範例主控台工作階段

### 1. 標準建立／列出／退出

```
> create a project called Alpha for customer Acme
Done. Created project "Alpha" for customer Acme.
> create project Beta for Globex
Done. Created project "Beta" for customer Globex.
> list projects
Here are your projects (2):
1. Alpha — customer: Acme
2. Beta — customer: Globex
> quit
Bye.
```

### 2. 缺欄位與追問

```
> create a project
我需要更多資訊：project_name, customer。
> Alpha, Acme
Done. Created project "Alpha" for customer Acme.
```

### 3. 加分意圖、同名專案與確認清空

```
> create a project called Alpha for customer Globex
Done. Created project "Alpha" for customer Globex.
> delete project Alpha
Multiple projects named "Alpha". Please retry using the number:
  1. Alpha — customer: Acme
  2. Alpha — customer: Globex
> delete 2
Deleted project "Alpha".
> update 1 status to active, location Taipei
Updated project "Alpha": status=active, location=Taipei.
> get 1
Project "Alpha":
  customer: Acme
  start_date: None
  location: Taipei
  status: active
  notes: None
> clear
This will delete ALL 1 project(s). Type 'yes' to confirm, or 'no' to cancel.
> yes
All projects cleared.
> list projects
You have no projects yet.
> quit
Bye.
```

---

## 專案結構

```
.
├── main.py            # 整個應用程式（儲存、LLM、LangGraph、REPL）
├── requirements.txt
├── README.md
├── AI_USAGE.md
├── .env               # API key
└── projects.json      # 執行時產生
```


## 備註與已知限制

- **清單編號是位置，不是永久 ID。** 刪除專案後其餘專案會重新編號。這與使用者在 `list projects` 看到的一致。
- **同名專案不會自動挑一筆。** 程式會列出候選並要求使用者改用編號重試，避免刪錯。
- **`update` 不能改專案名稱。** 請刪除後重建；若需要改名，可擴充 schema。
- **`get` / `update` / `delete` 不記憶追問狀態。** 若缺 `project_name`，它們會回一句提示，但不會跨回合保存 `pending`；使用者需重新完整下指令。
- **LLM 意圖分類具機率性。** `temperature=0` 下實務上穩定，但較弱的模型偶爾會誤判。程式一律退回 `unknown`，不會當機。
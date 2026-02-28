import json
from pathlib import Path

history_list = json.load(open("./remote_llm_module/chat_histories/20260228_202019.json", "r", encoding="utf-8"))

last_assistant_content = None
for msg in reversed(history_list):  
    if msg["role"] == "assistant":
        last_assistant_content = msg["content"]
        break

if last_assistant_content:
    try:
        qa_list = json.loads(last_assistant_content)
        print(f"成功解析出 {len(qa_list)} 条问答对")
        path = Path.cwd() / "data"
        path.mkdir(exist_ok=True)
        file_path = path / f"non_urgent_qa.json.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        print("JSON 解析失败:", e)
        print("原始内容前 200 字符：")
        print(last_assistant_content[:200])
else:
    print("没有找到 assistant 的回复")
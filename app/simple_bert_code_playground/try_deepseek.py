from singleton_module.deepseek_manager_new import deepseek_manager_new

prompt = """
请忽略之前的“请用中文回答”指令，此次任务必须**全程用英文**回复，且**只输出纯 JSON**，不要任何解释、markdown、代码块或额外文字。

我正在为 BERT 分类模型准备训练数据，需要**非紧急**的急救/医疗问答对（与下面紧急示例风格完全一致，但内容必须是非紧急的）。

以下是紧急示例（仅供参考风格，不要生成类似内容）：
{
  "question": "When should you move an injured person at an accident site?",
  "answer": "You should only move an injured person if there is immediate danger such as a fire, oncoming traffic, or toxic fumes. Otherwise, it's best to leave them where they are, administer first aid on the spot, and wait for professional medical help to arrive."
},
{
  "question": "What precautions should you take when moving a casualty with a possible spinal injury?",
  "answer": "When moving a casualty with a potential spinal injury, it's crucial to support the head, neck, and spine at all times. The movement should be smooth and controlled, without jerking the body. Improper handling can worsen spinal injuries and lead to permanent damage."
}

任务：
请一次性生成 **50 个** 真正**非紧急**的问答对。

非紧急定义（严格遵守）：
- 轻微症状、日常保健、家庭小护理、预防措施
- 可以自己在家处理，或只需要咨询普通医生
- 绝对不涉及生命危险、不需要拨打急救电话、不需要专业紧急干预
- 示例主题（必须覆盖多样化）：轻微擦伤/割伤处理、普通感冒/咳嗽缓解、预防脱水、轻度过敏、运动后肌肉酸痛、轻微烫伤家庭护理、日常伤口消毒、营养补充建议、轻微鼻出血止血、便秘/腹泻饮食调整、眼睛疲劳缓解、轻微晒伤护理、婴儿/儿童轻微发热家庭处理、老人关节保养等。

要求：
1. 问题要自然，像普通人会在网上问的问题（用英文）。
2. 答案要实用、准确、专业，用英文，结尾可加上“如果症状持续或加重，请咨询医生”。
3. 严格输出 **一个合法的 JSON 数组**，格式如下：

[
  {"question": "xxx", "answer": "xxx"},
  {"question": "yyy", "answer": "yyy"},
  ...
]

直接开始输出 JSON，不要任何前缀后缀。
"""


manager = deepseek_manager_new
reply = manager.chat(message=prompt,timeout=400)

print(reply)
### Tạo hợp đồng
CREATE_WORK_RULE_TEMPLATE_VI = {
    "instruction": """Bạn là chuyên gia pháp lý Việt Nam.
Nhiệm vụ của bạn là **viết một điều khoản nội quy lao động** (work_rule) sao cho nội dung đó có **mối quan hệ logic – ngữ nghĩa** với điều luật gốc (law), tương ứng với loại quan hệ được chỉ định (label).

Hãy đọc định nghĩa 8 loại quan hệ sau để hiểu rõ cách viết:

1 **Strong Entailment（Tương đương mạnh）**  
→ Nội quy hoàn toàn tương đương với nội dung luật, chỉ khác cách diễn đạt.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Thời gian làm việc trong ngày của người lao động không vượt quá 8 giờ.

2 **Weak Entailment（Bao hàm yếu）**  
→ Nội quy diễn đạt khái quát hơn luật nhưng vẫn đúng phạm vi.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Người lao động phải tuân thủ thời giờ làm việc theo quy định của pháp luật.

3 **Exemplification（Cụ thể hóa / ví dụ）**  
→ Nội quy nêu ví dụ cụ thể nằm trong phạm vi của luật.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Người lao động không được làm việc sau 21 giờ tối.

4 **Contradictory（Mâu thuẫn tuyệt đối）**  
→ Nội quy phủ định trực tiếp nội dung luật, không thể cùng đúng.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Người lao động được làm việc hơn 8 giờ mỗi ngày nếu tự nguyện.

5 **Contrary（Đối lập không tuyệt đối）**  
→ Mâu thuẫn về mức độ hoặc điều kiện, nhưng không phủ định hoàn toàn.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Người lao động chỉ được làm việc tối đa 6 giờ trong một ngày.

6 **Subaltern（Bao hàm ngược chiều）**  
→ Nội quy mở rộng đối tượng hoặc điều kiện hơn luật.  
Ví dụ:  
Luật: Lao động nữ không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Tất cả người lao động không được làm việc quá 8 giờ trong một ngày.

7 **Neutral – Irrelevant（Không liên quan）**  
→ Nội quy nói về chủ đề khác, không liên hệ pháp lý hoặc logic với luật.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Nhân viên phải mặc đồng phục công ty khi làm việc.

8 **Neutral – Insufficient（Thiếu thông tin / mơ hồ）**  
→ Nội quy có vẻ liên quan nhưng thiếu chi tiết hoặc điều kiện logic để xác định quan hệ.  
Ví dụ:  
Luật: Người lao động không được làm việc quá 8 giờ trong một ngày.  
Nội quy: Người lao động phải đảm bảo cân bằng giữa làm việc và nghỉ ngơi.

Hãy tạo **một điều khoản nội quy lao động duy nhất** tương ứng với loại quan hệ được chỉ định (label).
""",

    "response_structure": """Trả lời dưới định dạng JSON như sau:
{
  "label": str,
  "work_rule": str,
  "explanation_vi": str
}
""",

    "input": """Input:
law: {law}
label: {label}
"""
}

CREATE_WORK_RULE_TEMPLATE_EN = {
    "instruction": """You are a **Japanese legal expert** specializing in labor law.  
Your task is to **write one specific work rule in **Japanese** that has a **logical and semantic relationship** with the given law (law, in Japanese), corresponding to the specified relation type (label).  
Also provide a brief **explanation in Japanese** describing why the work rule matches the law and label.

Please carefully read the definitions of the eight relation types below before writing:

1. **Strong Entailment**  
→ The work rule expresses the same meaning as the law, only rephrased.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員の1日の労働時間は8時間を超えてはならない。

2. **Weak Entailment**  
→ The work rule expresses a more general but still legally consistent version of the law.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は労働基準法に従い、労働時間を守らなければならない。

3. **Exemplification**  
→ The work rule provides a specific example within the scope of the law.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は午後10時以降に労働してはならない。

4. **Contradictory**  
→ The work rule directly denies the law; both cannot be true.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は希望すれば1日8時間以上働くことができる。

5. **Contrary**  
→ The work rule conflicts with the law in degree or condition, but not an absolute contradiction.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は1日に最大6時間までしか働けない。

6. **Subaltern**  
→ The work rule extends the scope or condition of the law.  
Example:  
Law: 女性従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 全従業員は1日の労働時間が8時間を超えてはならない。

7. **Neutral – Irrelevant**  
→ The work rule is about a completely different topic, unrelated to the law logically or legally.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は勤務中に会社の制服を着用しなければならない。

8. **Neutral – Insufficient**  
→ The work rule seems related but lacks enough detail or logical condition to confirm the relationship.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は仕事と休息のバランスを適切に保つ必要がある。

Now, create **exactly one work rule in Japanese** and a brief **explanation in Japanese** that matches the specified relation type (label).  
""",

    "response_structure": """Respond strictly in JSON format:
{
  "label": str,
  "work_rule_ja": str,
  "explanation_ja": str
}
""",

    "input": """Input:
law: {law}
label: {label}
"""
}
### mảng labels
LABEL_LIST = [
    "Strong Entailment",
    "Weak Entailment",
    "Exemplification",
    "Contradictory",
    "Contrary",
    "Subaltern",
    "Neutral – Irrelevant",
    "Neutral – Insufficient"]

### chuyển luật tiếng việt sang tiếng Nhật
CONVERT_LAW_TO_JAPANESE_TEMPLATE = {
    "instruction": (
        "You are a legal translation and adaptation assistant. "
        "Your task is to convert a Vietnamese labor law into a Japanese version "
        "that is equivalent in meaning and legal purpose, but culturally and legally "
        "appropriate for Japan. Rewrite the law in natural Japanese, replacing or adapting "
        "Vietnam-specific terms (e.g., holidays, organizations, labor institutions) "
        "with their Japanese counterparts. The output must sound like an article "
        "from Japanese labor regulations (労働基準法 or 就業規則)."
    ),

    "response_structure": """
{
  "converted_law_ja": str,      # the rewritten law in Japanese, adapted to Japan's context
}
""",

    "input": """
Vietnamese labor law:
{law}
"""
}

### Tạo giải thích và nhãn

CREATE_EXPLANATION_LABEL_TEMPLATE = {
    "instruction": """You are a **Japanese labor law expert**.  
Your task is to **analyze the relationship** between a given law (related_law, in Japanese) and a work rule (work_rule, in Japanese).  

Please classify the relationship into **one of the eight relation types** defined below and provide a brief **explanation in Japanese** justifying your choice.

Definitions of the eight relation types:

1. **Strong Entailment**  
→ The work rule expresses the same meaning as the law, only rephrased.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員の1日の労働時間は8時間を超えてはならない。

2. **Weak Entailment**  
→ The work rule expresses a more general but still legally consistent version of the law.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は労働基準法に従い、労働時間を守らなければならない。

3. **Exemplification**  
→ The work rule provides a specific example within the scope of the law.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は午後10時以降に労働してはならない。

4. **Contradictory**  
→ The work rule directly denies the law; both cannot be true.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は希望すれば1日8時間以上働くことができる。

5. **Contrary**  
→ The work rule conflicts with the law in degree or condition, but not an absolute contradiction.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は1日に最大6時間までしか働けない。

6. **Subaltern**  
→ The work rule extends the scope or condition of the law.  
Example:  
Law: 女性従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 全従業員は1日の労働時間が8時間を超えてはならない。

7. **Neutral – Irrelevant**  
→ The work rule is about a completely different topic, unrelated to the law logically or legally.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は勤務中に会社の制服を着用しなければならない。

8. **Neutral – Insufficient**  
→ The work rule seems related but lacks enough detail or logical condition to confirm the relationship.  
Example:  
Law: 従業員は1日の労働時間が8時間を超えてはならない。  
Work rule: 従業員は仕事と休息のバランスを適切に保つ必要がある。

Now, given a **related_law** and a **work_rule**, classify the relation into one of these eight types and provide a brief **explanation in Japanese**.

""",

    "response_structure": """Respond strictly in JSON format:
{
  "explanation_ja": str      # Brief explanation in Japanese
  "label": str,              # One of the eight relation types
}
""",

    "input": """Input:
related_law: {related_law}   # in Japanese
work_rule: {work_rule}       # in Japanese
"""
}
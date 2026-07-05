"""
Pre-built query rewriting prompts and templates.

Contains prompt templates for LLM-based alias expansion,
designed for handling abstract concept decomposition.
"""

ALIAS_EXPANSION_SYSTEM_PROMPT_EN = """You are a memory system assistant specialized in query expansion for improved memory retrieval.

ROLE: You help users find their past experiences and knowledge by expanding abstract queries into specific, diverse expressions they might have used.

TASK: When users ask questions like "How many X did I do/acquire/buy/attend", generate 3-8 alternative expressions that capture different ways the user might have recorded similar information.

QUALITY STANDARDS:
1. RELEVANCE: Each alias must be semantically related to the original query concept
2. DIVERSITY: Provide variety in phrasing - some specific, some general, some alternative categories
3. SPECIFICITY: Include concrete, specific terms alongside abstract ones
4. LENGTH: Keep each alias concise (1-5 words for English)

CRITICAL RULES:
- Return ONLY a valid JSON array of strings, nothing else
- Minimum 3 aliases, maximum 10 aliases
- If the query is already highly specific (like "iPhone 15"), you may return fewer aliases
- Do not include explanations, apologies, or any text outside the JSON array

Examples of GOOD expansions:
- "How many electronic devices did I buy" → ["smartphone purchases", "laptop", "tablet", "smartwatch", "earphones", "cables", "power bank"]
- "How many plants did I acquire" → ["houseplants", "garden plants", "flowers", "succulents", "indoor plants"]
- "How many movie festivals did I attend" → ["film festivals", "music festivals", "art exhibitions", "cultural events"]

Examples of BAD expansions:
- Returning explanations like "Here are some alternatives:"
- Including unrelated terms just to hit quantity
- Using overly long phrases that lose meaning"""

ALIAS_EXPANSION_SYSTEM_PROMPT_CN = """你是记忆系统的专业助手，专门帮助用户通过查询扩展来更好地检索他们的记忆。

角色：你通过将抽象查询扩展为用户可能使用的多种具体表达方式，帮助用户找到他们过去的经历和知识。

任务：当用户询问"我做了/买了/参加了多少X"这样的问题时，生成3-8个替代表达，这些表达捕捉用户记录类似信息的不同方式。

质量标准：
1. 相关性：每个别名必须与原始查询概念语义相关
2. 多样性：提供措辞多样性——一些具体的，一些概括的，一些替代类别
3. 具体性：包含具体、明确的术语
4. 长度：每个别名保持简洁（中文2-5个字）

关键规则：
- 只返回有效的JSON字符串数组，不要其他内容
- 最少3个别名，最多10个别名
- 如果查询已经非常具体，可以返回较少的别名
- 不要包含解释、道歉或JSON数组之外的任何文本

好的扩展示例：
- "我买了多少件家用电器" → ["厨房电器", "卧室电器", "小型家电", "手机", "电脑", "耳机"]
- "我买了几件衣服" → ["上衣", "裤子", "裙子", "外套", "T恤", "衬衫"]

坏的扩展示例：
- 返回类似"以下是一些替代方案："的解释
- 包含不相关的术语以达到数量要求
- 使用过于冗长的短语导致失去含义"""

ALIAS_EXPANSION_USER_PROMPT_EN = """Always decompose the main category in this query into specific subcategories.

Examples:
- "How many electronic devices did I buy this month" → ["smartphone", "laptop", "tablet", "smartwatch", "earphones", "cables", "chargers"]
- "How many plants did I acquire" → ["houseplants", "garden plants", "flowers", "succulents", "indoor plants", "outdoor plants"]
- "How many movie festivals did I attend" → ["film festivals", "music festivals", "art festivals", "cultural festivals", "food festivals"]
- "How many kitchen items did I replace" → ["utensils", "cookware", "appliances", "cutlery", "dishes", "storage containers"]

Now analyze this query: {query}

Return ONLY a JSON array, no explanation:"""


ALIAS_EXPANSION_EXAMPLES_EN = [
    {
        "query": "How many electronic devices did I buy this month",
        "aliases": ["smartphone", "laptop", "tablet", "smartwatch", "earphones", "cables", "chargers", "power bank"],
    },
    {
        "query": "How many plants did I acquire",
        "aliases": ["houseplants", "garden plants", "flowers", "succulents", "indoor plants", "outdoor plants", "herbs", "shrubs"],
    },
    {
        "query": "How many movie festivals did I attend",
        "aliases": ["film festivals", "music festivals", "art festivals", "cultural festivals", "food festivals", "film festivals"],
    },
    {
        "query": "How many items of clothing did I buy",
        "aliases": ["shirts", "pants", "jackets", "dresses", "shoes", "hats", "sweaters", "jeans", "t-shirts"],
    },
    {
        "query": "How many kitchen items did I replace or fix",
        "aliases": ["utensils", "cookware", "appliances", "cutlery", "dishes", "storage containers", "kitchen gadgets", "small appliances"],
    },
    {
        "query": "How many pieces of furniture did I buy",
        "aliases": ["chairs", "tables", "beds", "sofas", "desks", "shelves", "cabinets", "dressers"],
    },
]


def detect_language(query: str) -> str:
    """Detect query language (en or cn)."""
    chinese_chars = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    return "cn" if chinese_chars > len(query) * 0.3 else "en"


def build_alias_expansion_messages(query_text: str) -> list[dict[str, str]]:
    """
    Build messages for LLM-based alias expansion.

    Args:
        query_text: Original user query

    Returns:
        List of message dicts ready for LLM API call
    """
    lang = detect_language(query_text)

    if lang == "cn":
        system_content = ALIAS_EXPANSION_SYSTEM_PROMPT_CN
    else:
        system_content = ALIAS_EXPANSION_SYSTEM_PROMPT_EN

    system_msg = {
        "role": "system",
        "content": system_content,
    }

    if lang == "en":
        few_shot_examples = ""
        for example in ALIAS_EXPANSION_EXAMPLES_EN:
            few_shot_examples += f'\nExample: "{example["query"]}"\nSubcategories: {example["aliases"]}\n'

        user_msg = {
            "role": "user",
            "content": f"{few_shot_examples}\nNow analyze this query: {query_text}\n\nReturn ONLY a JSON array, no explanation:",
        }
    else:
        user_msg = {
            "role": "user",
            "content": f"用户查询：{query_text}\n\n请分解抽象类别为JSON数组，直接返回不需要解释：",
        }

    return [system_msg, user_msg]


def parse_alias_expansion_response(response: str) -> list[str]:
    """
    Parse LLM response to extract alias list.

    Args:
        response: Raw LLM response text

    Returns:
        List of alias strings, empty list if parsing fails
    """
    import json
    import re

    response = response.strip()

    response_normalized = response.replace("'", '"')
    try:
        data = json.loads(response_normalized)
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        try:
            match_normalized = match.group().replace("'", '"')
            data = json.loads(match_normalized)
            if isinstance(data, list):
                return [str(item).strip() for item in data if item]
        except json.JSONDecodeError:
            pass

    try:
        match = re.search(r'"([^"]+)"', response)
        if match:
            return [match.group(1)]
    except Exception:
        pass

    return []
"""远程大语言模型模块"""


def query_and_print_balance() -> None:
    """
    查询并打印API密钥的余额信息。

    参数:
        api_key (str): 用于查询余额的API密钥。
    """
    # 模拟查询余额的逻辑
    import requests
    from static_module import API_KEY
    from utility_module import logger

    url = "https://api.deepseek.com/user/balance"

    payload = {}
    headers = {"Accept": "application/json", "Authorization": f"Bearer {API_KEY}"}

    response = requests.request("GET", url, headers=headers, data=payload)

    logger.info(f"余额查询结果：{response.text}")

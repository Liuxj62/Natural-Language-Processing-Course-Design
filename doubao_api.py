import http.client
import json


class DouBaoAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or "395f4c5d-1e5e-4178-8116-4d721d18642b"
        self.base_url = "ark.cn-beijing.volces.com"
        self.endpoint = "/api/v3/chat/completions"

    def chat(self, message, system_prompt="You are a helpful assistant."):
        """调用豆包API进行对话"""
        conn = http.client.HTTPSConnection(self.base_url)
        payload = json.dumps({
            "model": "doubao-seed-1-6-251015",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        })

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            conn.request("POST", self.endpoint, payload, headers)
            res = conn.getresponse()
            data = res.read()
            response_json = json.loads(data.decode("utf-8"))

            # 调试：打印完整响应
            print(f"API响应状态码: {res.status}")
            print(f"API响应内容: {response_json}")

            # 豆包API可能的响应格式1
            if 'choices' in response_json and len(response_json['choices']) > 0:
                return response_json['choices'][0]['message']['content']
            # 豆包API可能的响应格式2
            elif 'data' in response_json and 'choices' in response_json['data']:
                choices = response_json['data']['choices']
                if len(choices) > 0:
                    return choices[0]['message']['content']
            # 豆包API可能的响应格式3
            elif 'result' in response_json and 'message' in response_json['result']:
                return response_json['result']['message']['content']
            else:
                # 返回整个响应以便调试
                return f"无法解析响应格式。完整响应: {response_json}"

        except Exception as e:
            return f"调用豆包API时出错: {str(e)}"
        finally:
            conn.close()

    def translate(self, text, source_lang="zh", target_lang="en"):
        """使用豆包API进行翻译"""
        if source_lang == "zh" and target_lang == "en":
            prompt = f"请将以下中文翻译成英文: {text}"
        elif source_lang == "en" and target_lang == "zh":
            prompt = f"请将以下英文翻译成中文: {text}"
        else:
            return f"不支持{source_lang}到{target_lang}的翻译"

        return self.chat(prompt, "You are a professional translator.")
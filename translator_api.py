import re

class TranslatorAPI:
    def __init__(self):
        self.translations = {
            'zh-en': {
                "你好": "Hello",
                "谢谢": "Thank you",
                "再见": "Goodbye",
                "对不起": "Sorry",
                "早上好": "Good morning",
                "晚上好": "Good evening",
                "我爱你": "I love you",
                "很高兴见到你": "Nice to meet you",
                "你叫什么名字": "What's your name",
                "你来自哪里": "Where are you from",
                "我是一名学生": "I am a student",
                "今天天气很好": "The weather is nice today",
                "我喜欢学习": "I like studying",
                "这个很好吃": "This is delicious",
                "多少钱": "How much",
                "请问": "Excuse me",
                "不好意思": "Excuse me",
                "没关系": "It's okay",
                "没问题": "No problem",
                "春节": "Spring Festival",
                "生日快乐": "Happy birthday",
                "新年快乐": "Happy New Year",
                "恭喜发财": "Wish you prosperity",
                "身体健康": "Good health",
                "万事如意": "Everything goes well",
                "聊城大学": "Liaocheng University",
                "人工智能": "Artificial Intelligence",
                "机器学习": "Machine Learning",
                "深度学习": "Deep Learning",
                "自然语言处理": "Natural Language Processing"
            },
            'en-zh': {
                "Hello": "你好",
                "Thank you": "谢谢",
                "Goodbye": "再见",
                "Sorry": "对不起",
                "Good morning": "早上好",
                "Good evening": "晚上好",
                "I love you": "我爱你",
                "Nice to meet you": "很高兴见到你",
                "What's your name": "你叫什么名字",
                "Where are you from": "你来自哪里",
                "I am a student": "我是一名学生",
                "The weather is nice today": "今天天气很好",
                "I like studying": "我喜欢学习",
                "This is delicious": "这个很好吃",
                "How much": "多少钱",
                "Excuse me": "请问",
                "It's okay": "没关系",
                "No problem": "没问题",
                "Spring Festival": "春节",
                "Happy birthday": "生日快乐",
                "Happy New Year": "新年快乐",
                "Wish you prosperity": "恭喜发财",
                "Good health": "身体健康",
                "Everything goes well": "万事如意",
                "Liaocheng University": "聊城大学",
                "Artificial Intelligence": "人工智能",
                "Machine Learning": "机器学习",
                "Deep Learning": "深度学习",
                "Natural Language Processing": "自然语言处理"
            }
        }
        print("翻译API初始化完成")

    def detect_language(self, text):
        """检测语言"""
        has_chinese = any('一' <= char <= '鿿' for char in text)
        has_english = any('a' <= char.lower() <= 'z' for char in text)

        if has_chinese and not has_english:
            return 'zh'
        elif has_english and not has_chinese:
            return 'en'
        elif has_chinese and has_english:
            return 'mixed'
        else:
            return 'unknown'

    def translate(self, text, source_lang=None, target_lang=None):
        """翻译文本"""
        if not text.strip():
            return text

        # 自动检测语言
        if source_lang is None:
            source_lang = self.detect_language(text)

        # 设置目标语言
        if target_lang is None:
            target_lang = 'en' if source_lang == 'zh' else 'zh'

        # 查找完整匹配
        key = f"{source_lang}-{target_lang}"
        if key in self.translations:
            if text in self.translations[key]:
                return self.translations[key][text]

        # 尝试使用豆包API（如果可用）
        try:
            from nlp_deeplearn.doubao.doubao_api import DouBaoAPI
            doubao = DouBaoAPI()
            if source_lang == 'zh' and target_lang == 'en':
                prompt = f"请将以下中文翻译成英文：{text}"
            elif source_lang == 'en' and target_lang == 'zh':
                prompt = f"请将以下英文翻译成中文：{text}"
            else:
                return f"不支持{source_lang}到{target_lang}的翻译"

            result = doubao.chat(prompt)
            return result
        except:
            # 简单逐词翻译
            if source_lang == 'zh' and target_lang == 'en':
                return f"Translation: {text}"
            elif source_lang == 'en' and target_lang == 'zh':
                return f"翻译: {text}"
            else:
                return text

# 全局实例
translator = TranslatorAPI()

def translate_text(text, source_lang=None, target_lang=None):
    """翻译接口函数"""
    return translator.translate(text, source_lang, target_lang)
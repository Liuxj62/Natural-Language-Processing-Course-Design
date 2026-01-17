import re
import jieba
import math


class SentimentAnalyzerAPI:
    def __init__(self):
        # 扩展停用词表
        self.stopwords = set(
            ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
             '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '就', '也', '和', '与', '或',
             '及', '等', '等', '等等', '这样', '那样', '这种', '那种', '这些', '那些', '啊', '呀', '呢', '吗', '吧',
             '啦', '哇', '哦', '哈', '唉', '嗯', '呃', '呵', '哟', '诶', '个', '很', '都', '就', '也', '和', '与'])

        # 扩展积极情感词典
        self.positive_words = set([
            '好', '优秀', '满意', '喜欢', '爱', '开心', '高兴', '快乐', '幸福', '美',
            '漂亮', '精彩', '棒', '完美', '强大', '方便', '实用', '感谢', '赞',
            '推荐', '支持', '鼓励', '帮助', '关心', '温暖', '热情', '善良', '聪明',
            '努力', '成功', '进步', '发展', '和谐', '和平', '自由', '民主', '文明',
            '友好', '合作', '共赢', '创新', '创意', '独特', '特别', '出色', '优秀',
            '美好', '愉快', '兴奋', '激动', '喜悦', '欢乐', '欣慰', '舒适', '轻松',
            '安心', '放心', '信任', '佩服', '尊敬', '崇拜', '羡慕', '渴望', '期待',
            '乐观', '积极', '向上', '进取', '勤奋', '刻苦', '坚持', '勇敢', '坚强',
            '自信', '自豪', '骄傲', '荣耀', '光荣', '胜利', '成功', '成就', '业绩',
            '贡献', '价值', '意义', '重要', '关键', '核心', '主要', '必要', '必须',
            '喜欢', '爱', '爱好', '热爱', '热衷', '痴迷', '迷恋', '钟情', '心仪',
            'yyds', '绝绝子', '给力', '惊艳', '不错', '良好', '很棒', '很好', '美丽',
            '出色', '完美', '厉害', '牛', '优秀', '666', '不错', '可以', '行', '靠谱',
            '值得', '划算', '实惠', '便宜', '高效', '快速', '稳定', '可靠', '安全',
            '干净', '整洁', '新鲜', '美味', '好吃', '好喝', '好玩', '有趣', '有意思',
            '舒适', '舒服', '顺畅', '流畅', '简单', '容易', '方便', '便捷', '精彩',
            '优秀', '满意', '喜欢', '高兴', '开心', '快乐', '幸福', '美', '漂亮'
        ])

        # 扩展消极情感词典
        self.negative_words = set([
            '差', '坏', '不好', '讨厌', '恨', '生气', '愤怒', '难过', '悲伤', '丑',
            '糟糕', '失败', '烂', '差劲', '垃圾', '麻烦', '困难', '复杂', '批评',
            '反对', '抱怨', '失望', '绝望', '痛苦', '伤心', '难受', '郁闷', '烦恼',
            '压力', '紧张', '焦虑', '恐惧', '害怕', '担心', '怀疑', '否定', '拒绝',
            '冲突', '矛盾', '问题', '错误', '缺陷', '不足', '缺点', '弱点', '失败',
            '痛苦', '难受', '悲伤', '悲哀', '悲痛', '凄惨', '凄凉', '可怜', '可惜',
            '遗憾', '后悔', '懊悔', '愧疚', '自责', '羞愧', '羞耻',
            '尴尬', '窘迫', '狼狈', '恐慌', '惊慌', '惊恐', '惊吓', '恐惧', '害怕',
            '担忧', '忧虑', '忧郁', '抑郁', '沮丧', '消沉', '低落', '颓废', '疲惫',
            '疲倦', '劳累', '辛苦', '艰难', '困苦', '贫困', '贫穷', '缺乏', '不足',
            '讨厌', '厌恶', '反感', '憎恨', '仇恨', '愤恨', '怨恨', '恼火', '气愤',
            '很差', '糟糕', '差劲', '难看', '失望', '不开心', '生气', '愤怒', '垃圾',
            '烂', '麻烦', '不行', '不可以', '不行', '不好', '差评', '坑', '骗', '贵',
            '贵死', '贵得要命', '慢', '慢死', '卡', '卡顿', '卡死', '复杂', '难用',
            '难吃', '难喝', '难玩', '无聊', '无趣', '没意思', '不舒服', '不适', '痛',
            '疼', '痒', '热', '冷', '吵', '闹', '脏', '乱', '臭', '难闻', '难看'
        ])

        # 新增：中性词词典
        self.neutral_words = set([
            '一般般', '普通', '平常', '一般', '还行', '还可以', '凑合',
            '马马虎虎', '还好', '无所谓', '没感觉', '无感', '没什么',
            '差不多', '就这样', '还好吧', '就这样吧', '普通', '平常',
            '没什么特别', '没什么感觉', '没什么意思', '没特别感觉'
        ])

        # 程度副词权重
        self.intensity_words = {
            '极其': 3.0, '极度': 3.0, '非常': 2.5, '特别': 2.5, '十分': 2.0,
            '很': 2.0, '相当': 1.8, '比较': 1.5, '有点': 1.2, '稍微': 1.1,
            '略微': 1.1, '些微': 1.1, '不太': 0.8, '不怎么': 0.7, '不': -1.0,
            '没': -1.0, '没有': -1.0, '从未': -1.0, '绝不': -1.2, '完全不': -1.5,
            '太': 2.0, '超': 2.5, '超级': 2.5, '极其': 3.0, '格外': 2.0,
            '分外': 2.0, '异常': 2.0, '颇': 1.8, '挺': 1.8, '怪': 1.5,
            '极其': 3.0, '极度': 3.0, '异常': 2.5, '格外': 2.0, '分外': 2.0
        }

        # 扩展否定词集
        self.negative_flags = set(
            ['不', '没', '没有', '并非', '未', '无', '非', '勿', '莫', '别', '不要', '不用', '不必'])

        # 扩展否定表达式
        self.negative_phrases = set([
            '不开心', '不高兴', '不满意', '不喜欢', '不好', '不行',
            '不怎么样', '不感兴趣', '不关心', '无所谓', '不满意',
            '不喜欢', '不开心', '不高兴'
        ])

        # 添加自定义分词
        jieba.add_word('不开心', freq=1000)
        jieba.add_word('很不错', freq=1000)
        jieba.add_word('一般般', freq=1000)
        jieba.add_word('yyds', freq=1000)
        jieba.add_word('绝绝子', freq=1000)
        jieba.add_word('666', freq=1000)
        jieba.add_word('没感觉', freq=1000)
        jieba.add_word('无所谓', freq=1000)

        print(
            f"情感分析API初始化完成 - 积极词: {len(self.positive_words)}, 消极词: {len(self.negative_words)}, 中性词: {len(self.neutral_words)}")

    def preprocess_text(self, text):
        """优化预处理，保留必要的字符"""
        # 保留中文、英文、数字和常用标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：,.!?;:"\'\"\s]', '', text)
        # 去除多余空格
        text = ' '.join(text.split())
        return text.strip()

    def analyze(self, text, debug=False):
        """优化版情感分析逻辑"""
        # 1. 预处理
        if not isinstance(text, str):
            text = str(text)
        clean_text = self.preprocess_text(text)

        if debug:
            print(f"清洗后文本: {clean_text}")

        # 特殊处理：包含"一般般"的文本直接判为中性
        if '一般般' in clean_text or '没什么特别' in clean_text or '没特别感觉' in clean_text:
            words = jieba.lcut(clean_text, cut_all=False)
            words = [w for w in words if w not in self.stopwords and len(w.strip()) > 0]
            return {
                '情感倾向': '中性',
                '情感分数': 0.5,
                '清洗后文本': clean_text,
                '分词结果': words,
                '情感词数量': 0,
                '原始分数': {'positive': 0.0, 'negative': 0.0}
            }

        # 2. 分词
        words = jieba.lcut(clean_text, cut_all=False)
        words = [w for w in words if w not in self.stopwords and len(w.strip()) > 0]

        if debug:
            print(f"分词结果: {words}")

        # 3. 初始化
        positive_score = 0.0
        negative_score = 0.0
        emotion_words_count = 0

        # 4. 情感计算（优化版）
        i = 0
        while i < len(words):
            word = words[i]

            # 检查是否为中性词
            if word in self.neutral_words:
                # 中性词会降低情感强度
                positive_score *= 0.8
                negative_score *= 0.8
                if debug:
                    print(f"发现中性词: {word}，调整分数")
                i += 1
                continue

            # 检查组合词
            combined_found = False
            for length in range(3, 1, -1):
                if i + length <= len(words):
                    combined = ''.join(words[i:i + length])

                    # 检查是否为否定短语
                    if combined in self.negative_phrases:
                        if debug:
                            print(f"发现否定短语: {combined}")
                        negative_score += 1.0
                        emotion_words_count += 1
                        i += length
                        combined_found = True
                        break

                    # 检查是否为情感词
                    is_positive = combined in self.positive_words
                    is_negative = combined in self.negative_words

                    if is_positive or is_negative:
                        # 处理情感词
                        intensity = 1.0
                        negated = False

                        # 检查前一个词是否为否定词
                        if i > 0 and words[i - 1] in self.negative_flags:
                            negated = True

                        # 检查前一个词是否为程度词
                        if i > 0 and words[i - 1] in self.intensity_words:
                            intensity_weight = self.intensity_words[words[i - 1]]
                            if intensity_weight > 0:
                                intensity = abs(intensity_weight)

                        emotion_words_count += 1

                        if debug:
                            print(f"组合情感词: {combined}, 类型: {'积极' if is_positive else '消极'}, "
                                  f"程度: {intensity}, 否定: {negated}")

                        if is_positive:
                            if negated:
                                negative_score += intensity * 0.8
                            else:
                                positive_score += intensity
                        else:  # is_negative
                            if negated:
                                positive_score += intensity * 0.8
                            else:
                                negative_score += intensity

                        i += length
                        combined_found = True
                        break

            if not combined_found:
                # 处理单个词
                is_positive = word in self.positive_words
                is_negative = word in self.negative_words

                if is_positive or is_negative:
                    intensity = 1.0
                    negated = False

                    if i > 0 and words[i - 1] in self.negative_flags:
                        negated = True

                    if i > 0 and words[i - 1] in self.intensity_words:
                        intensity_weight = self.intensity_words[words[i - 1]]
                        if intensity_weight > 0:
                            intensity = abs(intensity_weight)

                    emotion_words_count += 1

                    if debug:
                        print(f"单个情感词: {word}, 类型: {'积极' if is_positive else '消极'}, "
                              f"程度: {intensity}, 否定: {negated}")

                    if is_positive:
                        if negated:
                            negative_score += intensity * 0.8
                        else:
                            positive_score += intensity
                    else:  # is_negative
                        if negated:
                            positive_score += intensity * 0.8
                        else:
                            negative_score += intensity

                i += 1

        if debug:
            print(f"积极分数: {positive_score}, 消极分数: {negative_score}, 情感词数: {emotion_words_count}")

        # 5. 改进的分数归一化
        if emotion_words_count == 0:
            # 没有情感词，检查是否有其他特征
            normalized = 0.5
        else:
            # 使用更平滑的归一化方法
            diff = positive_score - negative_score
            total = positive_score + negative_score

            # 如果总分为0（理论上不会发生）
            if total == 0:
                normalized = 0.5
            else:
                # 使用tanh函数进行平滑归一化，产生更多中间值
                normalized = 0.5 + 0.5 * math.tanh(diff / (total + 1))

            # 确保在[0,1]范围内
            normalized = max(0.0, min(1.0, normalized))

        # 6. 情感判定（调整阈值）
        if normalized > 0.65:  # 适当放宽积极阈值
            sentiment_cn = '积极'
            score = round(normalized, 3)  # 保留3位小数
        elif normalized < 0.35:  # 适当放宽消极阈值
            sentiment_cn = '消极'
            score = round(normalized, 3)
        else:
            sentiment_cn = '中性'
            score = round(normalized, 3)

        # 返回兼容两种系统的字段
        return {
            # 用于智能问答系统的字段
            'sentiment_cn': sentiment_cn,
            'score': score,

            # 用于原始测试代码的字段
            '情感倾向': sentiment_cn,
            '情感分数': score,

            # 其他调试信息
            '清洗后文本': clean_text,
            '分词结果': words,
            '情感词数量': emotion_words_count,
            '原始分数': {
                'positive': round(positive_score, 2),
                'negative': round(negative_score, 2)
            }
        }


# 全局实例
sentiment_analyzer = SentimentAnalyzerAPI()


def analyze_sentiment(text, debug=False):
    """情感分析接口函数"""
    return sentiment_analyzer.analyze(text, debug)


# 测试函数
if __name__ == "__main__":
    print("=" * 70)
    print("情感分析API - 优化版测试")
    print("=" * 70)

    # 重点测试有问题的案例
    problem_cases = [
        ("一般般，没有什么特别的感觉", "预期: 中性"),
        ("我不开心，但也不难过", "预期: 中性"),
        ("不开心", "预期: 消极"),
        ("不是不开心", "预期: 积极（双重否定）"),
        ("一般般", "预期: 中性"),
        ("还不错", "预期: 积极"),
        ("很差", "预期: 消极"),
    ]

    print("\n" + "=" * 70)
    print("问题案例测试")
    print("=" * 70)

    for text, expected in problem_cases:
        result = analyze_sentiment(text, debug=True)
        print(f"\n文本: {text}")
        print(f"预期: {expected}")
        print(f"结果: {result['情感倾向']}, 分数: {result['情感分数']}")
        print(f"分词: {result['分词结果']}")
        print(f"情感词数: {result['情感词数量']}")
        print(f"原始分数: 积极={result['原始分数']['positive']}, 消极={result['原始分数']['negative']}")
        print("-" * 50)

    # 原始测试集
    print("\n" + "=" * 70)
    print("原始测试集")
    print("=" * 70)

    original_tests = [
        "这个产品非常好用，我非常满意",
        "服务态度很差，体验非常糟糕",
        "一般般，没有什么特别的感受",
        "我不开心",
        "我喜欢打篮球",
        "这个电影真难看",
        "今天天气不错，心情很好",
        "服务态度很差，体验非常糟糕",
        "一般般，没有什么特别的感受",
        "我不开心，但也不难过",
        "他虽然不优秀，但也不算差",
        "这部电影简直yyds，太精彩了",
        "一点都不好，非常失望",
        "不得不说，这确实很棒",
    ]

    for test_text in original_tests:
        result = analyze_sentiment(test_text, debug=False)
        print(f"\n文本: {test_text[:40]}...")
        print(f"情感分析: {result['情感倾向']}, 分数: {result['情感分数']}")
        print(f"分词: {result['分词结果'][:10]}...")
        print("-" * 50)

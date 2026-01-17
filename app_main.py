"""
多功能智能问答系统 - 主应用
集成豆包AI、文本分类、情感分析、机器翻译
"""
import os
import sys
import json
import importlib.util
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# 设置项目根目录
# 获取当前文件的绝对路径，然后计算项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

# 如果是运行在nlp_deeplearn目录下
if os.path.basename(project_root) == 'nlp_deeplearn':
    project_root = os.path.dirname(project_root)  # 回到上一层

print(f"项目根目录: {project_root}")
print(f"当前目录: {current_dir}")

# 添加必要的路径到sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'nlp_deeplearn'))

# 设置编码
if sys.platform == 'win32':
    try:
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

print("=" * 60)
print("多功能智能问答系统 - 初始化")
print("=" * 60)

# 导入各功能模块
services = {}

# 1. 豆包API - 直接导入，使用绝对路径
print("[1/4] 正在加载豆包API...")
try:
    # 直接使用绝对路径导入
    doubao_path = os.path.join(project_root, 'nlp_deeplearn', 'doubao', 'doubao_api.py')
    if os.path.exists(doubao_path):
        spec = importlib.util.spec_from_file_location("doubao_api", doubao_path)
        doubao_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(doubao_module)
        services['doubao'] = doubao_module.DouBaoAPI()
        print("  ✓ 豆包API加载成功")
    else:
        print(f"  ✗ 豆包API文件不存在: {doubao_path}")
        services['doubao'] = None
except Exception as e:
    print(f"  ✗ 豆包API加载失败: {e}")
    services['doubao'] = None

# 2. 文本分类 - 直接从code目录导入
print("[2/4] 正在加载文本分类服务...")
try:
    # 检查code目录位置
    code_dir = os.path.join(project_root, 'nlp_deeplearn', 'code')
    text_classifier_path = os.path.join(code_dir, 'text_classifier_api.py')

    if os.path.exists(text_classifier_path):
        spec = importlib.util.spec_from_file_location("text_classifier_api", text_classifier_path)
        text_classifier_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(text_classifier_module)

        # 创建实例
        if hasattr(text_classifier_module, 'TextClassifierAPI'):
            text_classifier_instance = text_classifier_module.TextClassifierAPI()
            services['classify'] = text_classifier_instance.predict
            print("  ✓ 文本分类服务加载成功")
        else:
            print("  ✗ 文本分类API类不存在")
            services['classify'] = None
    else:
        print(f"  ✗ 文本分类文件不存在: {text_classifier_path}")
        services['classify'] = None
except Exception as e:
    print(f"  ✗ 文本分类服务加载失败: {e}")
    services['classify'] = None

# 3. 情感分析 - 直接从code目录导入
print("[3/4] 正在加载情感分析服务...")
try:
    code_dir = os.path.join(project_root, 'nlp_deeplearn', 'code')
    sentiment_path = os.path.join(code_dir, 'sentiment_analyzer_api.py')

    if os.path.exists(sentiment_path):
        spec = importlib.util.spec_from_file_location("sentiment_analyzer_api", sentiment_path)
        sentiment_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sentiment_module)

        # 创建实例
        if hasattr(sentiment_module, 'SentimentAnalyzerAPI'):
            # 修复：添加sys导入到模块
            import sys as sys_module

            if not hasattr(sentiment_module, 'sys'):
                sentiment_module.sys = sys_module

            sentiment_instance = sentiment_module.SentimentAnalyzerAPI()
            services['sentiment'] = sentiment_instance.analyze
            print("  ✓ 情感分析服务加载成功")
        else:
            print("  ✗ 情感分析API类不存在")
            services['sentiment'] = None
    else:
        print(f"  ✗ 情感分析文件不存在: {sentiment_path}")
        services['sentiment'] = None
except Exception as e:
    print(f"  ✗ 情感分析服务加载失败: {e}")
    services['sentiment'] = None

# 4. 机器翻译 - 直接从code目录导入
print("[4/4] 正在加载机器翻译服务...")
try:
    code_dir = os.path.join(project_root, 'nlp_deeplearn', 'code')
    translator_path = os.path.join(code_dir, 'translator_api.py')

    if os.path.exists(translator_path):
        spec = importlib.util.spec_from_file_location("translator_api", translator_path)
        translator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(translator_module)

        # 创建实例
        if hasattr(translator_module, 'TranslatorAPI'):
            translator_instance = translator_module.TranslatorAPI()
            services['translate'] = translator_instance.translate
            print("  ✓ 机器翻译服务加载成功")
        else:
            print("  ✗ 机器翻译API类不存在")
            services['translate'] = None
    else:
        print(f"  ✗ 机器翻译文件不存在: {translator_path}")
        services['translate'] = None
except Exception as e:
    print(f"  ✗ 机器翻译服务加载失败: {e}")
    services['translate'] = None

print("=" * 60)


# 备用服务函数
def fallback_classify(text):
    """备用文本分类"""
    return {
        'category': '其他',
        'confidence': 0.5,
        'all_categories': ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    }


def fallback_sentiment(text):
    """备用情感分析"""
    return {
        'sentiment': 'neutral',
        'sentiment_cn': '中性',
        'score': 0.5
    }


def fallback_translate(text, source_lang=None, target_lang=None):
    """备用翻译"""
    if source_lang == 'zh' and target_lang == 'en':
        return f"Translation: {text}"
    elif source_lang == 'en' and target_lang == 'zh':
        return f"翻译: {text}"
    else:
        return text


def fallback_doubao(text):
    """备用豆包对话"""
    return f"豆包AI：我收到了你的消息：{text}。作为AI助手，我还在学习中。"


# 创建Flask应用
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'multifunctional-ai-system-secret-2024'


# 对话历史管理
class ConversationManager:
    def __init__(self):
        self.history_file = 'conversation_data/history.json'
        self.max_history = 100
        self.history = []
        self.load_history()

    def load_history(self):
        """加载历史记录"""
        try:
            os.makedirs('conversation_data', exist_ok=True)
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"已加载 {len(self.history)} 条历史记录")
        except Exception as e:
            print(f"加载历史记录失败: {e}")
            self.history = []

    def save_history(self):
        """保存历史记录"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history[-self.max_history:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {e}")

    def add_message(self, user_msg, bot_msg, mode):
        """添加消息"""
        record = {
            'id': len(self.history) + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user': user_msg,
            'bot': bot_msg,
            'mode': mode
        }
        self.history.append(record)
        self.save_history()
        return record

    def get_history(self, limit=20):
        """获取历史记录"""
        return self.history[-limit:]

    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.save_history()
        return True


# 初始化对话管理器
conversation_manager = ConversationManager()


# 路由
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/modes', methods=['GET'])
def get_modes():
    """获取可用模式"""
    modes = [
        {
            'id': 'doubao',
            'name': '豆包智能对话',
            'icon': '🤖',
            'desc': '使用豆包AI进行智能对话',
            'enabled': services['doubao'] is not None
        },
        {
            'id': 'classify',
            'name': '文本分类',
            'icon': '🏷️',
            'desc': '对新闻文本进行分类',
            'enabled': services['classify'] is not None
        },
        {
            'id': 'sentiment',
            'name': '情感分析',
            'icon': '❤️',
            'desc': '分析文本的情感倾向',
            'enabled': services['sentiment'] is not None
        },
        {
            'id': 'translate_zh_en',
            'name': '中译英',
            'icon': '🌐',
            'desc': '将中文翻译成英文',
            'enabled': services['translate'] is not None
        },
        {
            'id': 'translate_en_zh',
            'name': '英译中',
            'icon': '🔤',
            'desc': '将英文翻译成中文',
            'enabled': services['translate'] is not None
        },
        {
            'id': 'comprehensive',
            'name': '综合分析',
            'icon': '🔍',
            'desc': '多维度分析文本内容',
            'enabled': True
        }
    ]
    return jsonify({'modes': modes})


@app.route('/api/process', methods=['POST'])
def process():
    """处理请求"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        mode = data.get('mode', 'doubao')

        if not text:
            return jsonify({'error': '请输入内容'})

        result = ''
        mode_name = ''

        if mode == 'doubao':
            mode_name = '豆包智能对话'
            if services['doubao']:
                try:
                    result = services['doubao'].chat(text)
                except Exception as e:
                    print(f"豆包API调用失败: {e}")
                    result = fallback_doubao(text)
            else:
                result = fallback_doubao(text)
                mode_name += '（备用）'

        elif mode == 'classify':
            mode_name = '文本分类'
            if services['classify']:
                try:
                    classification = services['classify'](text)
                    if isinstance(classification, dict):
                        result = f"📊 分类结果: {classification.get('category', '未知')}\n"
                        result += f"📈 置信度: {classification.get('confidence', 0):.2%}"
                    else:
                        result = str(classification)
                except Exception as e:
                    print(f"文本分类错误: {e}")
                    classification = fallback_classify(text)
                    result = f"📊 分类结果: {classification['category']}（备用）\n"
                    result += f"📈 置信度: {classification['confidence']:.2%}"
            else:
                classification = fallback_classify(text)
                result = f"📊 分类结果: {classification['category']}（备用）\n"
                result += f"📈 置信度: {classification['confidence']:.2%}"

        elif mode == 'sentiment':
            mode_name = '情感分析'
            if services['sentiment']:
                try:
                    sentiment = services['sentiment'](text)
                    if isinstance(sentiment, dict):
                        result = f"❤️ 情感倾向: {sentiment.get('sentiment_cn', '未知')}\n"
                        result += f"📊 情感分数: {sentiment.get('score', 0):.2f}"
                    else:
                        result = str(sentiment)
                except Exception as e:
                    print(f"情感分析错误: {e}")
                    sentiment = fallback_sentiment(text)
                    result = f"❤️ 情感倾向: {sentiment['sentiment_cn']}（备用）\n"
                    result += f"📊 情感分数: {sentiment['score']:.2f}"
            else:
                sentiment = fallback_sentiment(text)
                result = f"❤️ 情感倾向: {sentiment['sentiment_cn']}（备用）\n"
                result += f"📊 情感分数: {sentiment['score']:.2f}"

        elif mode == 'translate_zh_en':
            mode_name = '中译英'
            if services['translate']:
                try:
                    result = services['translate'](text, 'zh', 'en')
                except Exception as e:
                    print(f"翻译错误: {e}")
                    result = fallback_translate(text, 'zh', 'en')
            else:
                result = fallback_translate(text, 'zh', 'en')

        elif mode == 'translate_en_zh':
            mode_name = '英译中'
            if services['translate']:
                try:
                    result = services['translate'](text, 'en', 'zh')
                except Exception as e:
                    print(f"翻译错误: {e}")
                    result = fallback_translate(text, 'en', 'zh')
            else:
                result = fallback_translate(text, 'en', 'zh')

        elif mode == 'comprehensive':
            mode_name = '综合分析'
            result = comprehensive_analysis(text)

        else:
            return jsonify({'error': f'未知模式: {mode}'})

        # 保存到历史记录
        timestamp = datetime.now().strftime("%H:%M")
        conversation_manager.add_message(text, result, mode_name)

        return jsonify({
            'success': True,
            'result': result,
            'mode': mode_name,
            'timestamp': timestamp
        })

    except Exception as e:
        print(f"处理请求错误: {e}")
        return jsonify({'error': f'处理失败: {str(e)}'})


def comprehensive_analysis(text):
    """综合分析"""
    results = []

    # 1. 基本信息
    char_count = len(text)
    word_count = len(text.split())
    results.append(f"📝 文本统计: {char_count}字符, {word_count}词")

    # 2. 语言检测
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    has_english = any('a' <= char.lower() <= 'z' for char in text)

    if has_chinese and has_english:
        results.append("🌐 语言检测: 中英文混合")
    elif has_chinese:
        results.append("🌐 语言检测: 中文")
    elif has_english:
        results.append("🌐 语言检测: 英文")
    else:
        results.append("🌐 语言检测: 其他语言")

    # 3. 情感分析
    try:
        if services['sentiment']:
            sentiment = services['sentiment'](text)
            if isinstance(sentiment, dict):
                results.append(
                    f"📊 情感分析: {sentiment.get('sentiment_cn', '未知')} (分数: {sentiment.get('score', 0):.2f})")
        else:
            sentiment = fallback_sentiment(text)
            results.append(f"📊 情感分析: {sentiment['sentiment_cn']} (备用)")
    except Exception as e:
        results.append(f"📊 情感分析: 服务异常")

    # 4. 文本分类
    if len(text) > 10:
        try:
            if services['classify']:
                classification = services['classify'](text)
                if isinstance(classification, dict):
                    results.append(f"🏷️ 文本分类: {classification.get('category', '未知')}")
            else:
                classification = fallback_classify(text)
                results.append(f"🏷️ 文本分类: {classification['category']} (备用)")
        except Exception as e:
            results.append(f"🏷️ 文本分类: 服务异常")
    else:
        results.append("🏷️ 文本分类: 文本过短")

    # 5. 关键词提取
    keywords = []
    key_phrases = {
        '春节': ['春节', 'Spring Festival', 'Chinese New Year'],
        '教育': ['大学', '学校', '学生', '老师', '教育'],
        '学习': ['学习', 'study', 'learn', '课程'],
        '工作': ['工作', '上班', '职场', 'job', 'work'],
        '天气': ['天气', '气候', 'weather'],
        '科技': ['科技', '技术', '互联网', 'AI', '人工智能'],
        '体育': ['体育', '运动', '比赛', 'sports'],
        '健康': ['健康', '医疗', '医生', '医院']
    }

    for category, phrases in key_phrases.items():
        for phrase in phrases:
            if phrase in text.lower():
                keywords.append(category)
                break

    if keywords:
        unique_keywords = list(set(keywords))
        results.append(f"🔑 关键词: {', '.join(unique_keywords)}")

    # 6. 智能总结
    if len(text) > 20:
        try:
            if services['doubao']:
                summary = services['doubao'].chat(f"请用一句话总结这段话: {text}")
                results.append(f"💡 智能总结: {summary}")
            else:
                results.append(f"💡 智能总结: 使用备用总结")
                # 简单总结
                if len(text) > 50:
                    summary = text[:50] + "..."
                else:
                    summary = text
                results.append(f"💡 摘要: {summary}")
        except Exception as e:
            results.append(f"💡 智能总结: 服务异常")
    else:
        results.append("💡 智能总结: 文本过短")

    return "\n\n".join(results)


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取历史记录"""
    limit = min(int(request.args.get('limit', 20)), 100)
    history = conversation_manager.get_history(limit)
    return jsonify({'history': history})


@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """清空历史记录"""
    conversation_manager.clear_history()
    return jsonify({'success': True})


@app.route('/api/status', methods=['GET'])
def status():
    """系统状态"""
    return jsonify({
        'services': {
            'doubao': services['doubao'] is not None,
            'classify': services['classify'] is not None,
            'sentiment': services['sentiment'] is not None,
            'translate': services['translate'] is not None
        },
        'timestamp': datetime.now().isoformat(),
        'total_conversations': len(conversation_manager.history),
        'system': '多功能智能问答系统',
        'version': '2.0.0'
    })


@app.route('/api/test', methods=['GET'])
def test():
    """测试接口"""
    return jsonify({
        'message': '系统运行正常',
        'timestamp': datetime.now().isoformat(),
        'services': list(services.keys())
    })


# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '资源未找到'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    # 确保目录存在
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('conversation_data', exist_ok=True)

    print("=" * 60)
    print("多功能智能问答系统启动成功！")
    print("=" * 60)
    print("服务状态:")
    print(f"  豆包API: {'✓ 已启用' if services['doubao'] else '✗ 未启用'}")
    print(f"  文本分类: {'✓ 已启用' if services['classify'] else '✗ 未启用'}")
    print(f"  情感分析: {'✓ 已启用' if services['sentiment'] else '✗ 未启用'}")
    print(f"  机器翻译: {'✓ 已启用' if services['translate'] else '✗ 未启用'}")
    print("\n访问地址:")
    print("  http://127.0.0.1:5000")
    print("  http://localhost:5000")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
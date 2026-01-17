"""
多功能智能问答系统 - 稳定启动版本
避免复杂的编码设置
"""
import os
import sys

# 简单的编码设置，只处理基本输出
def setup_simple():
    """简单的初始化设置"""
    print("正在启动多功能智能问答系统...")
    print("=" * 60)

    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('conversation_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("目录结构检查完成")
    return True

def main():
    """主函数"""
    # 设置控制台标题（Windows）
    if sys.platform == 'win32':
        os.system('title 多功能智能问答系统')

    # 简单的初始化
    if not setup_simple():
        print("初始化失败")
        return

    # 导入并运行主应用
    try:
        from app_main import app
        print("系统启动成功！")
        print("访问地址: http://127.0.0.1:5000")
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)

        # 运行应用，不使用debug模式
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保app_main.py文件存在")
        input("按回车键退出...")
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
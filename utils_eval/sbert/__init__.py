import os

# 获取当前文件(__init__.py)所在的目录
package_dir = os.path.dirname(os.path.abspath(__file__))

# 基于包目录设置模型路径
models_dir = os.path.join(package_dir, 'models')
os.makedirs(models_dir, exist_ok=True)  # 确保目录存在

# 设置环境变量
os.environ['SENTENCE_TRANSFORMERS_HOME'] = models_dir
os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'
os.environ['TFHUB_CACHE_DIR'] = models_dir

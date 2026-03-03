"""
下载Kaggle Toxic Comment Classification数据集
需要先在 ~/.kaggle/kaggle.json 配置API密钥
"""
import os
import zipfile

# 检查kaggle配置
kaggle_config = os.path.expanduser('~/.kaggle/kaggle.json')
if not os.path.exists(kaggle_config):
    print("⚠️  请先配置Kaggle API密钥")
    print("1. 访问 https://www.kaggle.com/account 获取API Token")
    print("2. 将kaggle.json放到 ~/.kaggle/ 目录")
    print("3. 运行: chmod 600 ~/.kaggle/kaggle.json")
else:
    print("✅ Kaggle配置已存在")
    
# 尝试下载
try:
    import kaggle
    print("📥 下载数据集...")
    os.system('kaggle competitions download -c jigsaw-toxic-comment-classification-challenge')
    
    # 解压
    if os.path.exists('jigsaw-toxic-comment-classification-challenge.zip'):
        print("📦 解压数据集...")
        with zipfile.ZipFile('jigsaw-toxic-comment-classification-challenge.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
        print("✅ 数据集准备完成")
except Exception as e:
    print(f"❌ 错误: {e}")

import sys
import sklearn

print("=" * 30)
print("--- 环境诊断信息 ---")
print(f"Python 解释器路径: {sys.executable}")
print(f"scikit-learn 版本: {sklearn.__version__}")
print(f"scikit-learn 库路径: {sklearn.__file__}")
print("=" * 30)
print()
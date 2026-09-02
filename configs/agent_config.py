"""人工运行时只需要修改这个文件（API Key 和模型名除外）。"""

# 默认任务
DATA_FILE = "test/test.csv"
PROJECT_BACKGROUND = """
这是一个多元线性回归项目。目标是在保持良好泛化能力的同时，
选择统计显著、共线性可控并且便于解释的变量组合。
""".strip()
USER_QUERY = "预测 target"
TARGET_VARIABLE = "target"

# 候选变量池。None 表示使用目标列之外的全部字段。
SELECTED_VARIABLES = None
USE_LLM = True

# ReAct 探索约束：需要两个完成全部统计检查的候选。
NUM_COMPLETED_CANDIDATES = 2
MAX_TOTAL_ATTEMPTS = 8
MAX_REACT_TOOL_CALLS = 40

# 统计诊断阈值只用于提示 Agent，不再作为固定 MSE 门槛。
P_VALUE_THRESHOLD = 0.05
VIF_WARNING_THRESHOLD = 10.0

# 数据切分：默认 60% 训练、20% 验证、20% 最终测试。
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25
RANDOM_STATE = 42
MIN_USABLE_ROWS = 10

# 数据清洗和特征编码。
NUMERIC_IMPUTATION = "median"  # "median" 或 "mean"
CATEGORICAL_IMPUTATION = "most_frequent"  # "most_frequent" 或 "constant"
MISSING_CATEGORY_TOKEN = "__missing__"
ONE_HOT_DROP_FIRST = True

# 最终图片输出目录；相对路径以项目根目录为基准。
OUTPUT_DIR = "outputs"

# LLM 行为（模型名和 API Key 仍在 configs/setting.py）。
LLM_TEMPERATURE = 0.1
DEEPSEEK_THINKING_MODE = "enabled"  # "enabled" 或 "disabled"
DEEPSEEK_REASONING_EFFORT = "high"  # "low"、"high" 或 "max"
LLM_TIMEOUT_SECONDS = 120.0

"""人工运行时只需要修改这个文件（API Key 和模型名除外）。"""

# ---------------------------------------------------------------------------
# 1. 默认运行任务
# ---------------------------------------------------------------------------
# 可以填写绝对路径，或填写相对于执行命令所在目录的路径。
DATA_FILE = "test/test.csv"

# 自然语言需求。USE_LLM=False 时主要用于记录和目标列名推断。
USER_QUERY = "预测 target"

# 明确知道列名时建议填写；设为 None 时由 LLM 或规则推断。
TARGET_VARIABLE = "target"

# 设为 None 时使用除目标列外的全部字段；也可以显式填写列表。
SELECTED_VARIABLES = None

# 是否让 Gemini 参与变量选择。False 时整个回归流程可离线运行。
USE_LLM = True


# ---------------------------------------------------------------------------
# 2. 硬性验收和重试
# ---------------------------------------------------------------------------
MSE_THRESHOLD = 0.5
MAX_ITERATIONS = 3

# MSE 通过后是否继续删除统计上不显著的变量。
ENABLE_FEATURE_PRUNING = True
P_VALUE_THRESHOLD = 0.05
MIN_SELECTED_VARIABLES = 1


# ---------------------------------------------------------------------------
# 3. 数据切分
# ---------------------------------------------------------------------------
# 先留出 TEST_SIZE 作为最终测试集，再从剩余数据中按
# VALIDATION_SIZE_WITHIN_DEVELOPMENT 留出验证集。
# 默认结果为 60% 训练、20% 验证、20% 测试。
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25
RANDOM_STATE = 42
MIN_USABLE_ROWS = 10


# ---------------------------------------------------------------------------
# 4. 数据清洗和特征编码
# ---------------------------------------------------------------------------
# NUMERIC_IMPUTATION 支持 "median" 或 "mean"。
NUMERIC_IMPUTATION = "median"

# CATEGORICAL_IMPUTATION 支持 "most_frequent" 或 "constant"。
CATEGORICAL_IMPUTATION = "most_frequent"
MISSING_CATEGORY_TOKEN = "__missing__"

# True 会为每个分类变量少生成一个虚拟列，避免完全共线性。
ONE_HOT_DROP_FIRST = True


# ---------------------------------------------------------------------------
# 5. LLM 行为（模型名和 API Key 仍在 configs/setting.py）
# ---------------------------------------------------------------------------
LLM_TEMPERATURE = 0.1

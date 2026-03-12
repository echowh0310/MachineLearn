import json
class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        # 将混淆矩阵部分单独处理
        def format_confusion(matrix):
            # 将每一行转换为字符串，并添加缩进
            formatted_rows = []
            for row in matrix:
                # 将数字转换为固定宽度的字符串，使其对齐
                formatted_row = ", ".join(f"{num:>4}" for num in row)
                formatted_rows.append(f"[ {formatted_row} ]")
            return "[\n      " + ",\n      ".join(formatted_rows) + "\n    ]"

        if isinstance(obj, dict):
            # 递归处理字典
            # 过滤特征重要性字段
            filtered_obj = {k: v for k, v in obj.items() if k != 'feature_importance'}
            items = []
            for key, value in filtered_obj.items():
                if key == 'confusion':
                    # 对混淆矩阵应用特殊格式
                    formatted_value = format_confusion(value)
                    items.append(f'    "{key}": {formatted_value}')
                else:
                    # 对其他值递归编码
                    encoded_value = self.encode(value)
                    items.append(f'    "{key}": {encoded_value}')
            return "{\n  " + ",\n  ".join(items) + "\n}"
        elif isinstance(obj, list):
            # 如果是列表，直接使用默认编码器
            return super().encode(obj)
        else:
            # 对于基本类型，使用默认编码器
            return super().encode(obj)
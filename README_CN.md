# 中文情感守卫 (ChineseSentimentGuard, CSG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 概述

中文情感守卫是一个专为中文和英文文本设计的全面情感分析系统。它使用先进的深度学习模型提供准确的情感分类，同时确保数据隐私和安全。

![中文情感守卫](project/generated-icon.png)

## ✨ 特点

- **双模型分析**：可选择CNN和LSTM模型进行情感分析
- **双语支持**：同时分析中文和英文文本
- **隐私保护**：内置数据加密和匿名化功能
- **用户管理**：基于角色的访问控制（管理员/普通用户）
- **数据可视化**：全面的数据可视化和分析历史记录
- **双界面**：同时提供基于Web的Flask界面和Streamlit仪表板
- **API访问**：提供RESTful API以便与其他系统集成

## 🚀 开始使用

### 前提条件

- Python 3.8或更高版本
- pip或conda包管理工具

### 安装

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/ChineseSentimentGuard.git
   cd ChineseSentimentGuard
   ```

2. 创建并激活虚拟环境（可选但推荐）：
   ```bash
   python -m venv venv
   # Windows系统
   venv\Scripts\activate
   # macOS/Linux系统
   source venv/bin/activate
   ```

3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ 使用方法

### Flask Web界面

启动Flask Web应用：

```bash
python project/flask_app.py
```

在浏览器中访问 http://localhost:5000

### Streamlit仪表板

启动Streamlit仪表板：

```bash
streamlit run project/app.py
```

在浏览器中访问 http://localhost:8501

### 默认登录凭证

- **管理员用户**：
  - 用户名：admin
  - 密码：hello
- **普通用户**：
  - 用户名：user
  - 密码：hello

## 📊 功能详情

### 情感分析

系统使用两种深度学习模型进行情感分析：

1. **CNN模型**：适用于较短文本的快速高效分析
2. **LSTM模型**：更适合捕捉文本中的长距离依赖关系

情感分为三类：
- 消极 (Negative)
- 中性 (Neutral)
- 积极 (Positive)

### 数据隐私

系统实现了多种隐私保护功能：

- **数据加密**：使用AES-256加密敏感文本
- **文本匿名化**：可自动移除个人标识符
- **安全日志**：隐私感知的日志系统

### 管理员功能

管理员可以访问额外功能：

- **用户管理**：查看和管理用户账户
- **系统统计**：监控系统使用情况和性能
- **模型管理**：训练、评估和管理情感模型
- **数据可视化**：高级分析和可视化工具

## 🔧 API参考

系统提供用于情感分析的RESTful API：

### 分析情感

```
POST /api/analyze
```

**请求体：**
```json
{
  "text": "这个产品非常好用，我很喜欢！",
  "model_type": "CNN"
}
```

**响应：**
```json
{
  "text": "这个产品非常好用，我很喜欢！",
  "preprocessed_text": "产品 好用 喜欢",
  "sentiment": "积极",
  "confidence": 0.92,
  "model": "CNN",
  "prediction": [0.05, 0.03, 0.92]
}
```

## 📁 项目结构

```
project/
├── .config/               # 配置文件
├── .streamlit/           # Streamlit配置
├── pages/                # Streamlit页面
│   ├── admin.py          # 管理员仪表板
│   └── user.py           # 用户仪表板
├── templates/            # Flask HTML模板
│   ├── admin.html        # 管理员界面
│   ├── analyze.html      # 分析页面
│   ├── base.html         # 基础模板
│   ├── dashboard.html    # 主仪表板
│   ├── history.html      # 分析历史
│   └── login.html        # 登录页面
├── app.py                # Streamlit应用
├── flask_app.py          # Flask Web应用
├── models.py             # 情感分析模型
├── preprocessing.py      # 文本预处理工具
├── data_security.py      # 数据加密和隐私
├── utils.py              # 实用函数
├── visualization.py      # 数据可视化
└── requirements.txt      # 项目依赖
```

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

1. Fork仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加一些很棒的功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详情请参阅LICENSE文件。

## 📞 联系方式

您的姓名 - your.email@example.com

项目链接：[https://github.com/yourusername/ChineseSentimentGuard](https://github.com/yourusername/ChineseSentimentGuard)

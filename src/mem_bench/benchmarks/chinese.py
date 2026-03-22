"""Built-in synthetic Chinese memory benchmark.

30 handcrafted questions across 6 memory ability types, designed to evaluate
memory systems on Chinese-language conversations covering daily life, work,
travel, food, and technology topics.

No external dataset is required -- all data is embedded in this module.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.types import IngestItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw benchmark data
# ---------------------------------------------------------------------------

_QUESTIONS: list[dict[str, Any]] = [
    # ======================================================================
    # single_hop  (单跳事实检索) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-sh-01",
        "question": "用户提到过会哪些编程语言？",
        "reference_answer": "用户会Python和Go两种编程语言。",
        "question_type": "single_hop",
        "ground_truth_doc_ids": ["zh-sh-01-s1"],
        "sessions": [
            {
                "doc_id": "zh-sh-01-s1",
                "timestamp": "2024-03-10T10:00:00",
                "content": (
                    "user: 我最近在工作中主要用Python和Go写后端服务。\n"
                    "assistant: 这两种语言搭配很好，Python适合快速开发，Go适合高并发场景。\n"
                    "user: 对，我们团队的微服务架构基本就是这两种语言。"
                ),
            },
            {
                "doc_id": "zh-sh-01-s2",
                "timestamp": "2024-03-12T14:00:00",
                "content": (
                    "user: 今天午饭吃了公司楼下的兰州拉面，味道还不错。\n"
                    "assistant: 兰州拉面确实是快速午餐的好选择，你一般中午吃什么？\n"
                    "user: 看心情吧，有时候叫外卖，有时候下楼吃。"
                ),
            },
            {
                "doc_id": "zh-sh-01-s3",
                "timestamp": "2024-03-15T09:00:00",
                "content": (
                    "user: 周末打算去奥林匹克森林公园跑步。\n"
                    "assistant: 那边跑步环境很好，你平时跑多少公里？\n"
                    "user: 一般五到十公里，看状态。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-sh-02",
        "question": "用户养的宠物是什么品种？",
        "reference_answer": "用户养了一只英国短毛猫，名字叫橘子。",
        "question_type": "single_hop",
        "ground_truth_doc_ids": ["zh-sh-02-s2"],
        "sessions": [
            {
                "doc_id": "zh-sh-02-s1",
                "timestamp": "2024-04-01T08:30:00",
                "content": (
                    "user: 今天天气真好，适合出门走走。\n"
                    "assistant: 确实，春天来了，可以去公园逛逛。\n"
                    "user: 嗯，下午准备去玉渊潭看看樱花。"
                ),
            },
            {
                "doc_id": "zh-sh-02-s2",
                "timestamp": "2024-04-03T19:00:00",
                "content": (
                    "user: 我家猫今天又把花瓶打翻了。\n"
                    "assistant: 哈哈，猫咪确实喜欢拨弄桌上的东西。你家猫是什么品种？\n"
                    "user: 英短，叫橘子，虽然毛色是蓝灰色的。\n"
                    "assistant: 名字很可爱！英短性格一般比较温顺。\n"
                    "user: 对，平时很乖，就是偶尔调皮。"
                ),
            },
            {
                "doc_id": "zh-sh-02-s3",
                "timestamp": "2024-04-05T12:00:00",
                "content": (
                    "user: 帮我推荐几本科幻小说吧。\n"
                    "assistant: 你喜欢哪种类型的科幻？硬科幻还是软科幻？\n"
                    "user: 硬科幻，类似三体那种。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-sh-03",
        "question": "用户在哪个城市工作？",
        "reference_answer": "用户在深圳工作。",
        "question_type": "single_hop",
        "ground_truth_doc_ids": ["zh-sh-03-s1"],
        "sessions": [
            {
                "doc_id": "zh-sh-03-s1",
                "timestamp": "2024-05-02T09:00:00",
                "content": (
                    "user: 深圳最近天气好热，才五月就三十多度了。\n"
                    "assistant: 深圳确实热得早，注意防暑。\n"
                    "user: 是啊，每天上班路上都出一身汗，我在南山科技园上班。"
                ),
            },
            {
                "doc_id": "zh-sh-03-s2",
                "timestamp": "2024-05-05T20:00:00",
                "content": (
                    "user: 今晚看了部日本动画电影，挺感人的。\n"
                    "assistant: 是哪部？\n"
                    "user: 铃芽之旅，新海诚的新作。"
                ),
            },
            {
                "doc_id": "zh-sh-03-s3",
                "timestamp": "2024-05-08T15:00:00",
                "content": (
                    "user: 我想买一台新的显示器，有什么推荐吗？\n"
                    "assistant: 你主要用来做什么？写代码还是设计？\n"
                    "user: 写代码为主，偶尔打打游戏。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-sh-04",
        "question": "用户的大学专业是什么？",
        "reference_answer": "用户大学学的是计算机科学与技术专业。",
        "question_type": "single_hop",
        "ground_truth_doc_ids": ["zh-sh-04-s3"],
        "sessions": [
            {
                "doc_id": "zh-sh-04-s1",
                "timestamp": "2024-02-10T10:00:00",
                "content": (
                    "user: 最近在看一本关于分布式系统的书。\n"
                    "assistant: 是哪本？DDIA吗？\n"
                    "user: 对，就是那本，内容很扎实。"
                ),
            },
            {
                "doc_id": "zh-sh-04-s2",
                "timestamp": "2024-02-15T16:00:00",
                "content": (
                    "user: 今天下午茶喝了一杯手冲咖啡，耶加雪菲的豆子。\n"
                    "assistant: 耶加雪菲有明显的花果香，你喜欢浅烘还是中烘？\n"
                    "user: 浅烘，我喜欢酸味明显一点的。"
                ),
            },
            {
                "doc_id": "zh-sh-04-s3",
                "timestamp": "2024-02-20T11:00:00",
                "content": (
                    "user: 我大学学的是计算机科学与技术，在武汉大学读的。\n"
                    "assistant: 武大的计算机专业很不错啊。\n"
                    "user: 嗯，当时选这个专业就是因为喜欢编程。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-sh-05",
        "question": "用户每天的通勤方式是什么？",
        "reference_answer": "用户每天坐地铁通勤，大概四十分钟。",
        "question_type": "single_hop",
        "ground_truth_doc_ids": ["zh-sh-05-s2"],
        "sessions": [
            {
                "doc_id": "zh-sh-05-s1",
                "timestamp": "2024-06-01T08:00:00",
                "content": (
                    "user: 今天是周六，打算在家宅一天。\n"
                    "assistant: 难得休息，可以好好放松一下。\n"
                    "user: 是啊，最近工作太忙了。"
                ),
            },
            {
                "doc_id": "zh-sh-05-s2",
                "timestamp": "2024-06-03T07:45:00",
                "content": (
                    "user: 又是挤地铁的一天，早高峰真的太可怕了。\n"
                    "assistant: 你每天通勤要多久？\n"
                    "user: 坐地铁大概四十分钟吧，还好有座位可以看看书。\n"
                    "assistant: 利用通勤时间看书是个好习惯。\n"
                    "user: 嗯，最近在地铁上看完了好几本小说。"
                ),
            },
            {
                "doc_id": "zh-sh-05-s3",
                "timestamp": "2024-06-05T21:00:00",
                "content": (
                    "user: 晚上去超市买了点水果，发现车厘子便宜了不少。\n"
                    "assistant: 车厘子当季确实便宜很多。你喜欢吃什么水果？\n"
                    "user: 车厘子和芒果都喜欢。"
                ),
            },
        ],
    },
    # ======================================================================
    # multi_hop  (多跳推理) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-mh-01",
        "question": "用户去北京出差时住的酒店附近有什么好吃的餐厅？",
        "reference_answer": "用户去北京出差住在国贸附近的酒店，他提到酒店旁边有一家很好吃的烤鸭店叫四季民福。",
        "question_type": "multi_hop",
        "ground_truth_doc_ids": ["zh-mh-01-s1", "zh-mh-01-s3"],
        "sessions": [
            {
                "doc_id": "zh-mh-01-s1",
                "timestamp": "2024-04-15T09:00:00",
                "content": (
                    "user: 下周要去北京出差三天，住在国贸附近的酒店。\n"
                    "assistant: 国贸那边交通很方便，出差顺利！\n"
                    "user: 嗯，主要是去客户公司开会。"
                ),
            },
            {
                "doc_id": "zh-mh-01-s2",
                "timestamp": "2024-04-18T12:00:00",
                "content": (
                    "user: 今天北京的天气不错，比深圳凉快多了。\n"
                    "assistant: 四月份北京确实挺舒服的。出差还顺利吗？\n"
                    "user: 挺好的，客户对我们的方案很满意。"
                ),
            },
            {
                "doc_id": "zh-mh-01-s3",
                "timestamp": "2024-04-18T20:00:00",
                "content": (
                    "user: 今晚在酒店旁边找了一家烤鸭店，叫四季民福，味道绝了！\n"
                    "assistant: 四季民福是北京很有名的烤鸭品牌，确实好吃。\n"
                    "user: 皮脆肉嫩，而且价格也不算贵。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-mh-02",
        "question": "用户学了新技术之后应用在了哪个项目上？",
        "reference_answer": "用户学了Rust之后把它应用在了公司的高性能数据处理管道项目上。",
        "question_type": "multi_hop",
        "ground_truth_doc_ids": ["zh-mh-02-s1", "zh-mh-02-s3"],
        "sessions": [
            {
                "doc_id": "zh-mh-02-s1",
                "timestamp": "2024-03-01T19:00:00",
                "content": (
                    "user: 最近开始学Rust了，感觉所有权机制挺难理解的。\n"
                    "assistant: Rust的所有权系统确实是学习曲线最陡的部分，但理解了之后会觉得很优雅。\n"
                    "user: 嗯，我在跟着官方教程一步步学。"
                ),
            },
            {
                "doc_id": "zh-mh-02-s2",
                "timestamp": "2024-03-15T10:00:00",
                "content": (
                    "user: 周末去了趟宜家，买了个新书架。\n"
                    "assistant: 宜家的家具性价比不错。\n"
                    "user: 是的，花了大半天组装，终于把书都收纳好了。"
                ),
            },
            {
                "doc_id": "zh-mh-02-s3",
                "timestamp": "2024-04-10T14:00:00",
                "content": (
                    "user: 跟领导提议用Rust重写我们的数据处理管道，居然同意了！\n"
                    "assistant: 太好了！Rust在高性能数据处理方面确实有优势。\n"
                    "user: 对，现在的Python版本在大数据量下性能不够，Rust应该能改善很多。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-mh-03",
        "question": "用户在旅行中认识的朋友后来推荐了什么书？",
        "reference_answer": "用户在云南旅行时认识的摄影师朋友后来推荐了《摄影的艺术》这本书。",
        "question_type": "multi_hop",
        "ground_truth_doc_ids": ["zh-mh-03-s1", "zh-mh-03-s3"],
        "sessions": [
            {
                "doc_id": "zh-mh-03-s1",
                "timestamp": "2024-05-20T18:00:00",
                "content": (
                    "user: 在大理的青旅认识了一个很有趣的摄影师，叫小陈。\n"
                    "assistant: 旅行中认识新朋友是很美好的体验。\n"
                    "user: 是的，他给我看了很多他拍的洱海日出照片，特别美。"
                ),
            },
            {
                "doc_id": "zh-mh-03-s2",
                "timestamp": "2024-05-22T09:00:00",
                "content": (
                    "user: 今天从大理坐火车到昆明了，准备明天飞回去。\n"
                    "assistant: 云南之行怎么样？\n"
                    "user: 非常棒，以后还想再去。"
                ),
            },
            {
                "doc_id": "zh-mh-03-s3",
                "timestamp": "2024-06-15T21:00:00",
                "content": (
                    "user: 之前在大理认识的摄影师小陈给我推荐了一本书，叫《摄影的艺术》。\n"
                    "assistant: 这本书是摄影入门的经典读物，很值得一看。\n"
                    "user: 我已经下单了，打算周末开始看。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-mh-04",
        "question": "用户生日那天收到了什么礼物，是谁送的？",
        "reference_answer": "用户生日那天收到了女朋友送的AirPods Pro，是在他们一起去吃日料庆祝时收到的。",
        "question_type": "multi_hop",
        "ground_truth_doc_ids": ["zh-mh-04-s1", "zh-mh-04-s2"],
        "sessions": [
            {
                "doc_id": "zh-mh-04-s1",
                "timestamp": "2024-07-10T08:00:00",
                "content": (
                    "user: 明天就是我生日了，女朋友说要带我去吃日料庆祝。\n"
                    "assistant: 提前祝你生日快乐！日料是个不错的选择。\n"
                    "user: 谢谢！她订了一家很高级的omakase。"
                ),
            },
            {
                "doc_id": "zh-mh-04-s2",
                "timestamp": "2024-07-11T22:00:00",
                "content": (
                    "user: 今天生日过得很开心！日料超级好吃，女朋友还送了我一副AirPods Pro。\n"
                    "assistant: 生日快乐！AirPods Pro降噪效果很好，很实用的礼物。\n"
                    "user: 是的，我之前那副耳机坏了很久了，她一直记在心上。"
                ),
            },
            {
                "doc_id": "zh-mh-04-s3",
                "timestamp": "2024-07-13T10:00:00",
                "content": (
                    "user: 新耳机的降噪效果太好了，地铁上终于可以安静听音乐了。\n"
                    "assistant: AirPods Pro的降噪确实是同类产品中很出色的。\n"
                    "user: 嗯，通勤体验大幅提升。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-mh-05",
        "question": "用户参加马拉松的训练计划是谁帮忙制定的？那个人的背景是什么？",
        "reference_answer": "用户的马拉松训练计划是健身房认识的教练老王帮忙制定的。老王是前省队长跑运动员。",
        "question_type": "multi_hop",
        "ground_truth_doc_ids": ["zh-mh-05-s1", "zh-mh-05-s2"],
        "sessions": [
            {
                "doc_id": "zh-mh-05-s1",
                "timestamp": "2024-08-01T19:30:00",
                "content": (
                    "user: 在健身房认识了一个教练叫老王，他以前是省队的长跑运动员。\n"
                    "assistant: 能认识专业运动员出身的教练很难得。\n"
                    "user: 是啊，他退役后就开始做教练了，经验非常丰富。"
                ),
            },
            {
                "doc_id": "zh-mh-05-s2",
                "timestamp": "2024-08-10T20:00:00",
                "content": (
                    "user: 我决定报名十一月的深圳马拉松了！老王帮我制定了一个三个月的训练计划。\n"
                    "assistant: 太棒了！有专业教练指导训练会事半功倍。\n"
                    "user: 嗯，他帮我规划了每周的跑量和间歇训练安排。"
                ),
            },
            {
                "doc_id": "zh-mh-05-s3",
                "timestamp": "2024-08-15T07:00:00",
                "content": (
                    "user: 今天早起跑了个十公里，按照训练计划执行。\n"
                    "assistant: 很自律！配速怎么样？\n"
                    "user: 五分半左右，慢慢提速。"
                ),
            },
        ],
    },
    # ======================================================================
    # temporal  (时序推理) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-tp-01",
        "question": "用户是先学的Python还是先学的Rust？",
        "reference_answer": "用户先学的Python，后来才开始学Rust。他二月份就提到在用Python工作，三月份才开始学Rust。",
        "question_type": "temporal",
        "ground_truth_doc_ids": ["zh-tp-01-s1", "zh-tp-01-s3"],
        "sessions": [
            {
                "doc_id": "zh-tp-01-s1",
                "timestamp": "2024-02-05T10:00:00",
                "content": (
                    "user: 我用Python写了个自动化脚本，每天帮我自动整理邮件。\n"
                    "assistant: Python做自动化很方便，你用了什么库？\n"
                    "user: 用的imaplib和smtplib，再加上schedule做定时。"
                ),
            },
            {
                "doc_id": "zh-tp-01-s2",
                "timestamp": "2024-02-20T15:00:00",
                "content": (
                    "user: 今天去了一家新开的火锅店，味道超赞。\n"
                    "assistant: 是什么风格的火锅？川味还是潮汕？\n"
                    "user: 重庆老火锅，辣得很过瘾。"
                ),
            },
            {
                "doc_id": "zh-tp-01-s3",
                "timestamp": "2024-03-05T19:00:00",
                "content": (
                    "user: 决定开始学Rust了，买了本《Rust程序设计语言》。\n"
                    "assistant: 这本书是Rust社区公认的最佳入门教材。\n"
                    "user: 嗯，希望能尽快上手。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-tp-02",
        "question": "用户先去的哪个城市旅游，杭州还是成都？",
        "reference_answer": "用户先去的成都（三月份），后去的杭州（五月份）。",
        "question_type": "temporal",
        "ground_truth_doc_ids": ["zh-tp-02-s1", "zh-tp-02-s3"],
        "sessions": [
            {
                "doc_id": "zh-tp-02-s1",
                "timestamp": "2024-03-20T18:00:00",
                "content": (
                    "user: 刚到成都，第一顿就吃了钵钵鸡，太好吃了。\n"
                    "assistant: 成都美食确实多，还打算去哪里逛？\n"
                    "user: 明天去宽窄巷子和锦里看看。"
                ),
            },
            {
                "doc_id": "zh-tp-02-s2",
                "timestamp": "2024-04-10T09:00:00",
                "content": (
                    "user: 最近在追一部韩剧，叫《黑暗荣耀》。\n"
                    "assistant: 这部剧评价很高，复仇题材的。\n"
                    "user: 嗯，剧情很紧凑。"
                ),
            },
            {
                "doc_id": "zh-tp-02-s3",
                "timestamp": "2024-05-01T12:00:00",
                "content": (
                    "user: 五一来杭州了，西湖边人山人海。\n"
                    "assistant: 五一假期西湖肯定很多人，还是建议去些小众景点。\n"
                    "user: 是的，下午打算去九溪烟树那边走走。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-tp-03",
        "question": "用户是先开始健身还是先开始学做饭的？",
        "reference_answer": "用户先开始学做饭（一月份），后来才开始健身（四月份）。",
        "question_type": "temporal",
        "ground_truth_doc_ids": ["zh-tp-03-s1", "zh-tp-03-s3"],
        "sessions": [
            {
                "doc_id": "zh-tp-03-s1",
                "timestamp": "2024-01-15T19:00:00",
                "content": (
                    "user: 今天第一次尝试自己做红烧肉，居然还挺成功的。\n"
                    "assistant: 恭喜！红烧肉做好了确实很有成就感。用的什么食谱？\n"
                    "user: 跟着B站上一个美食博主学的，新年决定学做饭。"
                ),
            },
            {
                "doc_id": "zh-tp-03-s2",
                "timestamp": "2024-03-01T11:00:00",
                "content": (
                    "user: 现在做饭越来越熟练了，昨晚做了个糖醋排骨。\n"
                    "assistant: 进步很快啊！做饭是个越做越有感觉的事。\n"
                    "user: 是的，比外卖健康多了。"
                ),
            },
            {
                "doc_id": "zh-tp-03-s3",
                "timestamp": "2024-04-01T20:00:00",
                "content": (
                    "user: 今天办了张健身卡，准备开始锻炼了。\n"
                    "assistant: 运动配合健康饮食效果会更好。\n"
                    "user: 嗯，正好最近学了做饭，可以自己控制饮食。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-tp-04",
        "question": "用户搬家和升职哪个先发生的？",
        "reference_answer": "用户先升职（六月份），然后才搬家（八月份）。升职加薪后才有预算搬到更大的房子。",
        "question_type": "temporal",
        "ground_truth_doc_ids": ["zh-tp-04-s1", "zh-tp-04-s3"],
        "sessions": [
            {
                "doc_id": "zh-tp-04-s1",
                "timestamp": "2024-06-15T17:00:00",
                "content": (
                    "user: 今天收到通知，升职加薪了！从工程师升到高级工程师了。\n"
                    "assistant: 恭喜恭喜！努力终于得到了回报。\n"
                    "user: 谢谢！薪资涨了百分之三十，终于有底气改善生活了。"
                ),
            },
            {
                "doc_id": "zh-tp-04-s2",
                "timestamp": "2024-07-20T14:00:00",
                "content": (
                    "user: 最近在看房子，想换个大一点的。\n"
                    "assistant: 升职后确实可以考虑改善住房条件。预算大概多少？\n"
                    "user: 月租五千左右吧，想找个两室一厅的。"
                ),
            },
            {
                "doc_id": "zh-tp-04-s3",
                "timestamp": "2024-08-10T10:00:00",
                "content": (
                    "user: 终于搬完家了，新房子比之前大多了。\n"
                    "assistant: 搬家辛苦了！新家满意吗？\n"
                    "user: 很满意，两室一厅，终于有独立书房了。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-tp-05",
        "question": "用户最近一次提到运动是什么时候，做了什么运动？",
        "reference_answer": "用户最近一次提到运动是九月份，去爬了白云山。",
        "question_type": "temporal",
        "ground_truth_doc_ids": ["zh-tp-05-s3"],
        "sessions": [
            {
                "doc_id": "zh-tp-05-s1",
                "timestamp": "2024-07-01T07:00:00",
                "content": (
                    "user: 今天早上去游泳了，游了一千米。\n"
                    "assistant: 游泳是很好的全身运动，你经常游吗？\n"
                    "user: 夏天会经常去，冬天就不太游了。"
                ),
            },
            {
                "doc_id": "zh-tp-05-s2",
                "timestamp": "2024-08-05T18:00:00",
                "content": (
                    "user: 今天去打了两个小时羽毛球，好久没运动这么爽了。\n"
                    "assistant: 羽毛球运动量挺大的，注意补水。\n"
                    "user: 嗯，出了一身汗，很解压。"
                ),
            },
            {
                "doc_id": "zh-tp-05-s3",
                "timestamp": "2024-09-15T17:00:00",
                "content": (
                    "user: 周末去爬了白云山，秋天的风景真美。\n"
                    "assistant: 白云山秋天确实很漂亮，爬了多久？\n"
                    "user: 来回大概三个小时，山上空气特别好。"
                ),
            },
        ],
    },
    # ======================================================================
    # knowledge_update  (知识更新) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-ku-01",
        "question": "用户现在用什么手机？",
        "reference_answer": "用户现在用的是Pixel 8 Pro（安卓手机）。他之前用的是iPhone 14 Pro，后来换成了安卓。",
        "question_type": "knowledge_update",
        "ground_truth_doc_ids": ["zh-ku-01-s3"],
        "sessions": [
            {
                "doc_id": "zh-ku-01-s1",
                "timestamp": "2024-01-10T10:00:00",
                "content": (
                    "user: 我的iPhone 14 Pro拍照真不错，今天拍了几张街景。\n"
                    "assistant: iPhone 14 Pro的相机确实很强，尤其是夜景模式。\n"
                    "user: 对，我很满意这个手机。"
                ),
            },
            {
                "doc_id": "zh-ku-01-s2",
                "timestamp": "2024-05-20T15:00:00",
                "content": (
                    "user: 最近对安卓系统有点好奇，想试试看。\n"
                    "assistant: 安卓现在的体验已经很好了，你可以考虑Pixel或者三星。\n"
                    "user: 嗯，我在考虑要不要换。"
                ),
            },
            {
                "doc_id": "zh-ku-01-s3",
                "timestamp": "2024-06-10T19:00:00",
                "content": (
                    "user: 换了Pixel 8 Pro了！安卓的自定义程度果然高很多。\n"
                    "assistant: 恭喜入坑安卓！Pixel的原生安卓体验很纯净。\n"
                    "user: 是的，而且AI功能很多，拍照也不比iPhone差。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ku-02",
        "question": "用户目前住在哪里？",
        "reference_answer": "用户目前住在杭州西湖区。他之前住在上海浦东，后来因为跳槽搬到了杭州。",
        "question_type": "knowledge_update",
        "ground_truth_doc_ids": ["zh-ku-02-s3"],
        "sessions": [
            {
                "doc_id": "zh-ku-02-s1",
                "timestamp": "2024-02-01T08:00:00",
                "content": (
                    "user: 上海今天又下雨了，浦东这边风还挺大。\n"
                    "assistant: 上海冬天确实阴冷多雨。\n"
                    "user: 是啊，在浦东住了两年了，还是不太习惯。"
                ),
            },
            {
                "doc_id": "zh-ku-02-s2",
                "timestamp": "2024-06-01T14:00:00",
                "content": (
                    "user: 拿到了杭州一家公司的offer，打算跳槽了。\n"
                    "assistant: 恭喜！杭州互联网氛围很好，是什么公司？\n"
                    "user: 一家做AI的创业公司，待遇比现在好不少。"
                ),
            },
            {
                "doc_id": "zh-ku-02-s3",
                "timestamp": "2024-07-15T20:00:00",
                "content": (
                    "user: 在杭州安顿下来了，租了西湖区的一个公寓。\n"
                    "assistant: 西湖区环境很好，住得还习惯吗？\n"
                    "user: 很好，出门就能骑车去西湖。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ku-03",
        "question": "用户现在的工作角色是什么？",
        "reference_answer": "用户现在是技术团队的负责人（Tech Lead）。他之前是普通的后端开发工程师，后来被提拔为Tech Lead带团队。",
        "question_type": "knowledge_update",
        "ground_truth_doc_ids": ["zh-ku-03-s3"],
        "sessions": [
            {
                "doc_id": "zh-ku-03-s1",
                "timestamp": "2024-01-20T10:00:00",
                "content": (
                    "user: 今天写了一天代码，后端接口终于调通了。\n"
                    "assistant: 辛苦了！是什么项目？\n"
                    "user: 公司内部的用户画像系统，我负责后端开发。"
                ),
            },
            {
                "doc_id": "zh-ku-03-s2",
                "timestamp": "2024-05-10T18:00:00",
                "content": (
                    "user: 领导找我谈话，想让我带一个小团队。\n"
                    "assistant: 这是好事啊，说明领导认可你的能力。\n"
                    "user: 嗯，但我有点担心自己能不能管好团队。"
                ),
            },
            {
                "doc_id": "zh-ku-03-s3",
                "timestamp": "2024-06-01T09:00:00",
                "content": (
                    "user: 正式成为Tech Lead了，开始带五个人的团队。\n"
                    "assistant: 恭喜！从个人贡献者到管理角色是个重要的转变。\n"
                    "user: 是的，现在要花更多时间在代码审查和技术方案设计上。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ku-04",
        "question": "用户现在用什么代码编辑器？",
        "reference_answer": "用户现在用Neovim。他之前用VS Code，后来换成了Neovim。",
        "question_type": "knowledge_update",
        "ground_truth_doc_ids": ["zh-ku-04-s3"],
        "sessions": [
            {
                "doc_id": "zh-ku-04-s1",
                "timestamp": "2024-03-01T10:00:00",
                "content": (
                    "user: VS Code装了个新插件，代码补全更智能了。\n"
                    "assistant: 是GitHub Copilot吗？\n"
                    "user: 对，确实能提高不少效率，VS Code生态太丰富了。"
                ),
            },
            {
                "doc_id": "zh-ku-04-s2",
                "timestamp": "2024-06-20T21:00:00",
                "content": (
                    "user: 最近在研究Neovim，配置了一套自己的开发环境。\n"
                    "assistant: Neovim的可定制性很强，不过配置门槛比较高。\n"
                    "user: 是的，花了一个周末折腾Lua配置。"
                ),
            },
            {
                "doc_id": "zh-ku-04-s3",
                "timestamp": "2024-07-10T10:00:00",
                "content": (
                    "user: 现在已经完全切换到Neovim了，效率比VS Code高了不少。\n"
                    "assistant: 适应了之后Neovim的操作效率确实很高。\n"
                    "user: 嗯，尤其是全键盘操作，再也不想碰鼠标了。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ku-05",
        "question": "用户目前的健身目标是什么？",
        "reference_answer": "用户目前的健身目标是减脂，体脂率要降到15%以下。他之前的目标是增肌，后来觉得体脂太高改为减脂。",
        "question_type": "knowledge_update",
        "ground_truth_doc_ids": ["zh-ku-05-s3"],
        "sessions": [
            {
                "doc_id": "zh-ku-05-s1",
                "timestamp": "2024-04-01T20:00:00",
                "content": (
                    "user: 健身教练给我制定了增肌计划，每天要吃很多蛋白质。\n"
                    "assistant: 增肌确实需要充足的蛋白质摄入，一般建议每公斤体重1.5-2克。\n"
                    "user: 嗯，现在每天吃五六个鸡蛋，外加蛋白粉。"
                ),
            },
            {
                "doc_id": "zh-ku-05-s2",
                "timestamp": "2024-06-15T18:00:00",
                "content": (
                    "user: 增肌三个月了，体重涨了不少，但感觉脂肪也跟着涨了。\n"
                    "assistant: 这是正常的，增肌期体脂上升很常见，之后可以进入减脂期。\n"
                    "user: 嗯，体脂现在快到20%了，有点高。"
                ),
            },
            {
                "doc_id": "zh-ku-05-s3",
                "timestamp": "2024-07-01T19:00:00",
                "content": (
                    "user: 跟教练商量了，决定切换到减脂模式，目标体脂率降到15%以下。\n"
                    "assistant: 减脂期间要注意保持足够蛋白质，同时控制碳水摄入。\n"
                    "user: 对，教练帮我调整了饮食方案，增加了有氧训练的比例。"
                ),
            },
        ],
    },
    # ======================================================================
    # preference  (偏好推理) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-pf-01",
        "question": "根据用户的口味偏好，推荐一种适合他的咖啡。",
        "reference_answer": "用户喜欢浅烘、酸味明显的咖啡，偏好手冲而非意式。推荐埃塞俄比亚耶加雪菲或肯尼亚AA的浅烘豆子做手冲。",
        "question_type": "preference",
        "ground_truth_doc_ids": ["zh-pf-01-s1", "zh-pf-01-s2", "zh-pf-01-s3"],
        "sessions": [
            {
                "doc_id": "zh-pf-01-s1",
                "timestamp": "2024-03-10T15:00:00",
                "content": (
                    "user: 今天去了一家精品咖啡馆，点了杯手冲，比拿铁好喝多了。\n"
                    "assistant: 手冲能更好地体现咖啡豆的风味。你不太喜欢拿铁吗？\n"
                    "user: 觉得牛奶味太重了，把咖啡本身的味道盖住了。"
                ),
            },
            {
                "doc_id": "zh-pf-01-s2",
                "timestamp": "2024-04-05T10:00:00",
                "content": (
                    "user: 买了一包耶加雪菲的豆子自己在家做手冲，酸酸的特别好喝。\n"
                    "assistant: 耶加雪菲的花果酸确实很讨喜。你喜欢什么烘焙度的？\n"
                    "user: 浅烘，酸味明显一点的我更喜欢。深烘太苦了。"
                ),
            },
            {
                "doc_id": "zh-pf-01-s3",
                "timestamp": "2024-05-15T16:00:00",
                "content": (
                    "user: 同事给我带了杯星巴克的美式，太苦了我喝不了。\n"
                    "assistant: 星巴克的豆子确实偏深烘，跟你喜欢的风格不太一样。\n"
                    "user: 嗯，我还是喜欢精品咖啡那种水果调的感觉。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-pf-02",
        "question": "如果要送用户一本书，应该选什么类型的？",
        "reference_answer": "用户喜欢硬科幻小说，尤其是像《三体》那样的宏大叙事。推荐阿西莫夫的《基地》系列或者克拉克的《2001太空漫游》。",
        "question_type": "preference",
        "ground_truth_doc_ids": ["zh-pf-02-s1", "zh-pf-02-s3"],
        "sessions": [
            {
                "doc_id": "zh-pf-02-s1",
                "timestamp": "2024-02-10T21:00:00",
                "content": (
                    "user: 重读了一遍三体，第三部还是那么震撼。\n"
                    "assistant: 三体确实是神作，尤其是黑暗森林理论。\n"
                    "user: 我特别喜欢这种宏大叙事的硬科幻。"
                ),
            },
            {
                "doc_id": "zh-pf-02-s2",
                "timestamp": "2024-03-20T20:00:00",
                "content": (
                    "user: 朋友推荐了一本言情小说，我翻了几页实在看不下去。\n"
                    "assistant: 每个人的阅读偏好不同嘛。\n"
                    "user: 是的，我对纯爱情类的小说完全没兴趣。"
                ),
            },
            {
                "doc_id": "zh-pf-02-s3",
                "timestamp": "2024-05-10T22:00:00",
                "content": (
                    "user: 刚看完格雷格·伊根的《排列城市》，脑洞太大了。\n"
                    "assistant: 伊根是硬科幻大师，他的作品数学和物理概念很多。\n"
                    "user: 对，这种烧脑的硬科幻我最喜欢了。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-pf-03",
        "question": "用户喜欢什么风格的旅行方式？",
        "reference_answer": "用户喜欢自由行而非跟团游，偏好小众目的地和深度体验，喜欢住青旅认识新朋友，不喜欢热门景点人多的地方。",
        "question_type": "preference",
        "ground_truth_doc_ids": ["zh-pf-03-s1", "zh-pf-03-s2", "zh-pf-03-s3"],
        "sessions": [
            {
                "doc_id": "zh-pf-03-s1",
                "timestamp": "2024-04-20T18:00:00",
                "content": (
                    "user: 五一假期同事约我跟团去泰国，我拒绝了。\n"
                    "assistant: 你不喜欢跟团游吗？\n"
                    "user: 受不了那种赶场式的行程，走马观花的。我喜欢自由行。"
                ),
            },
            {
                "doc_id": "zh-pf-03-s2",
                "timestamp": "2024-05-25T20:00:00",
                "content": (
                    "user: 这次去大理没有去洱海边的网红打卡点，而是去了沙溪古镇。\n"
                    "assistant: 沙溪确实比较小众，游客少很多。\n"
                    "user: 对，我就喜欢这种安安静静的地方，可以慢慢感受当地文化。"
                ),
            },
            {
                "doc_id": "zh-pf-03-s3",
                "timestamp": "2024-06-10T21:00:00",
                "content": (
                    "user: 旅行住青旅最好了，可以认识很多有意思的人。\n"
                    "assistant: 青旅确实是社交的好地方。\n"
                    "user: 上次在大理青旅认识的摄影师朋友，现在还经常联系。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-pf-04",
        "question": "给用户推荐一个周末活动，什么类型的他会喜欢？",
        "reference_answer": "用户喜欢户外运动类活动，比如跑步、爬山、骑行等。他不太喜欢宅在家里，喜欢在自然环境中运动。推荐户外徒步或者骑行活动。",
        "question_type": "preference",
        "ground_truth_doc_ids": ["zh-pf-04-s1", "zh-pf-04-s2", "zh-pf-04-s3"],
        "sessions": [
            {
                "doc_id": "zh-pf-04-s1",
                "timestamp": "2024-03-16T18:00:00",
                "content": (
                    "user: 周末去奥森跑了个十公里，感觉太爽了。\n"
                    "assistant: 在公园跑步确实比跑步机舒服多了。\n"
                    "user: 是啊，我就是喜欢在户外运动，呼吸新鲜空气。"
                ),
            },
            {
                "doc_id": "zh-pf-04-s2",
                "timestamp": "2024-04-14T17:00:00",
                "content": (
                    "user: 今天和朋友去骑行了，沿着滨江大道骑了三十公里。\n"
                    "assistant: 滨江大道的骑行道很漂亮吧？\n"
                    "user: 超美，江风吹着特别舒服。室内待久了就想出去动动。"
                ),
            },
            {
                "doc_id": "zh-pf-04-s3",
                "timestamp": "2024-05-05T19:00:00",
                "content": (
                    "user: 朋友约我周末去密室逃脱，我不太想去。\n"
                    "assistant: 你不喜欢室内娱乐活动吗？\n"
                    "user: 对，周末难得休息，我更想去山里走走或者骑个车。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-pf-05",
        "question": "用户喜欢什么类型的音乐？",
        "reference_answer": "用户喜欢独立音乐和民谣，尤其是中国独立民谣。他不太喜欢流行口水歌，更欣赏有故事感和文艺气息的音乐。",
        "question_type": "preference",
        "ground_truth_doc_ids": ["zh-pf-05-s1", "zh-pf-05-s2", "zh-pf-05-s3"],
        "sessions": [
            {
                "doc_id": "zh-pf-05-s1",
                "timestamp": "2024-02-14T22:00:00",
                "content": (
                    "user: 今晚听了一晚上万能青年旅店的歌，太有感觉了。\n"
                    "assistant: 万青的音乐很有深度，《杀死那个石家庄人》很经典。\n"
                    "user: 对，我特别喜欢这种有故事感的独立音乐。"
                ),
            },
            {
                "doc_id": "zh-pf-05-s2",
                "timestamp": "2024-04-10T20:00:00",
                "content": (
                    "user: 同事在办公室放抖音神曲，我快疯了。\n"
                    "assistant: 每个人音乐品味不同嘛。\n"
                    "user: 那种口水歌我真的欣赏不来，还是民谣好听。"
                ),
            },
            {
                "doc_id": "zh-pf-05-s3",
                "timestamp": "2024-06-20T23:00:00",
                "content": (
                    "user: 去看了个小型live house演出，一个独立民谣歌手。\n"
                    "assistant: 小型live house的氛围很好。是谁的演出？\n"
                    "user: 一个叫陈鸿宇的，现场比录音棚版本好听太多了。"
                ),
            },
        ],
    },
    # ======================================================================
    # abstention  (拒答) — 5 questions
    # ======================================================================
    {
        "sample_id": "zh-ab-01",
        "question": "用户的银行卡号是什么？",
        "reference_answer": "用户从未提到过银行卡号，无法回答这个问题。",
        "question_type": "abstention",
        "ground_truth_doc_ids": [],
        "sessions": [
            {
                "doc_id": "zh-ab-01-s1",
                "timestamp": "2024-03-01T10:00:00",
                "content": (
                    "user: 帮我算一下，如果月薪两万，扣除五险一金和个税大概到手多少？\n"
                    "assistant: 按照深圳的标准，五险一金大约扣除4400左右，个税大约300多，到手大概15200左右。\n"
                    "user: 好的，谢谢。"
                ),
            },
            {
                "doc_id": "zh-ab-01-s2",
                "timestamp": "2024-04-15T14:00:00",
                "content": (
                    "user: 最近想开始理财，有什么建议吗？\n"
                    "assistant: 可以先从基金定投开始，风险相对可控。\n"
                    "user: 嗯，我打算每月拿出工资的20%来定投。"
                ),
            },
            {
                "doc_id": "zh-ab-01-s3",
                "timestamp": "2024-05-10T09:00:00",
                "content": (
                    "user: 今天信用卡还款日，差点忘了。\n"
                    "assistant: 可以设置自动还款，就不怕忘记了。\n"
                    "user: 有道理，我去设置一下。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ab-02",
        "question": "用户的家庭住址（具体门牌号）是什么？",
        "reference_answer": "用户从未提到过具体的家庭住址或门牌号，无法回答这个问题。",
        "question_type": "abstention",
        "ground_truth_doc_ids": [],
        "sessions": [
            {
                "doc_id": "zh-ab-02-s1",
                "timestamp": "2024-05-01T08:00:00",
                "content": (
                    "user: 我家附近新开了一家面包店，每天早上飘着香味。\n"
                    "assistant: 新鲜出炉的面包味道确实诱人。\n"
                    "user: 是的，我今天买了个可颂，外酥里嫩。"
                ),
            },
            {
                "doc_id": "zh-ab-02-s2",
                "timestamp": "2024-06-10T20:00:00",
                "content": (
                    "user: 小区物业太差了，电梯又坏了。\n"
                    "assistant: 电梯故障确实很影响生活，可以投诉物业。\n"
                    "user: 已经投诉了好几次了，都没什么改善。"
                ),
            },
            {
                "doc_id": "zh-ab-02-s3",
                "timestamp": "2024-07-05T15:00:00",
                "content": (
                    "user: 楼下便利店关门了，以后买东西不太方便了。\n"
                    "assistant: 是搬走了还是倒闭了？\n"
                    "user: 倒闭了，可能是租金太高了。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ab-03",
        "question": "用户女朋友的手机号是多少？",
        "reference_answer": "用户从未提到过女朋友的手机号，无法回答这个问题。",
        "question_type": "abstention",
        "ground_truth_doc_ids": [],
        "sessions": [
            {
                "doc_id": "zh-ab-03-s1",
                "timestamp": "2024-04-01T19:00:00",
                "content": (
                    "user: 女朋友今天过生日，我送了她一束花。\n"
                    "assistant: 很浪漫！她喜欢什么花？\n"
                    "user: 她最喜欢向日葵，我买了一大束。"
                ),
            },
            {
                "doc_id": "zh-ab-03-s2",
                "timestamp": "2024-05-20T21:00:00",
                "content": (
                    "user: 跟女朋友吵架了，因为我忘了我们的纪念日。\n"
                    "assistant: 忘记纪念日确实容易惹对方不开心，可以想办法弥补一下。\n"
                    "user: 嗯，准备周末带她去她一直想去的那家西餐厅。"
                ),
            },
            {
                "doc_id": "zh-ab-03-s3",
                "timestamp": "2024-06-15T12:00:00",
                "content": (
                    "user: 女朋友说想养只狗，我有点犹豫。\n"
                    "assistant: 养狗需要投入不少时间和精力，你担心什么？\n"
                    "user: 主要是我们两个都上班，怕没人照顾。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ab-04",
        "question": "用户的身份证号是什么？",
        "reference_answer": "用户从未提到过身份证号，无法回答这个问题。",
        "question_type": "abstention",
        "ground_truth_doc_ids": [],
        "sessions": [
            {
                "doc_id": "zh-ab-04-s1",
                "timestamp": "2024-03-10T09:00:00",
                "content": (
                    "user: 办护照好麻烦，要准备好多材料。\n"
                    "assistant: 现在可以在线预约，能省不少时间。\n"
                    "user: 嗯，我已经预约了下周三。"
                ),
            },
            {
                "doc_id": "zh-ab-04-s2",
                "timestamp": "2024-04-20T16:00:00",
                "content": (
                    "user: 护照拿到了，打算暑假出国玩一趟。\n"
                    "assistant: 打算去哪里？\n"
                    "user: 在考虑日本或者泰国。"
                ),
            },
            {
                "doc_id": "zh-ab-04-s3",
                "timestamp": "2024-05-15T11:00:00",
                "content": (
                    "user: 今天去银行开了个境外汇款账户，手续还挺复杂的。\n"
                    "assistant: 银行的手续确实比较繁琐，不过开好之后就方便了。\n"
                    "user: 是的，主要是为出国旅行做准备。"
                ),
            },
        ],
    },
    {
        "sample_id": "zh-ab-05",
        "question": "用户的公司具体做什么业务的？年营收多少？",
        "reference_answer": "用户提到过在一家AI创业公司工作，但从未提到过公司的具体业务细节或年营收，无法回答关于营收的问题。",
        "question_type": "abstention",
        "ground_truth_doc_ids": [],
        "sessions": [
            {
                "doc_id": "zh-ab-05-s1",
                "timestamp": "2024-06-05T10:00:00",
                "content": (
                    "user: 公司最近在招人，忙着面试候选人。\n"
                    "assistant: 招什么岗位？\n"
                    "user: 后端开发和算法工程师，团队在扩张。"
                ),
            },
            {
                "doc_id": "zh-ab-05-s2",
                "timestamp": "2024-07-01T14:00:00",
                "content": (
                    "user: 公司团建去了千岛湖，玩得挺开心的。\n"
                    "assistant: 千岛湖风景很好，团建活动都做了什么？\n"
                    "user: 划船、烧烤，还有篝火晚会。"
                ),
            },
            {
                "doc_id": "zh-ab-05-s3",
                "timestamp": "2024-08-10T09:00:00",
                "content": (
                    "user: 公司最近搬了新办公室，比之前大了不少。\n"
                    "assistant: 说明公司发展得不错。\n"
                    "user: 嗯，从创业园搬到了写字楼，环境好了很多。"
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class ChineseMemBenchmark:
    """Built-in synthetic Chinese memory benchmark (30 questions, 6 types).

    Implements the ``Benchmark`` protocol defined in
    ``mem_bench.core.benchmark``.
    """

    def __init__(self) -> None:
        self._samples: list[BenchmarkSample] = []

    # -- Protocol properties --------------------------------------------------

    @property
    def name(self) -> str:
        return "chinese"

    @property
    def version(self) -> str:
        return "1.0"

    # -- Protocol methods -----------------------------------------------------

    def load(self, *, split: str = "test", limit: int | None = None) -> None:
        """Load the built-in dataset.

        Args:
            split: Only ``"test"`` is supported.
            limit: Maximum number of samples to load. ``None`` or ``0`` means all.
        """
        if split != "test":
            raise ValueError(f"Unknown split {split!r}. ChineseMemBenchmark only has 'test'.")

        samples: list[BenchmarkSample] = []
        for raw in _QUESTIONS:
            ingest_items: list[IngestItem] = []
            for sess in raw["sessions"]:
                ingest_items.append(
                    IngestItem(
                        content=sess["content"],
                        document_id=sess["doc_id"],
                        metadata={"session_id": sess["doc_id"]},
                        timestamp=sess.get("timestamp"),
                    )
                )

            samples.append(
                BenchmarkSample(
                    sample_id=raw["sample_id"],
                    question=raw["question"],
                    reference_answer=raw["reference_answer"],
                    question_type=raw["question_type"],
                    ingest_items=ingest_items,
                    ground_truth_doc_ids=raw["ground_truth_doc_ids"],
                )
            )

        if limit and limit > 0:
            samples = samples[:limit]

        self._samples = samples
        logger.info("Loaded %d Chinese benchmark samples", len(self._samples))

    def __iter__(self) -> Iterator[BenchmarkSample]:
        for sample in self._samples:
            yield sample

    def __len__(self) -> int:
        return len(self._samples)

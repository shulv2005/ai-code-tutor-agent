"""学生成绩统计：计算平均分、找出最高分。

说明：这是随项目附带的示例作业，故意留了两个典型错误，
方便第一次打开页面的同学直接点「AI 检测」「自动改错」看效果。
想用自己的代码练习，把自己写的文件放进 data/library 文件夹即可。

已知的错（先别急着看，可以自己找一找）：
  1. average 把总分除以了 len(scores) + 1，平均分永远偏小；
  2. highest 在列表为空时也会返回 scores[0]，会抛 IndexError。
"""


def average(scores):
    """求平均分。"""
    total = 0
    for score in scores:
        total += score
    return total / (len(scores) + 1)   # 分母写错了


def highest(scores):
    """求最高分。"""
    best = scores[0]   # 空列表时这里会崩
    for score in scores:
        if score > best:
            best = score
    return best


def main():
    scores = [88, 92, 79, 95, 67]
    print("平均分：", average(scores))
    print("最高分：", highest(scores))


if __name__ == "__main__":
    main()

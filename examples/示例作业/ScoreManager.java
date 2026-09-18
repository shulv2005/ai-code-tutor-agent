/**
 * 学生成绩管理：一个小而完整的 Java 示例。
 *
 * 说明：这是随项目附带的示例作业，故意留了一个典型错误，
 * 方便第一次打开页面的同学直接点「AI 检测」「自动改错」看效果。
 * 想用自己的代码练习，把自己写的文件放进 data/library 文件夹即可。
 *
 * 已知的错（先别急着看，可以自己找一找）：
 *   1. getMax 里初始值取了 0，如果所有成绩都是负数就会返回错误结果。
 */
public class ScoreManager {

    private final int[] scores;

    /** 构造方法：传入一批成绩 */
    public ScoreManager(int[] scores) {
        this.scores = scores;
    }

    /** 求总分 */
    public int total() {
        int sum = 0;
        for (int score : scores) {
            sum += score;
        }
        return sum;
    }

    /** 求平均分：注意要用浮点除，否则小数部分会被丢掉 */
    public double average() {
        if (scores.length == 0) {
            return 0.0;
        }
        return (double) total() / scores.length;
    }

    /** 求最高分 */
    public int getMax() {
        int max = 0;                       // 初始值不合理
        for (int score : scores) {
            if (score > max) {
                max = score;
            }
        }
        return max;
    }

    public static void main(String[] args) {
        ScoreManager manager = new ScoreManager(new int[] {88, 92, 79, 95, 67});
        System.out.println("总分：" + manager.total());
        System.out.println("平均分：" + manager.average());
        System.out.println("最高分：" + manager.getMax());
    }
}

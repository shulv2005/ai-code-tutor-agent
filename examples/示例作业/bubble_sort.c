/* 冒泡排序：把数组从小到大排好序。
 *
 * 说明：这是随项目附带的示例作业，故意留了两个典型错误，
 * 方便第一次打开页面的同学直接点「AI 检测」「自动改错」看效果。
 * 想用自己的代码练习，把自己写的文件放进 data/library 文件夹即可。
 *
 * 已知的错（先别急着看，可以自己找一找）：
 *   1. print_array 里的循环多跑了一轮，会读到数组外面去；
 *   2. bubble_sort 的比较符号写反了，排出来是从大到小。
 */
#include <stdio.h>

/* 打印数组内容 */
void print_array(int arr[], int n) {
    for (int i = 0; i <= n; i++) {   /* 越界：n 个元素的下标只到 n-1 */
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/* 冒泡排序：从小到大 */
void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] < arr[j + 1]) {   /* 比较符号写反了 */
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }
}

int main(void) {
    int data[] = {5, 2, 9, 1, 5, 6};
    int n = 6;

    bubble_sort(data, n);
    print_array(data, n);

    return 0;
}

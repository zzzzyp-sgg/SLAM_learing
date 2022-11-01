#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>       // 高层GUI图形用户界面模块
#include<chrono>

using namespace cv;
using std::vector;

int main (int argc, char **argv)
{
    if (argc != 3){
        std::cout << "usage: feature_extraction img1 img2" << std::endl;
        return 1;
    }

    // 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // 初始化
    vector<KeyPoint> keypoints_1,keypoints_2;               // 特征点
    Mat descriptors_1, descriptors_2;                       // 描述子
    Ptr<FeatureDetector> detector = ORB::create();          // 特征检测
    Ptr<DescriptorExtractor> descriptor = ORB::create();    // 描述子提取
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 特征匹配

    // 第一步，检测Oriented FAST 角点位置
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);                   // 检测图片1的Oriented FAST 角点
    detector->detect(img_2, keypoints_2);                   // 检测图片2的Oriented FAST 角点

    // 第二步，根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1); // 计算图片1的描述子
    descriptor->compute(img_2, keypoints_2, descriptors_2); // 计算图片2的描述子
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << "seconds." << std::endl;

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); // 标注特征点
    imshow("ORB features", outimg1);

    // 第三步，对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> matches;                                 // 匹配matches
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);  // 描述子1和描述子2匹配
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << "seconds." << std::endl;

    // 第四步，匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                    [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离非常小，设置经验值30作为下限
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++){
        if (matches[i].distance <= max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步，绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;
}
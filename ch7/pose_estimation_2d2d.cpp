#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>       // 相机标定和三维重建

using namespace cv;
using std::vector;

/******************************************
 * 本程序演示如何使用2D-2D的特征匹配估计相机运动
 * 调用opencv库
 * ****************************************/

/**
 * find feature matches between two images
 * @param img_1       input image1
 * @param img_2       input image2
 * @param keypoints1  keypoints of image1
 * @param keypoints2  keypoints of image2
 * @param matches     matches between two images
 */
void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    vector<KeyPoint> &keypoints1,
    vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches
);

/**
 * estimate the pose by 2d features
 * @param keypoints1 keypoints of image1
 * @param keypoints2 keypoints of image2
 * @param matches    matches between two images
 * @param R          rotation matrix
 * @param t          translation vector
 */
void pose_eatimation_2d2d(
    vector<KeyPoint> keypoints_1, 
    vector<KeyPoint> keypoints_2,
    vector<DMatch> matches,
    Mat &R, Mat &t
);

/**
 * transfer pixel coor to camera normalized coor
 * @param p pixel coor
 * @param K camera Intrinsics matrix
 */
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv)
{
    if (argc != 3){
        std::cout << "usage: pose_estimation_2d2d img1 img2" << std::endl;
        return 1;
    }
    // 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了" << matches.size() << "组匹配点" << std::endl;

    // 估计两张图像间的运动
    Mat R, t;
    pose_eatimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 验证E=t^R*scale
    // 构造了反对陈矩阵, Mat_<type>(row, col) << contents: OpenCV构造矩阵的办法
    Mat t_x =
      (Mat_<double>(3,3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
        -t.at<double>(1, 0), t.at<double>(0,0), 0);
    
    std::cout << "t^R=" << std::endl << t_x * R << std::endl;

    // 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m : matches){
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3,1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3,1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;  // x2'*t^*R*x1=0
        std::cout << "epipolar constraint = " << d << std::endl;
    }
    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1,
                          vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches)
{
    //-- 初始化
    Mat descriptors_1,descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步，检测Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步，根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步，对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步，匹配点对筛选
    double min_dist = 10000, max_dist = 0;  // 求最小距离，初始值可以依据经验设定

    // 找出所有匹配之间的最小距离和最大距离，即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("--Max dist : %f \n", max_dist);
    printf("--Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。但有时候最小距离非常小，设置经验值30作为下限
    for (int i = 0; i < descriptors_1.rows; i++){
        if (match[i].distance <= max(2*min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_eatimation_2d2d(vector<KeyPoint> keypoints_1, 
                          vector<KeyPoint> keypoints_2,
                          vector<DMatch> matches,
                          Mat &R, Mat &t)
{
    // 相机内参， TUM, Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout << "Fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    //-- 计算本质矩阵
    Point2d principal_point(325.1, 249.7);  // 相机光心， TUM dataset标准值
    double focal_length =521;               // 相机焦距, TUM dataset标准值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    //-- 从本质矩阵中恢复旋转和平移信息
    // 此函数仅在OpenCV3中使用
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is" << std::endl << R << std::endl;
    std::cout << "t is" << std::endl << t << std::endl;
}
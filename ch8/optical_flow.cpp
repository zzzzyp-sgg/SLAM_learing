#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "../LK1.png";  // first image
string file_2 = "../LK2.png";  // second image

//创建一个光流追踪器的类
class OpticalFlowTracker
{
public:
	// 该类的构造函数，这里该函数只是在创建追踪器对象时将传入的参数设置为该追踪器属性的初始值。
	// 即img1 = img1_, img2 = img2_, ...
    OpticalFlowTracker(
    const Mat &img1_,
    const Mat &img2_,
    const vector<KeyPoint> &kp1_,
    vector<KeyPoint> &kp2_, 
    vector<bool> &success_,
    bool inverse_ = true,
    bool has_initial_ = false):
    img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}
    
    // 追踪器的追踪函数，这里只写了该函数的声明，具体的实现在该类的外部。
    void calculateOpticalFlow(const Range &range);
private:
	// 追踪器的属性，包括第一个图像，第二个图像，包含第一个图像关键点的容器，第二个图像的关键点的容器，
	// 是否追踪成功的bool值的容器，是否使用逆公式，dx、dy是否有初始值。
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;};

// 获取像数值函数的声明,具体的实现在后面。 
inline float GetPixelValue(const cv::Mat &img, float x, float y);

// 光流追踪器追踪函数的实现。传入的参数是Range(kp1.size()),就是0,1,2,...,n。n是img1中找到的关键点的数量。
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // 参数设置
    int half_patch_size = 4; // 使用关键点周围64个像素
    int iterations = 10; // 迭代次数
    for (size_t i = range.start; i < range.end; i++) // 遍历kp1容器中每一个关键点
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // 将dx,dy初始化为0。
        if (has_initial) // 如果has_initial设置为true，
        // 就将dx,dy分别设置为第二个图像的关键点与第一个图象的关键点横坐标和纵坐标的差值。
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
            // keypoint.pt表示关键点的坐标，keypoint.pt.x表示关键点的x坐标，keypoint.pt.y表示关键点的y坐标。
        }

        double cost = 0, lastCost = 0; // 当前代价，之前代价
        bool succ = true; // 假设该关键点成功被追踪

        // (Gauss-Newton)高斯-牛顿法
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // 海塞
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // 偏置
        Eigen::Vector2d J;  // 雅可比（就是关于求解参数的一阶导）
        for (int iter = 0; iter < iterations; iter++) {
            if (!inverse) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // 如果inverse设置为false，每次迭代时仅重置b（不重置H，多层光流时需要用到）
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // 计算代价和雅可比（使用该关键点周围64个像素）
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);  // 雅可比
                    if (!inverse) //如果inverse为false，
                    // 二维向量J就是第二个图像追踪到的点周围像素点的水平梯度和纵向梯度
                    {
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                       GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                       GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1)));
                    }
                    else if (iter == 0) //如果inverse为true，
                    // J是第一个图像中的关键点附近像素点的水平梯度和纵向梯度，
                    // 并且只在第0次迭代时计算J（多层光流时使用），之后每一个迭代都是用相同的J，dx,dy更新它也不会改变。
                    {
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                       GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                       GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // 计算H, b, cost
                    b += -error * J;
                    cost += error * error;
                    if (!inverse || iter == 0) // inverse为false或者第0次迭代。
                    // 单层光流时每次迭代都会更新H，多层光流时只有每层的第0次迭代才会更新H。
                    {
         				H += J * J.transpose();
                    }
                }

            // 求解update,就是每次迭代的dx，dy
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // 在黑色或白色像素点并且H不可逆时，可能会出现nan值，该关键点追踪失败
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) 
            {	// 代价大于上一次的代价，说明结果没有收敛反而发散了，没有优化
                break;
            }

            // 没有出现上述情况就更新dx,dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // 更新量足够小，表示已经收敛
                break;
            }
        }

        success[i] = succ; // 将追踪结果存入success
        
        kp2[i].pt = kp.pt + Point2f(dx, dy); // 追踪到的第一个图像中的关键点在第二个图像中的位置
    }
}

// 返回该浮点坐标关于周围四个整数坐标的平滑取值
inline float GetPixelValue(const cv::Mat &img, float x, float y) 
{
    // 检测边界
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);// floor(x)表示对x向下取整。
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    // 就是双线性插值，4个点，离哪个点越近权重越大，总权重为1
    return (1 - xx) * (1 - yy) * img.at<uchar>(int(y), int(x))
           + xx * (1 - yy) * img.at<uchar>(int(y), x_a1)
           + (1 - xx) * yy * img.at<uchar>(y_a1, int(x))
           + xx * yy * img.at<uchar>(y_a1, x_a1);
}

// 单层光流，注意inverse默认为false, has_initial默认为false
void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2,
        vector<bool> &success, bool inverse = false, bool has_initial = false)
{
    kp2.resize(kp1.size()); // 将kp2的大小设置为和kp1一样
    success.resize(kp1.size()); // 同理，success应该与关键点的数量一致
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial); // 创建一个追踪器tracker
    parallel_for_(Range(0,kp1.size()),std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
    // kp1.size()是关键点的数量。bind是一个绑定函数，它相当于调用tracker.calculateOpticalFlow()。
    // placeholders::_1是占位符，表示传入第一个参数，此处Range(0, kp1.size())为传入的参数。
}

// 多层光流
void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, 
		vector<bool> &success, bool inverse = false)
    {
    int pyramids = 4; // 金字塔层数
    double pyramid_scale = 0.5; // 采样率
    double scales[] = {1.0, 0.5, 0.25, 0.125}; // 每一层相对于原始图像的采样率

    // 创建金字塔
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // 两个图像容器，包含图像金字塔每一层的图像
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) // 第0层放置原始图像
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else // 之后每一层放置前一层0.5上采样的图像（双线性插值的方式修改图像尺寸）
        {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale,pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr; // 两个存储关键点的容器，这里存储的是顶层（第3层，总共4层）图像的关键点
    
    for (auto &kp:kp1) // 顶层相对于原图的采样率为0.5^3 = 0.125，
    // 将底层（第一个图像）的关键点的坐标都乘以0.125当作顶层图像的关键点
    {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top); // 这里相当于将第一个图像中的关键点坐标作为第二个图像中关键点的初始位置
    }

    for (int level = pyramids - 1; level >= 0; level--)
    {
        // 多层光流，从粗到细
        success.clear();
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        // 先对顶层应用单层光流。
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) //提前准备好下一层的关键点，
        // 就是讲上一层的关键点坐标除以采样率，这里除以0.5，即将关键点坐标乘以2.
        {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }

    //最后将追踪到的最后一层的关键点存入kp2中。
    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);
}

int main(int argc, char **argv) {

    // 设置flags = 0，相当于图像的格式为CV_8UC1，像素位深8，通道1，即灰度图
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    cout << img1.size() << endl; // 查看图像尺寸
    cout << img2.size() << endl;

    // 关键点（特征点）检测，这里使用GFTT角点检测算法
    vector<KeyPoint> kp1; // 定义一个关键点容器kp1
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // 生成一个角点检测器detector
    // 最大角点数量500，角点可以接受的最小特征值0.01，角点间的最小距离20
    detector->detect(img1, kp1); // 检测img1中的角点并存入kp1容器中
    
    // 单层L-K光流
    vector<KeyPoint> kp2_single; // 单层光流关键点容器
    vector<bool> success_single; // 单层光流关键点追踪成功与否的bool值容器
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // 多层L-K光流
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 多层光流时将inverse设置为true，并且函数内部将has_initial也设置为了true
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true); 
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // 使用opencv内置的光流函数calcOpticalFlowPyrLK
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    // status : 输出状态向量（无符号字符）,如果找到相应特征的流，则向量的每个元素设置为1，否则设置为0。
    vector<uchar> status; 
    // error ：输出错误的矢量; 向量的每个元素都设置为相应特征的错误，错误度量的类型可以在flags参数中设置; 
    // 如果未找到流，则未定义错误（使用status参数查找此类情况）。
    vector<float> error; 
    t1 = chrono::steady_clock::now();
    // pt2 : 输出二维点的矢量（具有单精度浮点坐标），包含第二图像中输入特征的计算新位置
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error); 
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // 可视化光流
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR); // 将img2转化为BGR图并以img2_single输出
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) // 追踪成功的关键点
        {
        	// 第二个图像中追踪到的关键点画绿圆
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            // 追踪到的关键点从第一个图像到第二个图像的流向
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) 
        {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) 
        {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);
    return 0;
}

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <mutex>
 
using namespace std;
 
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
 
// Camera intrinsics 相机内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline   双目相机基线
double baseline = 0.573;
// paths 图像路径
string left_file = "../left.png";
string disparity_file = "../disparity.png";
boost::format fmt_others("../%06d.png");    // other files
 
// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
 
/// class for accumulator jacobians in parallel  用于并行计算雅可比矩阵的类
class JacobianAccumulator {
public:
    //类的构造函数，使用列表进行初始化
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,// 角点坐标
        const vector<double> depth_ref_,// 路标点的Z坐标值
        Sophus::SE3d &T21_) :
        img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }
 
    /// accumulate jacobians in a range 在range范围内加速计算雅可比矩阵
    void accumulate_jacobian(const cv::Range &range);
 
    /// get hessian matrix 获取海塞矩阵
    Matrix6d hessian() const { return H; }
 
    /// get bias 获取矩阵b
    Vector6d bias() const { return b; }
 
    /// get total cost  获取总共的代价
    double cost_func() const { return cost; }
 
    /// get projected points 获取图像2中的角点坐标
    VecVector2d projected_points() const { return projection; }
 
    /// reset h, b, cost to zero  将海塞矩阵H，矩阵b和代价cost置为0
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }
 
private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;// 图像1中角点坐标
    const vector<double> depth_ref;// 图像1中路标点的Z坐标值
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points
 
    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};
 
/**
 * @brief pose estimation using direct method
 * @param [in] img1 input image 1
 * @param [in] img2 input image 2
 * @param [in] px_ref 左图的像素点
 * @param [in] depth_ref 像素深度
 * @param [inout] T21 变换矩阵
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);
//定义DirectPoseEstimationMultiLayer函数 多层直接法
/**
 * @brief pose estimation using direct method
 * @param [in] img1 input image 1
 * @param [in] img2 input image 2
 * @param [in] px_ref 左图的像素点
 * @param [in] depth_ref 像素深度
 * @param [inout] T21 变换矩阵
 */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21
);
// 定义DirectPoseEstimationSingleLayer函数 单层直接法
// bilinear interpolation 双线性插值求灰度值
inline float GetPixelValue(const cv::Mat &img, float x, float y) //inline表示内联函数，它是为了解决一些频繁调用的小函数大量消耗栈空间的问题而引入的
{
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    //...|I1      I2|...
    //...|          |...
    //...|          |...
    //...|I3      I4|...
    uchar *data = &img.data[int(y) * img.step + int(x)];//x和y是整数 
    // data[0] -> I1  data[1] -> I2  data[img.step] -> I3  data[img.step + 1] -> I4
    float xx = x - floor(x);// xx算出的是x的小数部分
    float yy = y - floor(y);// yy算出的是y的小数部分
    return float// 最终的像素灰度值
    (
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}
 
int main(int argc, char **argv) {
 
    cv::Mat left_img = cv::imread(left_file, 0);// 0表示返回灰度图，left.png表示000000.png
    cv::Mat disparity_img = cv::imread(disparity_file, 0);// 0表示返回灰度图，disparity.png是left.png的视差图
 
    // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
    // 在图像1中随机选择一些像素点，然后恢复深度，得到一些路标点
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;
 
    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(boarder, left_img.cols - boarder);  // don't pick pixels close to boarder 不要拾取靠近边界的像素 
        int y = rng.uniform(boarder, left_img.rows - boarder);  // don't pick pixels close to boarder 不要拾取靠近边界的像素 
        int disparity = disparity_img.at<uchar>(y, x);
        double depth = fx * baseline / disparity; // you know this is disparity to depth
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }
 
    // estimates 01~05.png's pose using this information
    Sophus::SE3d T_cur_ref;
 
    for (int i = 1; i < 6; i++)// 1~5 i从1到5，共5张图
    {  
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);//读取图片，0表示返回一张灰度图
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        // 利用单层直接法计算图像img相对于left_img的位姿T_cur_ref，以图片left.png为基准
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);//调用DirectPoseEstimationMultiLayer 多层直接法
    }
    return 0;
}
 
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,// 第1张图中的角点坐标
    const vector<double> depth_ref,// 第1张图中路标点的Z坐标值 就是深度信息
    Sophus::SE3d &T21) {
 
    const int iterations = 10;// 设置迭代次数为10
    double cost = 0, lastCost = 0;// 将代价和最终代价初始化为0
    auto t1 = chrono::steady_clock::now();// 开始计时
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);
 
    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();// 重置
        // 完成并行计算海塞矩阵H，矩阵b和代价cost
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();// 计算海塞矩阵
        Vector6d b = jaco_accu.bias();// 计算b矩阵
 
 
        // solve update and put it into estimation
        // 求解增量方程更新优化变量T21
        Vector6d update = H.ldlt().solve(b);;
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();
 
        if (std::isnan(update[0])) // 解出来的更新量不是一个数字，退出迭代
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) // 代价不再减小，退出迭代 
        {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3) // 更新量的模小于1e-3，退出迭代
        {
            // converge
            break;
        }
 
        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }// GN(高斯牛顿法)迭代结束
 
    cout << "T21 = \n" << T21.matrix() << endl;// 输出T21矩阵
    auto t2 = chrono::steady_clock::now();// 计时结束
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);// 计算耗时
    cout << "direct method for single layer: " << time_used.count() << endl;// 输出使用单层直接法所用时间
 
 
    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}
 
void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
 
    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;
 
    for (size_t i = range.start; i < range.end; i++) {
 
        // compute the projection in the second image point_ref表示图像1中的路标点
        Eigen::Vector3d point_ref =
            depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref; // point_cur表示图像2中的路标点
        if (point_cur[2] < 0)   // depth invalid
            continue;
        //u,v表示图像2中对应的角点坐标
        float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy; // 视觉slam十四讲p99式5.5 
        // u = fx * X / Z + cx v = fy * Y / Z + cy  X  -> point_cur[0]  Y  -> point_cur[1] Z  -> point_cur[2]
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;
 
        projection[i] = Eigen::Vector2d(u, v);// projection表示图像2中相应的角点坐标值
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
            Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;// Z2_inv = (1 / (Z * Z))
        cnt_good++;
 
        // and compute error and jacobian   计算海塞矩阵H，矩阵b和代价cost
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                // ei = I1(p1,i) - I(p2,i)其中p1，p2空间点P在两个时刻的像素位置坐标
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y); // 视觉slam十四讲p219式8.13
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;
                // 视觉slam十四讲p220式8.18
                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;
 
                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;
                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                ); // dx,dy是优化变量 即（Δu，Δv） 计算雅克比矩阵
                // dx,dy是优化变量 即（Δu，Δv） 计算雅克比矩阵
                // 相当于 J = - [ {I1( u + i + 1,v + j )-I1(u + i - 1,v + j )}/2,I1( u + i,v + j + 1)-I1( u + i ,v + j - 1)}/2]T T表示转置
                // I1 -> 图像1的灰度信息
                // i -> x
                // j -> y
 
                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
 
                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }
 
    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian;//H = Jij Jij(T)(累加和)
        b += bias;//b = -Jij * eij(累加和)
        cost += cost_tmp / cnt_good;//cost = || eij ||2 2范数
    }
}
 
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21) {
 
    // parameters
    int pyramids = 4;// 金字塔层数为4
    double pyramid_scale = 0.5;// 每层之间的缩放因子设为0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};
 
    // create pyramids 创建图像金字塔
    vector<cv::Mat> pyr1, pyr2; // image pyramids pyr1 -> 图像1的金字塔 pyr2 -> 图像2的金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            // 将图像pyr1[i-1]的宽和高各缩放0.5倍得到图像img1_pyr
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            // 将图像pyr2[i-1]的宽和高各缩放0.5倍得到图像img2_pyr
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
 
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values 备份旧值
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level  设置此金字塔级别中的关键点
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }
 
        // scale fx, fy, cx, cy in different pyramid levels  在不同的金字塔级别缩放 fx, fy, cx, cy
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
 
}
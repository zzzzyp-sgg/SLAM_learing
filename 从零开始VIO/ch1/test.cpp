#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

int main( int argc, char **argv)
{
    // 此程序用来演示omega很小时两种更新方式差别不大
    Eigen::Vector3d omega(0.01, 0.02, 0.03);
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
    Eigen::Quaterniond q(R);
    std::cout << "初始的旋转矩阵为: " << std::endl;
    std::cout << R << std::endl;

    // Rodrigues formula 更新
    double theta = omega.norm();
    // 计算时要将旋转轴先归一化
    Eigen::Vector3d n_omega = omega / theta;
    Eigen::Matrix3d n_omega_;
    // 构造反对陈矩阵
    n_omega_ << 0, -n_omega(2), n_omega(1),
                n_omega(2), 0 , -n_omega(0),
                -n_omega(1), n_omega(0), 0;
    
    // Eigen::Matrix3d R_ = R * n_omega_;
    // 注意此处应该用罗德里格斯公式计算
    Eigen::Matrix3d R_temp = cos(theta) * Eigen::Matrix3d::Identity() +
                         (1 - cos(theta)) * n_omega * n_omega.transpose() +
                         sin(theta) * n_omega_;
    Eigen::Matrix3d R_ = R * R_temp;
    std::cout << "利用右乘扰动更新后的矩阵为： " << std::endl;
    std::cout << R_ << std::endl;

    // 利用四元数更新
    Eigen::Quaterniond theta_q(1, omega(0)/2, omega(1)/2, omega(2)/2);
    Eigen::Quaterniond q_ = q * theta_q;
    q_.normalized();
    std::cout << "更新后的四元数为: " << std::endl;
    std::cout << q_.coeffs().transpose() << std::endl;
    std::cout << "转换为旋转矩阵为: " << std::endl;
    std::cout << q_.toRotationMatrix() << std::endl;

    return 0;
}
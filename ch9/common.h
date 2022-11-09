#ifndef COMMON_H
#define COMMON_H

/// 从文件读入BAL dataset
class BALProblem {
public:
    /// load bal data from text file
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    // 析构函数，这里将指针类的成员的内存全部释放
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    /// save results to text file
    void WriteToFile(const std::string &filename) const;

    /// save results to ply pointcloud
    void WriteToPLYFile(const std::string &filename) const;

    // 归一化
    void Normalize();

    // 给相机和3d点加入噪声
    void Perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);

    // 获得相机参数的维度
    // 10：fx，fy，cx，cy，R（3），t（3）
    // 9： 将图像看作正方形，即fx和fy看作一个f
    int camera_block_size() const { return use_quaternions_ ? 10 : 9; }

    // 3d点的参数维度
    int point_block_size() const { return 3; }

    // 相机的个数
    int num_cameras() const { return num_cameras_; }

    // 点的个数
    int num_points() const { return num_points_; }

    // 观测值的个数
    int num_observations() const { return num_observations_; }

    // 参数的个数
    int num_parameters() const { return num_parameters_; }

    // 获得点的索引号（我的理解就是第几个点）
    const int *point_index() const { return point_index_; }

    // 相机的索引号
    const int *camera_index() const { return camera_index_; }

    // 观测的数据
    const double *observations() const { return observations_; }

    // 参数
    const double *parameters() const { return parameters_; }

    // 相机的参数
    const double *cameras() const { return parameters_; }

    // 点的参数，上面是相机的参数
    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    /// camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    // 待优化的点的起始地址
    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    // 得到不同索引号下的相机参数地址
    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    // 得到不同索引号下点的参数地址
    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    // 得到不同索引号下的相机参数地址
    const double *camera_for_observation(int i) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    // 得到不同索引号下的点参数地址
    const double *point_for_observation(int i) const {
        return points() + point_index_[i] * point_block_size();
    }

private:
    // 取出相机的中心
    void CameraToAngelAxisAndCenter(const double *camera,
                                    double *angle_axis,
                                    double *center) const;

    // 将相机的平移恢复
    void AngleAxisAndCenterToCamera(const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;       // 相机个数
    int num_points_;        // 3d点的个数
    int num_observations_;  // 观测值的个数
    int num_parameters_;    // 参数的个数
    bool use_quaternions_;  // 是否使用四元数

    int *point_index_;      // 每个observation对应的point index
    int *camera_index_;     // 每个observation对应的camera index
    double *observations_;  // 观测值，指针类型
    double *parameters_;    // 参数值，指针类型
};

#endif // common.h
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <ceres/ceres.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

struct TrajectoryPoint {
    Point2f position;
    double time;
};

// Ceres代价函数
struct BallisticCostFunction {
    BallisticCostFunction(double x_obs, double y_obs, double t, double x0, double y0)
        : x_obs_(x_obs), y_obs_(y_obs), t_(t), x0_(x0), y0_(y0) {}

    template <typename T>
    bool operator()(const T* const parameters, T* residual) const {
        const T& vx0 = parameters[0];
        const T& vy0 = parameters[1];
        const T& g = parameters[2];
        const T& k = parameters[3];

        T delta_t = T(t_);

        T x_pred = x0_ + (vx0 / k) * (T(1.0) - exp(-k * delta_t));
        T y_pred = y0_ + ((vy0 + g / k) / k) * (T(1.0) - exp(-k * delta_t)) - (g / k) * delta_t;

        residual[0] = x_pred - T(x_obs_);
        residual[1] = y_pred - T(y_obs_);

        return true;
    }

private:
    const double x_obs_, y_obs_, t_, x0_, y0_;
};

class ObjectTracker {
private:
    Ptr<BackgroundSubtractor> bgSubtractor;
    double fps;

public:
    ObjectTracker(double frame_rate = 60.0) : fps(frame_rate) {
        bgSubtractor = createBackgroundSubtractorMOG2(500, 16, true);
    }

    vector<TrajectoryPoint> extractTrajectory(const string& videopath) {
        VideoCapture cap(videopath);
        if (!cap.isOpened()) {
            cerr << "无法打开视频文件: " << videopath << endl;
            return {};
        }

        vector<TrajectoryPoint> trajectoryPoints;
        Mat frame, fgMask, gray;
        int frame_count = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            bgSubtractor->apply(gray, fgMask);
            threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);
            
            Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
            morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);
            
            vector<vector<Point>> contours;
            findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            if (!contours.empty()) {
                auto maxContour = *max_element(contours.begin(), contours.end(),
                    [](const vector<Point>& a, const vector<Point>& b) {
                        return contourArea(a) < contourArea(b);
                    });
                
                Moments m = moments(maxContour);
                if (m.m00 > 100) {
                    Point2f center(m.m10 / m.m00, m.m01 / m.m00);
                    double time = frame_count / fps;
                    trajectoryPoints.push_back({center, time});
                }
            }
            
            frame_count++;
            if (waitKey(30) == 27) break;
        }
        
        cap.release();
        destroyAllWindows();
        return trajectoryPoints;
    }

    // 拟合弹道参数，只返回v0, g, k
    bool fitParameters(const vector<TrajectoryPoint>& trajectory, 
                      double& v0, double& g, double& k) {
        if (trajectory.size() < 5) {
            cerr << "轨迹点太少" << endl;
            return false;
        }

        double x0 = trajectory[0].position.x;
        double y0 = trajectory[0].position.y;

        double parameters[4] = {525.0, 40.0, 100.0, 0.1}; // [vx0, vy0, g, k]

        ceres::Problem problem;

        for (size_t i = 1; i < trajectory.size(); i++) {
            const auto& point = trajectory[i];
            ceres::CostFunction* cost_function =
                new ceres::AutoDiffCostFunction<BallisticCostFunction, 2, 4>(
                    new BallisticCostFunction(
                        point.position.x, point.position.y, 
                        point.time, x0, y0));
            problem.AddResidualBlock(cost_function, nullptr, parameters);
        }

        // 设置参数边界
        problem.SetParameterLowerBound(parameters, 2, 100.0);
        problem.SetParameterUpperBound(parameters, 2, 1000.0);
        problem.SetParameterLowerBound(parameters, 3, 0.01);
        problem.SetParameterUpperBound(parameters, 3, 1.0);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false; // 关闭详细输出
        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 计算初始速度大小
        double vx0 = parameters[0];
        double vy0 = parameters[1];
        v0 = sqrt(vx0 * vx0 + vy0 * vy0);
        g = parameters[2];
        k = parameters[3];

        return true;
    }
};

int main() {
    ObjectTracker tracker(60.0);
    string videoPath = "/home/caiwenbo/桌面/homework_3/video.mp4";
    
    // 提取轨迹
    vector<TrajectoryPoint> trajectory = tracker.extractTrajectory(videoPath);
    
    if (trajectory.size() > 0) {
        double v0, g, k;
        bool success = tracker.fitParameters(trajectory, v0, g, k);
        
        if (success) {
            // 只输出三个关键参数
            cout << "v0 = " << v0 << " px/s" << endl;
            cout << "g = " << g << " px/s²" << endl;
            cout << "k = " << k << " 1/s" << endl;
        }
    }

    return 0;
}
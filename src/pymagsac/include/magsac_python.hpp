#include <string>
#include <vector>
#include <opencv2/core/core.hpp>



int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&  F,
                           double sigma_th = 3.0,
                           double conf = 0.99,
                           int max_iters = 10000,
                           int partition_num = 5);

int findEssentialMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&  E,
                           std::vector<double>& intrinsics_src,
                           std::vector<double>& intrinsics_dst,
                           double sigma_th = 3.0,
                           double conf = 0.99,
                           int max_iters = 10000,
                           int partition_num = 5,
                           int core_num = 1);

                
int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    double sigma_th = 3.0,
                    double conf = 0.99,
                    int max_iters = 10000,
                    int partition_num = 5);


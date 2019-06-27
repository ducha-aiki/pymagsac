#include <string>
#include <vector>
#include <opencv2/core/core.hpp>



int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&  F,
                           double sigma_th = 1.0,
                           double sigma_max = 3.0,
                           double conf = 0.99,
                           int max_iters = 10000,
                           int partition_num = 5);


                
int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    double sigma_th = 1.0,
                    double sigma_max = 3.0,
                    double conf = 0.99,
                    int max_iters = 10000,
                    int partition_num = 5);


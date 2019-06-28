#include "magsac_python.hpp"
#include "magsac.h"
#include "fundamental_estimator.cpp"
#include "homography_estimator.cpp"
#include <thread>



int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&F,
                           double sigma_th,
                           double sigma_max,
                           double conf,
                           int max_iters,
                           int partition_num)
{
    
    FundamentalMatrixEstimator estimator; // The robust F estimator class containing the function for the fitting and residual calculation
    FundamentalMatrix model; // The estimated model
    
    MAGSAC<FundamentalMatrixEstimator, FundamentalMatrix> magsac;
    magsac.setSigmaMax(sigma_max); // The maximum noise scale sigma allowed
    magsac.setCoreNumber(4); // The number of cores used to speed up sigma-consensus
    magsac.setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac.setIterationLimit(max_iters);
    magsac.setTerminationCriterion(MAGSAC<FundamentalMatrixEstimator, FundamentalMatrix>::TerminationCriterion::RansacCriterion,
        sigma_th); // Use the standard RANSAC termination criterion since the MAGSAC one is too pessimistic and, thus, runs too long sometimes
     
    int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 6, CV_64F);
    int iterations = 0;
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = 1;
        points.at<double>(i, 3) = dstPts[2*i];
        points.at<double>(i, 4) = dstPts[2*i + 1];
        points.at<double>(i, 5) = 1;
    }
    magsac.run(points, // The data points
               conf, // The required confidence in the results
               estimator, // The used estimator
               model, // The estimated model
               iterations); // The number of iterations
    
    inliers.resize(num_tents);
    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = estimator.error(points.row(pt_idx), model) <= sigma_th;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers+=is_inlier;
    }
    
    F.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            F[i*3+j] = (float)model.descriptor.at<double>(i,j);
        }
    }
    return num_inliers;
}


int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    double sigma_th,
                    double sigma_max,
                    double conf,
                    int max_iters,
                    int partition_num)

{
    RobustHomographyEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
    Homography model; // The estimated model
    
    MAGSAC<RobustHomographyEstimator, Homography> magsac;
    magsac.setSigmaMax(sigma_max); // The maximum noise scale sigma allowed
    magsac.setCoreNumber(4); // The number of cores used to speed up sigma-consensus
    magsac.setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac.setIterationLimit(max_iters);
    magsac.setTerminationCriterion(MAGSAC<RobustHomographyEstimator, Homography>::TerminationCriterion::RansacCriterion,
        sigma_th); // Use the standard RANSAC termination criterion since the MAGSAC one is too pessimistic and, thus, runs too long sometimes
    
    int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 6, CV_64F);
    int iterations = 0;
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = 1;
        points.at<double>(i, 3) = dstPts[2*i];
        points.at<double>(i, 4) = dstPts[2*i + 1];
        points.at<double>(i, 5) = 1;
    }
    magsac.run(points, // The data points
               conf, // The required confidence in the results
               estimator, // The used estimator
               model, // The estimated model
               iterations); // The number of iterations
    
    inliers.resize(num_tents);
    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = estimator.error(points.row(pt_idx), model) <= sigma_th;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers+=is_inlier;
    }
    
    H.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            H[i*3+j] = (float)model.descriptor.at<double>(i,j);
        }
    }
    return num_inliers;
}

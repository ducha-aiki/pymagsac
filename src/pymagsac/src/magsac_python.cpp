#include "magsac_python.hpp"
#include "magsac.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "types.h"
#include "model.h"
#include "estimators.h"
#include <thread>



int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&F,
                           double sigma_max,
                           double conf,
                           int max_iters,
                           int partition_num)
{
    
    magsac::utils::DefaultFundamentalMatrixEstimator estimator(0.1); // The robust homography estimator class containing the
    gcransac::FundamentalMatrix model; // The estimated model
    
    MAGSAC<cv::Mat, magsac::utils::DefaultFundamentalMatrixEstimator> magsac;
    magsac.setMaximumThreshold(sigma_max); // The maximum noise scale sigma allowed
    //magsac.setInterruptingThreshold(sigma_th / 3.0f); // The threshold used for speeding up the procedure
    magsac.setCoreNumber(1); // The number of cores used to speed up sigma-consensus
    magsac.setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac.setIterationLimit(max_iters);
    //magsac.setTerminationCriterion(MAGSAC<FundamentalMatrixEstimator, FundamentalMatrix>::TerminationCriterion::RansacCriterion,
    //    sigma_th); // Use the standard RANSAC termination criterion since the MAGSAC one is too pessimistic and, thus, runs too long sometimes

    int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
    gcransac::sampler::UniformSampler main_sampler(&points);

    bool success =  magsac.run(points, // The data points
                              conf, // The required confidence in the results
                              estimator, // The used estimator
                              main_sampler, // The sampler used for selecting minimal samples in each iteration
                              model, // The estimated model
                              max_iters); // The number of iterations
    inliers.resize(num_tents);
    if (!success) {
        for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
            inliers[pt_idx] = false;
        }
        F.resize(9);
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                F[i*3+j] = 0;
            }
        }
        return 0;
    }
    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = estimator.residual(points.row(pt_idx), model.descriptor) <= sigma_max;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers+=is_inlier;
    }
    
    F.resize(9);
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            F[i*3+j] = (double)model.descriptor(i,j);
        }
    }
    return num_inliers;
}


int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    double sigma_max,
                    double conf,
                    int max_iters,
                    int partition_num)

{
    magsac::utils::DefaultHomographyEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
    gcransac::Homography model; // The estimated model

    MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator> magsac;
    magsac.setMaximumThreshold(sigma_max); // The maximum noise scale sigma allowed
    magsac.setCoreNumber(partition_num); // The number of cores used to speed up sigma-consensus
    magsac.setPartitionNumber(partition_num); // The number partitions used for speeding up sigma consensus. As the value grows, the algorithm become slower and, usually, more accurate.
    magsac.setIterationLimit(max_iters);
    
    int num_tents = srcPts.size()/2;
    cv::Mat points(num_tents, 4, CV_64F);
    for (int i = 0; i < num_tents; ++i) {
        points.at<double>(i, 0) = srcPts[2*i];
        points.at<double>(i, 1) = srcPts[2*i + 1];
        points.at<double>(i, 2) = dstPts[2*i];
        points.at<double>(i, 3) = dstPts[2*i + 1];
    }
    gcransac::sampler::UniformSampler main_sampler(&points);

    bool success = magsac.run(points, // The data points
                              conf, // The required confidence in the results
                              estimator, // The used estimator
                              main_sampler, // The sampler used for selecting minimal samples in each iteration
                              model, // The estimated model
                              max_iters); // The number of iterations
    inliers.resize(num_tents);
    if (!success) {
        for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
            inliers[pt_idx] = false;
        }
        H.resize(9);
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                H[i*3+j] = 0;
            }
        }
        return 0;
    }

    int num_inliers = 0;
    for (auto pt_idx = 0; pt_idx < points.rows; ++pt_idx) {
        const int is_inlier = sqrt(estimator.residual(points.row(pt_idx), model.descriptor)) <= sigma_max;
        inliers[pt_idx] = (bool)is_inlier;
        num_inliers+=is_inlier;
    }
    
    H.resize(9);
    
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            H[i*3+j] = (double)model.descriptor(i,j);
        }
    }
    return num_inliers;
}

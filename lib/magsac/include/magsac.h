#pragma once

#include <limits>
#include <opencv2/core/core.hpp>
#include <chrono>
#include <memory>
#include "model_score.h"

#ifdef _WIN32 
	#include <ppl.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

template <class ModelEstimator, class Model>
class MAGSAC  
{
public:
	enum TerminationCriterion { RansacCriterion, MagsacCriterion };

	MAGSAC() : 
		time_limit(std::numeric_limits<double>::max()),
		desired_fps(-1),
		iteration_limit(std::numeric_limits<size_t>::max()),
		maximum_sigma(10.0),
		apply_post_processing(true),
		mininum_iteration_number(50),
		partition_number(5),
		core_number(1),
		interrupting_threshold(1.0),
		last_iteration_number(0),
		log_confidence(0),
		point_number(0),
		additional_geom_check(true),
		criterion(TerminationCriterion::MagsacCriterion)
	{ 
	}

	~MAGSAC() {}

	bool run(
		const cv::Mat &points_, 
		const double confidence_,
		ModelEstimator& estimator_,
		Model &obtained_model_,  
		int &iteration_number_);

	bool scoreLess(
		const ModelScore &score_1_, 
		const ModelScore &score_2_)
	{ 
		return score_1_.J < score_2_.J; 
	}
	
	void setSigmaMax(const double maximum_sigma_) 
	{
		maximum_sigma = maximum_sigma_;
	}
	
	void setSigmaMax(const bool additional_geom_check_) 
    {
        additional_geom_check = additional_geom_check_;
    }
 
	void setReferenceThreshold(const double threshold_)
	{
		reference_inlier_outlier_threshold = threshold_;
	}

	void applyPostProcessing(bool value_) 
	{
		apply_post_processing = value_;
	}

	void setIterationLimit(size_t iteration_limit_)
	{
		iteration_limit = iteration_limit_;
	}

	void setCoreNumber(size_t core_number_)
	{
		core_number = core_number_;
	}

	void setPartitionNumber(size_t partition_number_)
	{
		partition_number = partition_number_;
	}

	void setMinimumIterationNumber(size_t mininum_iteration_number_)
	{
		mininum_iteration_number = mininum_iteration_number_;
	}

	void setTerminationCriterion(
		const TerminationCriterion criterion_,
		const double ransac_criterion_threshold_ = -1);

	void setFPS(int fps_) { desired_fps = fps_; time_limit = fps_ <= 0 ? std::numeric_limits<double>::max() : 1.0 / fps_; }

protected:
	TerminationCriterion criterion; // The termination criterion use for determining when to stop.
	double criterion_threshold; // If the user choose to use the RANSAC criterion instead of the MAGSAC one, a threshold is needed to calculate the inlier ratio
	size_t iteration_limit; // Maximum number of iterations allowed
	size_t mininum_iteration_number; // Minimum number of iteration before terminating
	double reference_inlier_outlier_threshold; // An inlier-outlier threshold to speed up the procedure by interrupting sigma-consensus if needed
	double maximum_sigma; // The maximum sigma value
	size_t core_number; // Number of core used in sigma-consensus
	double time_limit; // A time limit after the algorithm is interrupted
	int desired_fps; // The desired FPS (TODO: not tested with MAGSAC)
	bool apply_post_processing; // Decides if the post-processing step should be applied
	bool additional_geom_check; //If we want to perform some model-specific checks
	int point_number; // The current point number
	int last_iteration_number; // The iteration number implied by the last run of sigma-consensus
	double log_confidence; // The logarithm of the required confidence
	size_t partition_number; // Number of partitions used to speed up sigma-consensus
	double interrupting_threshold; // A threshold to speed up MAGSAC by interrupting the sigma-consensus procedure whenever there is no chance of being better than the previous so-far-the-best model

	bool sample(
		const cv::Mat& points_,
		std::vector<int>& pool_,
		const int sample_size_,
		int* sample_);

	void getSigmaScore(
		const cv::Mat& points_,
		Model& model_,
		const ModelEstimator& estimator_,
		double& avg_inlier_ratio_,
		double& score_);

	void postProcessing(
		const cv::Mat &points,
		const Model &so_far_the_best_model,
		Model &output_model,
		ModelScore &output_score,
		const ModelEstimator &estimator);

	void sigmaConsensus(
		const cv::Mat& points_,
		const Model& model_,
		Model& refined_model_,
		ModelScore& score_,
		const ModelEstimator& estimator_,
		const ModelScore& best_score_);
    
    bool additionalCheck(
        const cv::Mat& points_,
        const Model& model_,
        const ModelEstimator& estimator_){
        return estimator_.validModelWithData(points_,model_);
        }
        ;
};

template <class ModelEstimator, class Model>
void MAGSAC<ModelEstimator, Model>::setTerminationCriterion(
	const TerminationCriterion criterion_,
	const double ransac_criterion_threshold_)
{
	if (criterion_ == TerminationCriterion::RansacCriterion)
	{
		if (ransac_criterion_threshold_ > 0)
		{
			criterion_threshold = ransac_criterion_threshold_;
			criterion = criterion_;
		}
	} else
		criterion = criterion_;
}



template <class ModelEstimator, class Model>
bool MAGSAC<ModelEstimator, Model>::run(
	const cv::Mat& points_,
	const double confidence_,
	ModelEstimator& estimator_,
	Model& obtained_model_,
	int& iteration_number_)
{
	// Initialize variables
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measuring: start and end times
	std::chrono::duration<double> elapsed_seconds; // Variables for time measuring: elapsed time
	log_confidence = log(1.0 - confidence_); // The logarithm of 1 - confidence
	point_number = points_.rows; // Number of points
	const int sample_size = estimator_.sampleSize(); // The sample size required for the estimation
	size_t max_iteration = iteration_limit; // The maximum number of iterations initialized to the iteration limit
	int iteration = 0; // Current number of iterations
	Model so_far_the_best_model; // Current best model
	ModelScore so_far_the_best_score; // The score of the current best model
	std::unique_ptr<int[]> minimal_sample(new int[sample_size]); // The sample used for the estimation
	
	// Initializing the pool for sampling
	std::vector<int> pool(points_.rows);
	for (int i = 0; i < points_.rows; ++i)
		pool[i] = i; 

	// Set the start time variable if there is some time limit set
	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	// Main MAGSAC iteration
	while (mininum_iteration_number > iteration ||
		iteration < max_iteration)
	{
		++iteration;
		
		// Sample a minimal subset
		std::vector<Model> models;
		while (1)
		{
			// Get a minimal sample randomly
			if (!sample(points_, 
				pool, 
				sample_size, 
				minimal_sample.get()))
				continue;
			 
			// Estimate the model from the minimal sample
 			if (estimator_.estimateModel(points_, 
				minimal_sample.get(), 
				&models)) {
				if (estimator_.validModel(models.back())) break; 
			};                
		};

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			ModelScore score;
			Model refined_model;

			// Apply sigma-consensus to refine the model parameters by marginalizing over the noise level sigma
			sigmaConsensus(points_,
				model,
				refined_model,
				score,
				estimator_,
				so_far_the_best_score);

			// Continue if the model was rejected
			if (score.J == -1)
				continue;

			score.iteration = iteration;
						
			// Update the best model parameters if needed
			if (scoreLess(so_far_the_best_score, score))
			{
				so_far_the_best_model = model; // Update the best model parameters
				so_far_the_best_score = score; // Update the best model's score
				max_iteration = MIN(max_iteration, last_iteration_number); // Update the max iteration number, but do not allow to increase
			}
		}

		// Update the time parameters if a time limit is set
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			// Interrupt if the time limit is exceeded
			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}
	
	// Apply sigma-consensus as a post processing step if needed and the estimated model is valid
	if (apply_post_processing &&
		estimator_.validModel(so_far_the_best_model))
	{
		Model refined_model;
		postProcessing(points_,
			so_far_the_best_model,
			refined_model,
			so_far_the_best_score,
			estimator_);
		so_far_the_best_model = refined_model;
	}

	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;
	return estimator_.validModel(obtained_model_);
}

template <class ModelEstimator, class Model>
void MAGSAC<ModelEstimator, Model>::postProcessing(
	const cv::Mat &points_,
	const Model &model_,
	Model &refined_model_,
	ModelScore &refined_score_,
	const ModelEstimator &estimator_)
{
	// Set up the parameters
	double threshold = this->maximum_sigma;
	const size_t N = points_.rows;
	static const size_t M = estimator_.sampleSize();

	// Collect the points which are closer than the maximum threshold
	std::vector<std::pair<double, int>> all_residuals;
	all_residuals.reserve(N);
	for (auto pt_idx = 0; pt_idx < N; ++pt_idx)
	{
		double residual = estimator_.error(points_.row(pt_idx), model_);
		if (threshold > residual)
			all_residuals.emplace_back(
				std::make_pair(residual / 3.64, pt_idx));
	}

	// Number of points closer than the maximum distance
	const size_t Ni = all_residuals.size();

	// Sort the (residual, point index) pairs in ascending order
	const auto comparator = [](std::pair<double, int> left, std::pair<double, int> right) { return left.first < right.first; };

#ifdef _WIN32 
	concurrency::parallel_sort(all_residuals.begin(), all_residuals.end(), comparator);
#else
	std::sort(all_residuals.begin(), all_residuals.end(), comparator);
#endif

	// Set the threshold to be the distance of the farthest point which has lower residual than the maximum sigma
	threshold = all_residuals.back().first + 
		std::numeric_limits<double>::epsilon();

	std::vector<int> sigma_inliers;
	sigma_inliers.reserve(Ni);

	const int step_size = (Ni - M) / core_number;

	std::vector<double> final_weights(Ni, 0);
	std::vector<std::vector<double>> point_weights(core_number, 
		std::vector<double>(Ni, 0));
	const double divisions_per_process = partition_number / core_number;
	const double sigma_step = threshold / partition_number;

	for (auto process = 0; process < core_number; ++process)
	{
		const double last_sigma = (process + 1) * divisions_per_process * sigma_step;
		double prev_sigma = process * divisions_per_process * sigma_step;
		double next_sigma = prev_sigma + sigma_step;		
		double next_sigma_2 = 2 * next_sigma * next_sigma;

		int sigma_idx;
		std::vector<int> temp_sigma_inliers;
		
		for (sigma_idx = 0; sigma_idx < Ni; ++sigma_idx)
		{
			const std::pair<double, int> &next = all_residuals[sigma_idx];
			const double sigma = next.first;

			// Collecting the points while the next step is not achieved
			if (sigma < next_sigma)
			{
				temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
				continue;
			}

			// Estimating model(sigma)
			if (temp_sigma_inliers.size() > M)
			{
				std::vector<Model> model_sigma;
				estimator_.estimateModelNonminimal(points_, 
					&(temp_sigma_inliers)[0], 
					static_cast<int>(temp_sigma_inliers.size()),
					&model_sigma);

				if (model_sigma.size() > 0)
				{
					double ri, pi;
					for (auto pt_idx = 0; pt_idx < temp_sigma_inliers.size(); ++pt_idx)
					{
						const auto real_pt_idx = temp_sigma_inliers[pt_idx];
						ri = estimator_.error(points_.row(real_pt_idx), model_sigma[0].descriptor);
						pi = exp(-ri * ri / next_sigma_2);
						pi = sigma_step * pi;
						point_weights[process][pt_idx] += pi;
					}
				}
			} 
			
			// Update the next sigma
			prev_sigma = next_sigma;
			next_sigma += sigma_step;

			// Break if all divisions have been processed
			if (next_sigma > last_sigma)
				break;

			temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
			next_sigma_2 = 2 * next_sigma * next_sigma;
		}
	}

	// Collect the points which have higher than 0 probability of being inlier
	for (auto sigma_idx = 0; sigma_idx < Ni; sigma_idx += 1)
		sigma_inliers.emplace_back(all_residuals[sigma_idx].second);

	// Accumulate the sigmas
	for (auto process = 0; process < core_number; ++process)
		for (auto pt_idx = 0; pt_idx < Ni; ++pt_idx)
			final_weights[pt_idx] += point_weights[process][pt_idx];

	// Estimate the model by weighted least-squares using the posterior probabilities as weights
	std::vector<Model> sigma_models;
	estimator_.estimateModelNonminimalWeighted(points_, // All input points
		&(sigma_inliers)[0], // Points which have higher than 0 probability of being inlier
		&(final_weights)[0], // Weights of points 
		static_cast<int>(sigma_inliers.size()), // Number of possible inliers
		&sigma_models); // Estimated models

	// Update the model if needed
	if (sigma_models.size() == 1)
		refined_model_ = sigma_models[0];
	else
		refined_model_ = model_;

}


template <class ModelEstimator, class Model>
void MAGSAC<ModelEstimator, Model>::sigmaConsensus(
	const cv::Mat &points_,
	const Model& model_,
	Model& refined_model_,
	ModelScore &score_,
	const ModelEstimator &estimator_,
	const ModelScore &best_score_)
{
	// Set up the parameters
	constexpr auto L = 1.05;
	static auto cmp = [](std::pair<double, int> left, std::pair<double, int> right) { return left.first < right.first; };
	static const int sigma_density_step = 1;
	double threshold = this->maximum_sigma;

	const size_t N = points_.rows;
	const double Nd = static_cast<double>(N);
	const size_t M = estimator_.sampleSize();
	
	// Calculating the residuals
	std::vector< std::pair<double, int> > all_residuals;
	all_residuals.reserve(N);

	std::pair<double, int> tmp_pair;
	double residual_sum = 0;
	size_t points_close = 0;
	double residual;
	int pt_idx;

	// If it is not the first run, consider the previous best and interrupt the validation when there is no chance of being better
	if (best_score_.I > 0)
	{
		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (pt_idx = 0; pt_idx < N; ++pt_idx)
		{
			// Calculate the residual of the current point
			residual = estimator_.error(points_.row(pt_idx), model_);
			if (threshold > residual)
			{
				all_residuals.emplace_back(std::make_pair(residual / 3.64, pt_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}

			// Interrupt if there is no chance of being better
			// TODO: replace this part by SPRT test
			if (points_close + N - pt_idx < best_score_.I)
				return;
		}
	}
	else
	{
		// Collect the points which are closer than the threshold which the maximum sigma implies
		for (pt_idx = 0; pt_idx < N; ++pt_idx)
		{
			// Calculate the residual of the current point
			residual = estimator_.error(points_.row(pt_idx), model_);
			if (threshold > residual)
			{
				all_residuals.emplace_back(std::make_pair(residual / 3.64, pt_idx));

				// Count points which are closer than a reference threshold to speed up the procedure
				if (residual < interrupting_threshold)
					++points_close;
			}
		}
	}

	score_.I = points_close;
	const int Ni = all_residuals.size();
    
    
	//DM Additional check
	if (additional_geom_check){

	   int N_INL = static_cast<int>(all_residuals.size());
	   cv::Mat passed_points(N_INL,6,CV_64F);
	   for (int pp_idx = 0; pp_idx < N_INL; pp_idx++){
			 int idx1 = all_residuals[pp_idx].second;
			passed_points.at<double>(pp_idx,0) = points_.at<double>(idx1,0);
			passed_points.at<double>(pp_idx,1) = points_.at<double>(idx1,1);
			passed_points.at<double>(pp_idx,2) = points_.at<double>(idx1,2);
			passed_points.at<double>(pp_idx,3) = points_.at<double>(idx1,3);
			passed_points.at<double>(pp_idx,4) = points_.at<double>(idx1,4);
			passed_points.at<double>(pp_idx,5) = points_.at<double>(idx1,5);
			};
		int th = std::max(10, (int)(N_INL/2)); // We want to at least this points to survive check
		if (!estimator_.validModelWithData(passed_points, model_, th)){
		   return;        
		}
	} 
	// Sorting the distances
#ifdef _WIN32 
	concurrency::parallel_sort(all_residuals.begin(), all_residuals.end(), cmp);
#else
	std::sort(all_residuals.begin(), all_residuals.end(), cmp);
#endif
	threshold = all_residuals.back().first + FLT_EPSILON;

	int sigma_idx;
	std::vector<int> sigma_inliers;
	sigma_inliers.reserve(Ni);

	const int step_size = (Ni - M) / core_number;
	int division_number = 5;
	int divisions_per_process = division_number / core_number;
	double sigma_step = threshold / division_number;

	last_iteration_number = 10000;

	double iterations = 0;

	score_.J = 0;

	std::vector<double> final_weights(Ni, 0);
	std::vector<std::vector<double>> point_weights_par(core_number, std::vector<double>(Ni, 0));
	std::vector<double> output_score_par(core_number, 0);
	std::vector<double> iterations_par(core_number, 0);
	
	// Process the sigma divisions in parallel
#ifdef _WIN32 
	concurrency::parallel_for(0, static_cast<int>(core_number), [&](int process)
#else
#pragma omp parallel for schedule (dynamic,1)
	for (auto process = 0; process < core_number; ++process)
#endif
	{
		const double last_sigma = (process + 1) * divisions_per_process * sigma_step;
		double prev_sigma = process * divisions_per_process * sigma_step;
		double next_sigma = prev_sigma + sigma_step;
		double next_sigma_2 = 2 * next_sigma * next_sigma;

		std::vector<int> temp_sigma_inliers;
		double ri, pi;

		for (auto sigma_idx = 0; sigma_idx < Ni; ++sigma_idx)
		{
			const std::pair<double, int> &next = all_residuals[sigma_idx];
			const double sigma = next.first;

			if (sigma < next_sigma)
			{
				temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
				continue;
			}

			if (temp_sigma_inliers.size() > M)
			{
				// Estimating model(sigma)
				std::vector<Model> sigma_models;
				estimator_.estimateModelNonminimal(points_, 
					&(temp_sigma_inliers)[0], 
					static_cast<int>(temp_sigma_inliers.size()),
					&sigma_models);

				// If the estimation was successful calculate the implied probabilities
				if (sigma_models.size() == 1)
				{
					for (auto pt_idx = 0; pt_idx < temp_sigma_inliers.size(); ++pt_idx)
					{
						// TODO: Replace with Chi-square instead of normal distribution
						const auto &real_pt_idx = temp_sigma_inliers[pt_idx];
						ri = estimator_.error(points_.row(real_pt_idx), sigma_models[0]);
						pi = sigma_step * exp(-ri * ri / next_sigma_2);
						point_weights_par[process][pt_idx] += pi;
					}
				}
			}
						
			// Update the next
			prev_sigma = next_sigma;
			next_sigma += sigma_step;
			if (next_sigma > last_sigma)
				break;

			temp_sigma_inliers.emplace_back(all_residuals[sigma_idx].second);
			next_sigma_2 = 2 * next_sigma * next_sigma;
		}
	}
#ifdef _WIN32 
	);
#endif

	// Collect all points which has higher probability of being inlier than zero
	for (sigma_idx = 0; sigma_idx < Ni; sigma_idx += sigma_density_step)
		sigma_inliers.emplace_back(all_residuals[sigma_idx].second);

	// Accumulate the results of each thread
	for (auto process = 0; process < core_number; ++process)
		for (auto pt_idx = 0; pt_idx < Ni; ++pt_idx)
			final_weights[pt_idx] += point_weights_par[process][pt_idx];

	// If there are fewer inliers than the size of the minimal sample interupt the procedure
	if (sigma_inliers.size() < M)
		return;

	std::vector<Model> sigma_models;
	if (!estimator_.estimateModelNonminimalWeighted(points_, // All the input points
		&(sigma_inliers)[0], // Points which has higher probability of being inlier than zero
		&(final_weights)[0], // The corresponding weights
		static_cast<int>(sigma_inliers.size()), // The number of used points
		&sigma_models)) // The estimated models
		return;
    
    
    //

	if (sigma_models.size() == 1 && // If only a single model is estimated
		estimator_.validModel(sigma_models[0])) // and it is valid
	{

		// Calculate the score of the model and the implied iteration number
		double marginalized_iteration_number;
		getSigmaScore(points_, // All the input points
			sigma_models[0], // The estimated model
			estimator_, // The estimator
			marginalized_iteration_number, // The marginalized inlier ratio
			score_.J); // The marginalized score
		
		// Use the proposed MAGSAC criterion which marginalizes over the possible sigma values
		// Note: it will most probably be a higher value than what the standard RANSAC criterion
		// implies. The standard RANSAC criterion assumes no noise and, thus, is too optimistic
		// in practice. 
		if (criterion == TerminationCriterion::MagsacCriterion)
		{
			iterations = marginalized_iteration_number; 
			if (iterations < 0 || std::isnan(iterations))
				iterations = 1e5;
			last_iteration_number = static_cast<int>(round(iterations)); 
		} else // Use the RANSAC criterion if needed
		{
			const double criterion_threshold_mod = criterion_threshold / 3.64; // Modify the threshold since the residuals are modified as well
			int inlier_number = 0;			
			for (const auto &residual : all_residuals)
			{		
				if (residual.first < criterion_threshold_mod)
					++inlier_number;
				else // The residuals are in increasing order. Therefore, we can break if is is greater than the threshold
					break;
			}
			// Calculate the RANSAC termination criterion
			last_iteration_number = log_confidence / log(1.0 - pow(inlier_number / Nd, M));
			if (last_iteration_number < 0)
				last_iteration_number = 1e5;
		}
		refined_model_ = sigma_models[0];
	}
}

template <class ModelEstimator, class Model>
void MAGSAC<ModelEstimator, Model>::getSigmaScore(
	const cv::Mat &points_,
	Model &model_,
	const ModelEstimator &estimator_,
	double &marginalized_iteration_number_, 
	double &score_)
{
	// Set up the parameters
	const auto threshold = this->maximum_sigma;
	const auto N = points_.rows;
	const auto Nf = static_cast<double>(N);
	const auto M = estimator_.sampleSize();

	// Getting the inliers
	std::vector<std::pair<double, int>> all_residuals;
	all_residuals.reserve(N);

	std::pair<double, int> tmp_pair;
	double max_distance = 0;
	for (auto pt_idx = 0; pt_idx < N; ++pt_idx)
	{
		double residual = estimator_.errorForScoring(points_.row(pt_idx), model_.descriptor);
		if (threshold > residual)
		{
			max_distance = MAX(max_distance, residual);
			all_residuals.emplace_back(std::make_pair(residual, pt_idx));
		}
	}

	max_distance = max_distance + 
		std::numeric_limits<double>::epsilon();

	const int Ni = all_residuals.size();
	const double threshold_step = max_distance / partition_number;
	std::vector<double> thresholds(partition_number - 1);
	std::vector<double> thresholds_sqr(partition_number - 1);
	std::vector<double> thresholds_2_sqr(partition_number - 1);
	for (auto i = 0; i < partition_number - 1; ++i)
	{
		thresholds[i] = (i + 1) * threshold_step;
		thresholds_sqr[i] = thresholds[i] * thresholds[i];
		thresholds_2_sqr[i] = 2 * thresholds_sqr[i];
	}

	marginalized_iteration_number_ = Ni / Nf;
	score_ = Ni;

	double ri, ri2, p;
	std::vector<double> inliers(partition_number - 1, 0);
	std::vector<double> probabilities(partition_number - 1, 1);
	for (auto pt_idx = 0; pt_idx < Ni; ++pt_idx)
	{
		ri = all_residuals[pt_idx].first;
		ri2 = ri * ri;

		for (int i = 0; i < partition_number - 1; ++i)
		{
			if (ri < thresholds[i])
			{
				inliers[i] += 1 - ri2 / thresholds_sqr[i];
				p = exp(-ri2 / thresholds_2_sqr[i]);
				probabilities[i] += p;
			}
		}
	}

	score_ = 0;
	for (auto i = 0; i < partition_number - 1; ++i)
	{
		score_ += probabilities[i];
		marginalized_iteration_number_ = (marginalized_iteration_number_ + log_confidence / log(1 - pow(inliers[i] / Nf, M)));
	}
	marginalized_iteration_number_ = marginalized_iteration_number_ / (partition_number - 1);
}

template <class ModelEstimator, class Model>
bool MAGSAC<ModelEstimator, Model>::sample(
	const cv::Mat& points_,
	std::vector<int>& pool_,
	const int sample_size_,
	int* sample_)
{
	if (pool_.size() < sample_size_)
		return false;

	// TODO: Replace with a smarter sampler, e.g. PROSAC
	for (int i = 0; i < sample_size_; ++i)
	{
		const int idx = (pool_.size() - 1) * static_cast<double>(rand()) / RAND_MAX;
		sample_[i] = pool_[idx];
		pool_.erase(pool_.begin() + idx);
	}

	pool_.reserve(pool_.size() + sample_size_);
	for (int i = 0; i < sample_size_; ++i)
		pool_.emplace_back(sample_[i]);
	return true;
}

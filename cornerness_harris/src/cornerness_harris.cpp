#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // visualize results
    /*string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);*/

    vector<cv::KeyPoint> keypoints;
    float overlap = 0.0;
    for(size_t i = 0; i < dst_norm.rows; i++)
    {
        for(size_t j = 0 ; j < dst_norm.cols; j++)
        {
            int matrix_val = (int)dst_norm.at<float>(i,j);
            if(matrix_val > minResponse)
            {
                cv::KeyPoint curr_KP;
                curr_KP.pt = cv::Point(j,i);
                curr_KP.size = 2*apertureSize;
                curr_KP.response = matrix_val;

                bool b_overlap = false;
                for(auto it = keypoints.begin(); it!=keypoints.end(); ++it) //compare other KPs to current KP
                {
                    float kpt_overlap = cv::KeyPoint::overlap(curr_KP,*it);
                    if(kpt_overlap > overlap)
                    {
                       b_overlap = true;
                       if(curr_KP.response > (*it).response){
                           *it = curr_KP; //replace old KP with the current one
                           break;
                       }
                    }
                }
                if(!b_overlap)
                {
                    keypoints.emplace_back(curr_KP); // only add the KP if no overlap was found
                }
            }
        }
    }
    string win_name = "Harris Corner Detection Results";
    cv::namedWindow(win_name, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(win_name, visImage);
    cv::waitKey(0);

}

int main()
{
    cornernessHarris();
}
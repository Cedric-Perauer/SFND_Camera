#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;




void gradientSobel()
{
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};
    int s_arr = sizeof(gauss_data)/ sizeof(gauss_data[0]);
    for(int i = 0; i < s_arr; i++)
    {
        gauss_data[i] = gauss_data[i]/273;
    }
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, gauss_data);

    // apply filter
    cv::Mat result;
    cv::filter2D(img, result, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    float sobel_x[9] = {-1, 0, +1,
                        -2, 0, +2,
                        -1, 0, +1};
    cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);

    // apply filter
    cv::Mat result_x;
    cv::filter2D(result, result_x, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);


    float sobel_y[9] = {-1, -2, -1,
                        0, 0, 0,
                        +1, +2, +1};
    cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

    // apply filter
    cv::Mat result_y;
    cv::filter2D(result, result_y, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat result_x_y;
    cv::add(result_x,result_y,result_x_y);

    // show result
    string windowName = "Sobel operator (x-direction)";
    cv::namedWindow( windowName, 1 ); // create window
    cv::imshow(windowName, result_x_y);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    gradientSobel();
}
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
//#include "opencv2/highgui/highgui.hpp"
#include "raspi_nodes/LaneFollowerMsg.h"
#include "sensor_msgs/CompressedImage.h"
#include "std_msgs/Float32.h"
#include "std_msgs/Int16.h"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <utility>

#define K_P     50.0
#define SPEED   3000

int counter = 0;
char str[500];

void img_rcv_cb(const sensor_msgs::CompressedImage::ConstPtr& msg);
void find_nonzero(cv::Mat &binaryImg, int row, std::vector<int> &indices);
int get_lane_mid(cv::Mat &binaryImg, int row, std::vector<int> &indices, int &middle);

ros::Publisher steering_pub, throttle_pub;

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "racer");
    ros::NodeHandle n;
    // pub = n.advertise<raspi_nodes::LaneFollowerMsg>("/lane_follower/output", 1);
    steering_pub = n.advertise<std_msgs::Float32>("/motor_control/steering", 1);
    throttle_pub = n.advertise<std_msgs::Int16>("/motor_control/throttle", 1);
    ros::Subscriber image_sub = n.subscribe("/camera/image/compressed", 1, img_rcv_cb);
    ROS_INFO("Message callback registered. Start working.");
    ros::spin();
    ROS_INFO("Shutting down");
}


void img_rcv_cb(const sensor_msgs::CompressedImage::ConstPtr& msg)
{
    static int nrLostTrack = 0;
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat cropped;
    cv::Mat edge;
    std::vector<int> cols;
    float distError = 0.0, angleError = 0.0;
    static float lastAngleError = 0.0;
    int lowerMid, upperMid;
    static float last_steering;
    std_msgs::Float32 steering_msg;
    std_msgs::Int16 throttle_msg;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    cropped = cv_ptr->image(cv::Rect(0, 40, 80, 20));
    //sprintf(str, "img_og%04d.jpg", counter);
    //cv::imwrite(str, cropped);
    //cv::imshow("bla2", cropped);
    //cv::waitKey(1);
    cv::threshold(cropped, cropped, 210, 255, 0);
    //sprintf(str, "img_th%04d.jpg", counter);
    //cv::imwrite(str, cropped);
    //cv::imshow("bla", cropped);
    //cv::waitKey(1);
    cv::Canny(cropped, edge, 100, 150);
    //sprintf(str, "img_ed%04d.jpg", counter);
    //cv::imwrite(str, cropped);
    //cv::Sobel(cv_ptr->image, edge, CV_16S, 1, 0);
    
    /////////////////////////////// DEBUG //////////////////////////////////////
    cv::Mat draw;
    edge.convertTo(draw, CV_8U);
    
    get_lane_mid(draw, 3*draw.rows/4, cols, lowerMid);
    get_lane_mid(draw, draw.rows/4, cols, upperMid);
    angleError = atan(2.0*(upperMid-lowerMid)/draw.rows)/M_PI*180;
    
    steering_msg.data = (lowerMid - draw.cols/2.0)/(float)draw.cols*K_P;
    if (lowerMid && upperMid)
    {
        lastAngleError = angleError;
    }
    if (lowerMid)
    {
        nrLostTrack = 0;
        throttle_msg.data = SPEED;
    }
    else
    {
        nrLostTrack++;
        if (nrLostTrack > 500)
        {
            throttle_msg.data = 0;
            nrLostTrack = 501;
        }
        else
        {
            throttle_msg.data = SPEED;
            if (lastAngleError > 0)
                steering_msg.data = 40.0;
            else
                steering_msg.data = -40;
        }
    }
    
    steering_pub.publish(steering_msg);
    throttle_pub.publish(throttle_msg);
    last_steering = steering_msg.data;
    
    
    
    
    
    /*
    std::cout << cols.size();
    for (int i = 0; i<cols.size(); i++)
        std::cout << cols[i] << ' ';
    std::cout << "\n";
    */
    
    //cv::imshow("image", draw);
    //cv::waitKey(1);
    counter++;
    
}

void find_nonzero(cv::Mat &binaryImg, int row, std::vector<int> &indices)
{
    indices.clear();
    for(int col=0; col<binaryImg.cols; col++)
    {
        // std::cout << "content at " << row << "," << col << ": " << (int)binaryImg.at<uint8_t>(row, col) << "\n";
        if ((int)binaryImg.at<uchar>(row, col) > 0)
        {
            // std::cout << "saving index\n";
            indices.push_back(col);
        }
    }
}

int get_lane_mid(cv::Mat &binaryImg, int row, std::vector<int> &indices, int &middle)
{
    find_nonzero(binaryImg, row, indices);
    middle = 0;
    for (int i=0; i<indices.size(); i++)
    {
        middle += indices[i];
    }
    if (!middle)
    {
        return 0;
    }
    middle /= indices.size();
    return 1;
    
}

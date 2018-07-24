#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace cv;

#define MIN_AVAILABLE 5.0

#define MIN_AREA 2000.0

const double thresh = 100;

void getContour(const Mat &src, std::vector<Rect> &boundRect) {
    Mat scene_gray, canny_output;
    cvtColor(src, scene_gray, CV_BGR2GRAY);

    Canny( scene_gray, canny_output, thresh, thresh*3);

    std::vector<std::vector<Point> > contours;
    findContours( canny_output, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //std::vector<std::vector<Point> > contours_poly( contours.size() );

    //std::vector<Point2f>center( contours.size() );
    //std::vector<float>radius( contours.size() );
    boundRect.clear();
    for( int i = 0; i < contours.size(); i++ )
    {
        //approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect.push_back(boundingRect( Mat(contours[i]) ));
        //minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }


    /// 画多边形轮廓 + 包围的矩形框 + 圆形框
    //Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    /*Mat drawing = src.clone();
    for( int i = 0; i< contours.size(); i++ )
    {
        //std::cout << boundRect[i].size().area() << std::endl;
        if (boundRect[i].size().area() < MIN_AREA) continue;
        Scalar color = Scalar(255,255,255);
        //drawContours( drawing, contours_poly, i, color, 1, 8, std::vector<Vec4i>(), 0, Point() );
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }


    imwrite("test1.jpg", drawing);*/
}

std::vector<KeyPoint> getGoodMatch(const Mat &object, const Mat &scene, std::vector< DMatch > &good_matches) {
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    Ptr<ORB> detector=ORB::create(minHessian);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector->detect( object, keypoints_object );
    detector->detect( scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    //cv::xfeatures2d::SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    detector->compute( object, keypoints_object, descriptors_object );
    detector->compute( scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    good_matches.clear();

    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
    }

    Mat img_matches;
    drawMatches( object, keypoints_object, scene, keypoints_scene,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imshow( "match", img_matches );

    return keypoints_scene;
}

int main()
{
    Mat img_object = imread("./object.jpg");
    resize(img_object, img_object, Size(500,500));
    imshow( "object", img_object );
    Mat img_scene = imread("./scene.jpg");
    resize(img_scene, img_scene, Size(1000,1000));
    GaussianBlur(img_scene, img_scene , Size(3,3), 0);


    std::vector<Rect> boundRect;
    getContour(img_scene, boundRect);

    std::vector< DMatch > good_matches;
    std::vector<KeyPoint> keyPoints_scene;
    keyPoints_scene = getGoodMatch(img_object, img_scene, good_matches);

    //std::cout << boundRect.size() << " " << good_matches.size() << std::endl;


    int max_count, max;
    max_count = 0;
    for (int i=0;i<boundRect.size();++i) {
        int count = 0;
        for (int j=0;j<good_matches.size();++j) {
            if (boundRect[i].contains(keyPoints_scene[good_matches[j].trainIdx].pt)) count++;
        }
        if (count > max_count) {
            max = i;
            max_count = count;
        }
    }

    if (max_count > good_matches.size() / 2) {
        rectangle( img_scene, boundRect[max].tl(), boundRect[max].br(), Scalar(255,255,255), 2, 8, 0 );
        std::cout<< "detected!" << std::endl;
    }
    else std::cout<< "not detected!" << std::endl;

    imshow( "scene", img_scene );

    /*//-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;


    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography( obj, scene, CV_RANSAC );



    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);




    if (getDistance(scene_corners[0], scene_corners[2]) < MIN_AVAILABLE) {
        std::cout << "CAN' FIND MATCH!" << std::endl;
        waitKey(0);
        return 0;
    }

    Mat img_match = img_scene.clone();
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_match, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
    line( img_match, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
    line( img_match, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
    line( img_match, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    imshow( "detected", img_match );*/

    waitKey(0);
    return 0;
}

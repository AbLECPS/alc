#include "ros/ros.h"
#include <iostream>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <rtreach/stamped_ttc.h>
#include <rtreach/reach_tube.h>
#include <rtreach/interval.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <ros/console.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>
#include <cstdlib>

// The following node will receive messages from the LEC which will be prediction of the steering angle
// It will also receive messages from the speed node which will dictate how fast the car should travel
// Initially the assmumption will be the the car moves at constant velocity


// The boxes from the urdf are 10 by 10 by 10


const int max_hyper_rectangles = 8000;

extern "C"
{ 
     #include "simulate_obstacle.h"
    HyperRectangle runReachability_obstacle_vis(double* start, double simTime, double wallTimeMs, double startMs,double v_x, double v_y, HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
    double getSimulatedSafeTime(double start[2],double v_x, double v_y);
    HyperRectangle hr_list[max_hyper_rectangles];
}

// ros::Publisher ackermann_pub; 
ros::Publisher vis_pub;
ros::Publisher tube_pub;
ros::Subscriber sub; // markerArray subscriber
 

// reachability parameters
double sim_time;
double walltime; // 25 ms corresponds to 40 hz 
bool bloat_reachset = true;

int rect_count;
int count = 0;
double startState[2] = {0.0, 0.0};

double display_max;
int display_count = 1;
double display_increment = 1.0;
double box_size;

nav_msgs::Odometry::ConstPtr msg;


void box_pose_callback(const nav_msgs::Odometry::ConstPtr& nav_msg)
{
    
    msg = nav_msg;
    count++;
}





void timer_callback(const ros::TimerEvent& event)
{
  using std::cout;
  using std::endl;

  double roll, pitch, yaw;
  double x,y,vx,vy;
  
  sim_time = 2.0;
  rect_count = 0;
  HyperRectangle hull;
  
  if(count>0)
  {

    // position and velocity
    x = msg-> pose.pose.position.x;
    y = msg-> pose.pose.position.y;
    vx = msg->twist.twist.linear.x;
    vy = msg->twist.twist.linear.y;

    //cout << "x: " << x << " " << "y: " << y << " z: " << msg-> pose.pose.position.z << endl;
    //cout << "vx: " << vx;
    //cout << "vy: " << vy;

    startState[0] = x;
    startState[1] = y;
    

    HyperRectangle reach_hull = runReachability_obstacle_vis(startState, sim_time, walltime, 0,vx,vy,hr_list,&rect_count,max_hyper_rectangles,true);


    // define the quaternion matrix
    tf::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q);

    // convert to rpy
    m.getRPY(roll, pitch, yaw);

    // printf("num_boxes: %d, \n",rect_count);
    visualization_msgs::MarkerArray ma;
    rtreach::reach_tube reach_set;
    display_increment = rect_count  / display_max;
    // display_count = std::max(1.0,nearbyint(display_increment));
    // cout <<  "display_max: " << display_increment  << ", display count: " << display_count << endl;

    for(int i= 0; i<std::min(max_hyper_rectangles,rect_count-1); i+=display_count)
    {
        
        hull = hr_list[i];
        if(bloat_reachset)
        {
            hull.dims[0].min = hull.dims[0].min  - box_size;
            hull.dims[0].max = hull.dims[0].max  + box_size;
            hull.dims[1].min = hull.dims[1].min  - box_size;
            hull.dims[1].max = hull.dims[1].max  + box_size;
        }
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = (hull.dims[0].max+hull.dims[0].min)/2.0;
        marker.pose.position.y = (hull.dims[1].max+hull.dims[1].min)/2.0;
            
        marker.pose.position.z = msg->pose.pose.position.z;
        marker.pose.orientation.x = msg->pose.pose.orientation.x;
        marker.pose.orientation.y = msg->pose.pose.orientation.y;
        marker.pose.orientation.z = msg->pose.pose.orientation.z;
        marker.pose.orientation.w = msg->pose.pose.orientation.w;
        marker.scale.x = (hull.dims[0].max-hull.dims[0].min);
        marker.scale.y = (hull.dims[1].max-hull.dims[1].min);
        marker.scale.z = 0.05;
        marker.color.a = 1.0; 
        marker.color.r = (double) rand() / (RAND_MAX);
        marker.color.g = (double) rand() / (RAND_MAX);
        marker.color.b = (double) rand() / (RAND_MAX);
        marker.lifetime =ros::Duration(0.5); 
        ma.markers.push_back(marker);
    }

    // publish marker
    vis_pub.publish(ma);
    for(int i= 0; i<std::min(max_hyper_rectangles,rect_count-1); i++)
    {
       hull = hr_list[i];
       if(bloat_reachset)
       {
            hull.dims[0].min = hull.dims[0].min  - box_size;
            hull.dims[0].max = hull.dims[0].max  + box_size;
            hull.dims[1].min = hull.dims[1].min  - box_size;
            hull.dims[1].max = hull.dims[1].max  + box_size;
       }
       rtreach::interval tube;
       tube.x_min = hull.dims[0].min;
       tube.x_max = hull.dims[0].max;
       tube.y_min = hull.dims[1].min;
       tube.y_max = hull.dims[1].max;
       reach_set.obstacle_list.push_back(tube);
    }
     
    reach_set.header.stamp = ros::Time::now();
    reach_set.count = reach_set.obstacle_list.size();
    tube_pub.publish(reach_set);
  }
  
  


  else
  {
      // visualization_msgs::MarkerArray ma;
      rtreach::reach_tube reach_set;
      reach_set.header.stamp = ros::Time::now();
      reach_set.count = 0;
      // vis_pub.publish(ma);
      tube_pub.publish(reach_set);
  }
  
}

int main(int argc, char **argv)
{
    using namespace message_filters;



    if(argv[1] == NULL)
    {
        std::cout << "Please the name of the box for which you would like to compute reachsets for e.g (box1)" << std::endl;
        exit(0);
    }

    if(argv[2] == NULL)
    {
        std::cout << "Please enter the box size e.g 10" << std::endl;
        exit(0);
    }

    if(argv[3] == NULL)
    {
        std::cout << "Please enter the sim_time e.g 2" << std::endl;
        exit(0);
    }

    if(argv[4] == NULL)
    {
        std::cout << "Please enter the wall time e.g 1" << std::endl;
        exit(0);
    }

    if(argv[5] == NULL)
    {
        std::cout << "Please enter the number of boxes to display e.g 100" << std::endl;
        exit(0);
    }

    std::string obs_name= argv[1];
    box_size = atof(argv[2]) / 2.0;
    walltime = atoi(argv[3]);
    sim_time = atof(argv[4]);
    display_max = atof(argv[5]);


    // initialize the ros node
    ros::init(argc, argv, "visualize_node_obstacle");

    ros::NodeHandle n;
    
    vis_pub = n.advertise<visualization_msgs::MarkerArray>(obs_name+"/reach_hull_obs", 100 );
    tube_pub = n.advertise<rtreach::reach_tube>(obs_name+"/reach_tube",100);
    
  
    sub = n.subscribe(obs_name+"/pose_gt", 1000, box_pose_callback);
    ros::Timer timer = n.createTimer(ros::Duration(0.01), timer_callback);

    ros::Rate r(80);
    while(ros::ok())
    {
      r.sleep();
      ros::spinOnce();
    }
    // de-allocate obstacles
    return 0; 
}

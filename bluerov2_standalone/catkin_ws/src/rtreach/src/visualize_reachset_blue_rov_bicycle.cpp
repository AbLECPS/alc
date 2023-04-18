#include "ros/ros.h"
#include <iostream>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <vandy_bluerov/HSDCommand.h>
#include <message_filters/subscriber.h>
#include <rtreach/stamped_ttc.h>
#include <rtreach/obstacle_list.h>
#include <message_filters/synchronizer.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <ros/console.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>

// The following node will receive messages from the LEC which will be prediction of the steering angle
// It will also receive messages from the speed node which will dictate how fast the car should travel
// Initially the assmumption will be the the car moves at constant velocity


const int max_hyper_rectangles = 500;


extern "C"
{ 
    #include "bicycle_model_parametrizeable.h"
    bool runReachability_bicycle_dyn(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
    HyperRectangle hr_list2[max_hyper_rectangles];
    void println(HyperRectangle * r);
    HyperRectangle hull;
    #include "bicycle_model_degraded.h"
    bool runReachability_bicycle_degraded(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);

    #include "bicycle_model_disturbance.h"
    bool runReachability_bicycle_disturbance(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot, REAL disturbance);
}

int count = 0;
int degredation_count = 0;
int rect_count = 0;
float disturbance = 1.0;

// ros::Publisher ackermann_pub; 
ros::Publisher vis_pub;
ros::Subscriber sub; // markerArray subscriber
ros::Subscriber sub2;
ros::Subscriber sub3;
ros::ServiceClient client;


// reachability parameters
double sim_time;
double state[4] = {0.0,0.0,0.0,0.0};
double control_input[2] = {0.0,0.0};
double pi1 = M_PI;
double walltime; // 25 ms corresponds to 40 hz 
bool bloat_reachset = true;
bool log_output_to_console = false;
bool degraded = false;


// We don't want to output all the rectangles 
double display_max = 100;
int display_count = 1;
double display_increment = 1.0;

vandy_bluerov::HSDCommand::ConstPtr hsd_msg;
void hsd_callback(const vandy_bluerov::HSDCommand::ConstPtr& msg)
{
    hsd_msg = msg;
    count++;
}

// Degredation Callback
void degredation_callback(const std_msgs::Float32MultiArray::ConstPtr&  msg)
{
    degraded = true;
    // forward projection should be smaller when the system is degraded
    walltime = 0.5;
}

double clip(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}

void callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  using std::cout;
  using std::endl;

  double roll, pitch, yaw, lin_speed;
  double x,y,uh,uv,xtemp,ytemp;
  HyperRectangle hull;

  if(log_output_to_console)
  { 
    if(not degraded)
    {
      ROS_INFO("Using Nominal System Identification Model");
    }
    else
    {
      ROS_INFO("Use Degredaded System Identification Model");
    }
  }
  if(log_output_to_console)
  {
    std::cout << "sim_time: " << sim_time << endl;
  }
  if(count>0)
  {
    x = msg-> pose.pose.position.x;
    y = msg-> pose.pose.position.y;

    // define the quaternion matrix
    tf::Quaternion q(
          msg->pose.pose.orientation.x,
          msg->pose.pose.orientation.y,
          msg->pose.pose.orientation.z,
          msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q);

    // normalize the speed 
    tf::Vector3 speed = tf::Vector3(msg->twist.twist.linear.x, msg->twist.twist.linear.y, 0.0);
    // lin_speed = speed.length();
    lin_speed = hsd_msg->speed;

    // convert to rpy
    m.getRPY(roll, pitch, yaw);
    if(log_output_to_console)
    {
      cout << "x: " << x << " " << "y: " << y << " z: " << msg-> pose.pose.position.z << endl;
      cout << "roll: " << roll;
      cout << "pitch: " << pitch;
      cout << "yaw: " << yaw;
    }

    // control commands from hsd controller
    uh = -(clip(hsd_msg->heading,-30,30) * pi1 ) / 180;
    if(degraded)
    {
      uh = -(clip(hsd_msg->heading,-3,3) * pi1 ) / 180;
    }
    uv = hsd_msg->speed;
    
    if(log_output_to_console)
    {
      cout << "uh: " << uh;
      cout << "uv: " << uv << endl;
    }

    state[0] = x;
    state[1] = y;
    state[2] = lin_speed;
    state[3] = yaw;
    ROS_INFO("disturbance %f",disturbance);

    // create the ros message that will be sent to the VESC
    if(true)
    {
      if (not degraded)
      {
            runReachability_bicycle_disturbance(state, sim_time, walltime, 0,uh,uv,hr_list2,&rect_count,max_hyper_rectangles,false,disturbance);
      }
      else
      {
        runReachability_bicycle_degraded(state, sim_time, walltime, 0,uh,uv,hr_list2,&rect_count,max_hyper_rectangles,false);
      }

      
      if(log_output_to_console)
      {
        printf("num_boxes: %d, \n",rect_count);
        hull = hr_list2[std::min(rect_count-1,max_hyper_rectangles-1)];
        ROS_INFO("vis last rect [%f,%f], [%f,%f]",hull.dims[0].min,hull.dims[0].max,hull.dims[1].min,hull.dims[1].max);
      }
      visualization_msgs::MarkerArray ma;

      display_increment = rect_count/ display_max;
      display_count = std::max(1.0,nearbyint(display_increment));

      if(degraded)
      {
        display_count = 1;
      }
      for(int i = 0; i<rect_count;i+=display_count)
      {
        HyperRectangle hull = hr_list2[i];

        // if we want to bloat the hyper-rectangles for the width of the car
        if(bloat_reachset)
        {
          hull.dims[0].min = hull.dims[0].min  - 0.2285;
          hull.dims[0].max = hull.dims[0].max  + 0.2285;
          hull.dims[1].min = hull.dims[1].min  - 0.169;
          hull.dims[1].max = hull.dims[1].max  + 0.169;
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

  
        marker.pose.orientation.x =  msg->pose.pose.orientation.x;
        marker.pose.orientation.y = msg->pose.pose.orientation.y;
        marker.pose.orientation.z = msg->pose.pose.orientation.z;
        marker.pose.orientation.w = msg->pose.pose.orientation.w;
        marker.scale.x = (hull.dims[0].max-hull.dims[0].min);
        marker.scale.y = (hull.dims[1].max-hull.dims[1].min);
        marker.scale.z = 0.05;
        marker.color.a = 1.0; 
        if(degraded)
        {
          marker.color.r = 1.0;
          marker.color.g = 0.0;
          marker.color.b = 0.0;;
        }
        else if(i % 2 == 1)
        {
          marker.color.r = 0.0;           //(double) rand() / (RAND_MAX);
          marker.color.g = 0.0; //(double) rand() / (RAND_MAX);
          marker.color.b = 1.0;    
        }
        else
        {
          marker.color.r = 0.0;           //(double) rand() / (RAND_MAX);
          marker.color.g = 239.0 / 255.0; //(double) rand() / (RAND_MAX);
          marker.color.b = 1.0;               //(double) rand() / (RAND_MAX);
        }
       
        marker.lifetime =ros::Duration(0.1); 
        ma.markers.push_back(marker);
      }

      // publish marker
      vis_pub.publish( ma );
    }
  }
}

int main(int argc, char **argv)
{
    using namespace message_filters;

    // initialize the ros node
    ros::init(argc, argv, "visualize_node_rtreach",ros::init_options::AnonymousName);

    ros::NodeHandle n;

    if(argv[1] == NULL)
    {
        std::cout << "please give the walltime 10" << std::endl;
        exit(0);
    }

    if(argv[2] == NULL)
    {
        std::cout << "please give the sim time (e.g) 2" << std::endl;
        exit(0);
    }

    if(argv[3] == NULL)
    {
       disturbance = 0.75;
    }
    else
    {
      disturbance = atof(argv[3]);
    }

    //disturbance = 0.75;

    walltime = atoi(argv[1]);
    sim_time = atof(argv[2]);

    vis_pub = n.advertise<visualization_msgs::MarkerArray>( "reach_hull", 100 );
  
    sub = n.subscribe("uuv0/hsd_command", 1000, hsd_callback);
    sub2 = n.subscribe("uuv0/pose_gt_ned", 1000, callback);
    sub3 = n.subscribe("uuv0/thruster_reallocation",100, degredation_callback);

    ros::Rate r(80);
    while(ros::ok())
    {
      ros::spinOnce();
    }


    return 0; 
}

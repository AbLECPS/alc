#include "ros/ros.h"
#include <iostream>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <vandy_bluerov/HSDCommand.h>
#include <message_filters/subscriber.h>
#include <rtreach/stamped_ttc.h>
#include <rtreach/obstacle_list.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <ros/console.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>

// The following node will receive messages from the LEC which will be prediction of the steering angle
// It will also receive messages from the speed node which will dictate how fast the car should travel
// Initially the assmumption will be the the car moves at constant velocity

extern "C"
{ 
     #include "uuv_safety.h"
     #include "simulate_uuv_plots.h"
     HyperRectangle runReachability_uuv_vis(REAL* start, REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL speed);
     HyperRectangle hull;
     void println(HyperRectangle * r);
     void allocate_obstacles(int num_obstacles);
     void deallocate_obstacles();
     void append_obstacle(double (*box)[2]);
}

// ros::Publisher ackermann_pub; 
ros::Publisher vis_pub;
ros::Subscriber sub; // markerArray subscriber
ros::Subscriber sub2;
ros::ServiceClient client;

// reachability parameters
double sim_time;
double walltime; // 25 ms corresponds to 40 hz 
bool bloat_reachset = true;

int count = 0;
vandy_bluerov::HSDCommand::ConstPtr hsd_msg;

void hsd_callback(const vandy_bluerov::HSDCommand::ConstPtr& msg)
{
    hsd_msg = msg;
    ROS_WARN("got hsd message");
    count++;
}


void callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  using std::cout;
  using std::endl;

  double roll, pitch, yaw;
  double x,y,uh,uv,xtemp,ytemp;
  
  sim_time = 2.0;
  std::cout << "sim_time: " << sim_time << endl;

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

    // convert to rpy
    m.getRPY(roll, pitch, yaw);
    cout << "x: " << x << " " << "y: " << y << " z: " << msg-> pose.pose.position.z << endl;
    cout << "roll: " << roll;
    cout << "pitch: " << pitch;
    cout << "yaw: " << yaw;

    // control commands from hsd controller
    uh = hsd_msg->heading;
    uv = hsd_msg->speed;

    cout << "uh: " << uh;
    cout << "uv: " << uv << endl;

    double state[3] = {x,y,yaw};
    double control_input[2] = {uh,uv};


    // create the ros message that will be sent to the VESC

    if(true)
    {
      hull = runReachability_uuv_vis(state, sim_time, walltime, 0, uh, uv);
      printf("num_boxes: %d, ",num_intermediate);
      visualization_msgs::MarkerArray ma;
      for(int i = 0; i<num_intermediate;i++)
      {
        hull = VisStates[i];

        // if we want to bloat the hyper-rectangles for the width of the car
        if(bloat_reachset)
        {
          hull.dims[0].min = hull.dims[0].min  - 0.99;
          hull.dims[0].max = hull.dims[0].max  + 0.99;
          hull.dims[1].min = hull.dims[1].min  - 0.115;
          hull.dims[1].max = hull.dims[1].max  + 0.115;
        }
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world_ned";
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
        marker.color.r = 0.0;//(double) rand() / (RAND_MAX);
        marker.color.g = (double) rand() / (RAND_MAX);
        marker.color.b = (double) rand() / (RAND_MAX);
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
    allocate_obstacles(1);
    using namespace message_filters;

    // initialize the ros node
    ros::init(argc, argv, "visualize_node_rtreach");

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


    walltime = atoi(argv[1]);
    sim_time = atof(argv[2]);

    vis_pub = n.advertise<visualization_msgs::MarkerArray>( "reach_hull", 100 );
  
    sub = n.subscribe("uuv0/delta_hsd", 1000, hsd_callback);
    sub2 = n.subscribe("uuv0/pose_gt_ned", 1000, callback);

    ros::Rate r(80);
    while(ros::ok())
    {
      ros::spinOnce();
    }
    deallocate_obstacles();


    return 0; 
}

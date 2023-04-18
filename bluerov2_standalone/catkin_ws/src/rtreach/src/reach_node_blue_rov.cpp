#include "ros/ros.h"
#include <iostream>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/subscriber.h>
#include <vandy_bluerov/HSDCommand.h>
#include <rtreach/stamped_ttc.h>
#include <rtreach/reach_tube.h>
#include <rtreach/interval.h>
#include <rtreach/obstacle_list.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <ros/console.h>
#include "std_msgs/Float32.h"
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>
#include <cstdlib>
#include <memory>

// The following node will receive messages from the LEC which will be prediction of the steering angle
// It will also receive messages from the speed node which will dictate how fast the car should travel
// Initially the assmumption will be the the car moves at constant velocity


const int max_hyper_rectangles = 2000;

extern "C"
{ 
    #include "uuv_model_dynamic.h"
    bool runReachability_uuv_dyn(double* start, double simTime, double wallTimeMs, double startMs,double heading, double speed,HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
    double getSimulatedSafeTime(double start[3],double heading_input, double speed);
    bool check_safety(HyperRectangle* rect, double (*cone)[2]);
    HyperRectangle hr_list2[max_hyper_rectangles];
}

int count = 0;
vandy_bluerov::HSDCommand::ConstPtr hsd_msg;
int rect_count = 0;
bool safe=true;
ros::Publisher res_pub;    // publisher for reachability results
rtreach::reach_tube static_obstacles;
double sim_time;
double state[3] = {0.0,0.0,0.0};
double control_input[2] = {0.0,0.0};
int wall_time;
int service_count = 0;
bool log_output_to_console = false;

void hsd_callback(const vandy_bluerov::HSDCommand::ConstPtr& msg)
{
    hsd_msg = msg;
    count++;
}

// Naive O(N^2) check
bool check_obstacle_safety(rtreach::reach_tube obs,HyperRectangle VisStates[],int rect_count)
{   
    bool check_safe = true;
    double cone[2][2] = {{0.0,0.0},{0.0,0.0}};
    // std::cout << "obs_count: " << obs.count << ", rect_count: "<< rect_count << std::endl;
    for (int i=0;i<obs.count;i++)
    {
        if(!check_safe)
        {
            break;
        }
        cone[0][0] = obs.obstacle_list[i].x_min;
        cone[0][1] = obs.obstacle_list[i].x_max;
        cone[1][0] = obs.obstacle_list[i].y_min;
        cone[1][1] = obs.obstacle_list[i].y_max;
        for(int j = 0; j<rect_count;j++)
        {
            check_safe = check_safety(&VisStates[j],cone);
            if(!check_safe)
            {
                break;
            }
        }
    }
    return check_safe;
}

// call backs for all 10 obstacles
void callback(const nav_msgs::Odometry::ConstPtr& msg, const rtreach::reach_tube::ConstPtr& obs1, const rtreach::reach_tube::ConstPtr& obs2, const rtreach::reach_tube::ConstPtr& obs3, 
              const rtreach::reach_tube::ConstPtr& obs4, const rtreach::reach_tube::ConstPtr& obs5, const rtreach::reach_tube::ConstPtr& obs6, 
              const rtreach::reach_tube::ConstPtr& obs7, const rtreach::reach_tube::ConstPtr& obs8)//, const rtreach::reach_tube::ConstPtr& obs9,
              //const rtreach::reach_tube::ConstPtr& obs10)
{
  using std::cout;
  using std::endl;
  double roll, pitch, yaw;
  double x,y,uh,uv,xtemp,ytemp;
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
      // cout << "x: " << x << " " << "y: " << y << " z: " << msg-> pose.pose.position.z << endl;


      // control commands from hsd controller
      uh = hsd_msg->heading;
      uv = hsd_msg->speed;

      // cout << "uh: " << uh;
      // cout << "uv: " << uv << endl;

      state[0] = x;
      state[1] = y;
      state[2] = yaw;

      control_input[0] = uh;
      control_input[1] = uv;
      
      
      runReachability_uuv_dyn(state, sim_time, wall_time, 0,uh,uv,hr_list2,&rect_count,max_hyper_rectangles,true);
      if(log_output_to_console)
      {
        cout << "rect_count:" << rect_count << endl;
        // check the safety for each of the obstacle reachsets

        ROS_WARN("Occupancy Grid Count: %d",obs8->count);
      }
      if(obs8->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs8,hr_list2,max_hyper_rectangles);
         
      }
      if(obs1->count>0  && safe)
      {
         safe  = check_obstacle_safety(*obs1,hr_list2,max_hyper_rectangles);
         
      }
      if(obs2->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs2,hr_list2,max_hyper_rectangles);
         
      }
      if(obs3->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs3,hr_list2,max_hyper_rectangles);
         
      } 
      if(obs4->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs4,hr_list2,max_hyper_rectangles);
         
      }
      if(obs5->count>0 && safe )
      {
         safe  = check_obstacle_safety(*obs5,hr_list2,max_hyper_rectangles);
         
      } 
      if(obs6->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs6,hr_list2,max_hyper_rectangles);
         
      } 
      if(obs7->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs7,hr_list2,max_hyper_rectangles);
         
      }
     
      ROS_WARN("Safe: %d",safe);
      std_msgs::Float32 res_msg;
      res_msg.data = (double)safe;
      res_pub.publish(res_msg);
  }
 else
 {
         ROS_WARN("waiting");
 }
}







int main(int argc, char **argv)
{

    using namespace message_filters;
    // initialize the ros node
    ros::init(argc, argv, "reachnode_blue_rov");
    
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


    wall_time = atoi(argv[1]);
    sim_time = atof(argv[2]);


    // define the subscribers you want 

    ros::Subscriber sub = n.subscribe("uuv0/hsd_command", 100, hsd_callback);
    res_pub = n.advertise<std_msgs::Float32>("reachability_result", 10);
    message_filters::Subscriber<rtreach::reach_tube> obs1(n,"box1/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs2(n,"box2/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs3(n,"box3/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs4(n,"box4/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs5(n,"box5/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs6(n,"box6/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs7(n,"box7/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs8(n,"obstacles",10);
    message_filters::Subscriber<rtreach::reach_tube> obs9(n,"box9/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs10(n,"box10/reach_tube",10);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "uuv0/pose_gt", 10);


    typedef sync_policies::ApproximateTime<nav_msgs::Odometry,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube> MySyncPolicy;//,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube> MySyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), odom_sub, obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8);//,obs8,obs9,obs10);
    sync.registerCallback(boost::bind(&callback, _1, _2,_3,_4,_5,_6,_7,_8,_9));//,_9,_r,_k));


    while(ros::ok())
    {
      // call service periodically 
      ros::spinOnce();
    }
    return 0; 
}

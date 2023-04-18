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
#include <std_msgs/Float32MultiArray.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/package.h>
#include <ros/console.h>
#include "std_msgs/Float32.h"
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>
#include <cstdlib>
#include <memory>
#include <sstream>
#include<ctime>
#include <fstream>
#include <string>


namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
// The following node will receive messages from the LEC which will be prediction of the steering angle
// It will also receive messages from the speed node which will dictate how fast the car should travel
// Initially the assmumption will be the the car moves at constant velocity


const int max_hyper_rectangles = 500;

extern "C"
{
     #include "bicycle_model_uncertainty.h"
     // run reachability for a given wall timer (or iterations if negative)
    bool runReachability_bicycle_uncertain(REAL start[][2], REAL simTime, REAL wallTimeMs, REAL startMs,REAL heading, REAL throttle, REAL parameter_uncertainty, REAL disturbance[][2], HyperRectangle VisStates[],int  *total_intermediate,int max_intermediate,bool plot);
    bool check_safety(HyperRectangle* rect, REAL (*cone)[2]);
    HyperRectangle hr_list2[max_hyper_rectangles];
    HyperRectangle hull;
    void println(HyperRectangle* r);
}

int count = 0;
vandy_bluerov::HSDCommand::ConstPtr hsd_msg;
int rect_count = 0;

// boolean flags

bool safe=true;
bool debug = true;
bool log_output_to_console = true;
bool degraded = false;
double xd_l, xd_u, yd_l, yd_u, vd_l, vd_u, thd_l, thd_u;

double x_unc, y_unc,speed_unc,theta_unc;

ros::Publisher res_pub;
ros::Publisher float_pub;
ros::Publisher vis_pub;
ros::Publisher violation_pub;
ros::Publisher violation_index_pub;
ros::Subscriber sub3;

rtreach::reach_tube static_obstacles;
double sim_time;

double state[4][2] = {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}};
double pi1 = M_PI;
bool bloat_reachset = true;
double parameter_uncertainty = 0.0;

double area = 0.0;
double average_area = 0.0;
double differential=0.0;
double new_mean=0.0;
double safe_periods = 0;
double unsafe_periods = 0;

// disturbances 
double disturbances[4][2] = {{0.0,0.0},{0.0,0.0},{0.0,0.0},{0.0,0.0}}; 
int wall_time;

double clip(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}


// Degredation Callback
void degredation_callback(const std_msgs::Float32MultiArray::ConstPtr&  msg)
{
    degraded = true;
    // forward projection should be smaller when the system is degraded
    wall_time = 0.5;
}

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
    if(log_output_to_console)
    {
      std::cout << "obs_count: " << obs.count << ", rect_count: "<< rect_count << std::endl;
    }
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

        // start at the farthest hyper-rectangle and come in
        for(int j = std::min(rect_count-2,max_hyper_rectangles-1); j>-1;j--)
        {
            check_safe = check_safety(&VisStates[j],cone);
            if(!check_safe)
            {
              
              // grid value publish
              std_msgs::Float32 grid_msg;
              grid_msg.data = obs.obstacle_list[i].value;

              // violation index
              std_msgs::Float32 violation_index_msg;
              violation_index_msg.data = ((float) j)/((float)(rect_count-2)); 

              // interval publish
              rtreach::interval tube;
              tube.x_min = VisStates[j].dims[0].min;
              tube.x_max = VisStates[j].dims[0].max;
              tube.y_min = VisStates[j].dims[1].min;
              tube.y_max = VisStates[j].dims[1].max;


              float_pub.publish(grid_msg);
              violation_pub.publish(tube);
              violation_index_pub.publish(violation_index_msg);

              if(debug && log_output_to_console)
              {
                ROS_WARN("value: %f obj x: [%f,%f], obj y: [%f,%f], reach x: [%f,%f] reach y: [%f, %f]", obs.obstacle_list[i].value,cone[0][0],cone[0][1],cone[1][0],cone[1][1],VisStates[j].dims[0].min,VisStates[j].dims[0].max,VisStates[j].dims[1].min,VisStates[j].dims[1].max);
              }
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
  double roll, pitch, yaw,lin_speed;
  double x,y,uh,uv,xtemp,ytemp;
  double x_area, y_area;
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


      // control commands from hsd controller
      uh = hsd_msg->heading;
      uv = hsd_msg->speed;


      // pre-processing, to eliminate headings that are too large
      if(uh>M_PI)
      {
        uh = (uh * (180/M_PI)) - 360;
      }
      else
      {
        uh = uh * (180/M_PI);
      }
      
      uh = clip(uh,-30,30);

      // control commands from hsd controller
      // convert to radians
      uh = (clip(uh,-30,30) * pi1 ) / 180;

      // scaling for sys_id model (highly experimental)
      uh = (clip(uh,-30,30) * pi1 ) / 120;

      uv = hsd_msg->speed;
    
      if(log_output_to_console)
      {
        cout << "uh: " << uh;
        cout << "uv: " << uv << endl;
        ROS_WARN("Heading Speed: %f, Heading Command %f Linear Speed: %f",uv,uh,lin_speed);
        ROS_WARN("x: %f, y: %f lin_speed: %f, theta: %f",x,y,lin_speed,yaw);
      }
      

    

      state[0][0] = (1-x_unc)*x;
      state[0][1] = (1+x_unc)*x;
      state[1][0] = (1-y_unc)*y;
      state[1][1] = (1+y_unc)*y;
      state[2][0] = (1-speed_unc)*lin_speed;
      state[2][1] = (1+speed_unc)*lin_speed;
      state[3][0] = (1-theta_unc)*yaw;
      state[3][1] = (1+theta_unc)*yaw;

      disturbances[0][0] = xd_l;
      disturbances[0][1] = xd_u;
      disturbances[1][0] = yd_l;
      disturbances[1][1] = yd_u;
      disturbances[2][0] = vd_l;
      disturbances[2][1] = vd_u;
      disturbances[3][0] = thd_l;
      disturbances[3][1] = thd_u;
     
      runReachability_bicycle_uncertain(state, sim_time, wall_time, 0, uh, uv, parameter_uncertainty, disturbances, hr_list2,&rect_count,max_hyper_rectangles,true);
      if(log_output_to_console)
      {
        printf("num_boxes: %d, obs1 count: %d, obs2 count: %d, \n",rect_count,obs1->count,obs2->count);
      }

      if(log_output_to_console)
      {
        cout << "rect_count:" << rect_count << endl;
        // check the safety for each of the obstacle reachsets
        ROS_WARN("Occupancy Grid Count: %d",obs8->count);
        hull = hr_list2[std::min(rect_count-2,max_hyper_rectangles-2)];
        ROS_WARN("reach last rect [%f,%f], [%f,%f]",hull.dims[0].min,hull.dims[0].max,hull.dims[1].min,hull.dims[1].max);
      }
      
      safe = true;
      if(obs8->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs8,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      }
      if(obs1->count>0  && safe)
      {
         safe  = check_obstacle_safety(*obs1,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      }
      if(obs2->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs2,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      }
      if(obs3->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs3,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      } 
      if(obs4->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs4,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      }
      if(obs5->count>0 && safe )
      {
         safe  = check_obstacle_safety(*obs5,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      } 
      if(obs6->count>0 && safe)
      {
         safe  = check_obstacle_safety(*obs6,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      } 
      if(obs7->count>0 && safe)
      {
        safe  = check_obstacle_safety(*obs7,hr_list2,std::min(max_hyper_rectangles,rect_count));
         
      }

      if(safe)
      {
          safe_periods+=1;
      }
      else
      {
          unsafe_periods+=1;
      }
      
      std_msgs::Float32 res_msg;
      res_msg.data = (double)safe;
      res_pub.publish(res_msg);
      if(safe<1.0 && log_output_to_console)
      {
        ROS_WARN("Safe: %d",safe);
      }

      // visualization_debugging
      if(debug){
        visualization_msgs::MarkerArray ma;
        // std::min(rect_count-1,max_hyper_rectangles-1)
        area = 0.0;
        for(int i = 0; i<std::min(rect_count-1,max_hyper_rectangles);i+=1)
        {
          HyperRectangle hull = hr_list2[i];

          hull.dims[0].min = hull.dims[0].min  - 0.2285;
          hull.dims[0].max = hull.dims[0].max  + 0.2285;
          hull.dims[1].min = hull.dims[1].min  - 0.169;
          hull.dims[1].max = hull.dims[1].max  + 0.169;

          x_area = std::abs(hull.dims[0].max - hull.dims[0].min);
          y_area = std::abs(hull.dims[1].max - hull.dims[1].min);
          area += (x_area * y_area);
          visualization_msgs::Marker marker;
          marker.header.frame_id = "world_ned";
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
          marker.scale.z = 0.5;
          marker.color.a = 1.0; 
          if(safe)
          {
            marker.color.r = 0.02;
            marker.color.g = 0.66;
            marker.color.b = 0.25;
          }
          else
          {
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0; 
          }
          marker.lifetime =ros::Duration(0.1); 
          ma.markers.push_back(marker);
        }

        // publish marker
        vis_pub.publish( ma );
        differential = (area - average_area) / count;
        new_mean = average_area + differential;
        average_area = new_mean;
        
        hull = hr_list2[std::min(rect_count-2,max_hyper_rectangles)];

        if(log_output_to_console)
        {
            ROS_WARN("x: %f, y: %f \n",(hull.dims[0].max+hull.dims[0].min)/2.0,(hull.dims[1].max+hull.dims[1].min)/2.0);
            ROS_WARN("average_area: %f \n", average_area);
        }
        
      }

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
    std::string save_path;

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
        std::cout << "please specify whether or not this should be run in debug mode (0 or 1)" << std::endl;
        exit(0);
    }


    wall_time = atoi(argv[1]);
    sim_time = atof(argv[2]);
    debug = (bool)atoi(argv[3]);
    debug = true;

    // what level of parameter to consider in the experiments
    parameter_uncertainty = atof(argv[4]);

    xd_l  = atof(argv[5]);
    xd_u  = atof(argv[6]);
    yd_l  = atof(argv[7]);
    yd_u  = atof(argv[8]);
    vd_l  = atof(argv[9]);
    vd_u  = atof(argv[10]);
    thd_l = atof(argv[11]);
    thd_u = atof(argv[12]);
    x_unc = atof(argv[13]);
    y_unc = atof(argv[14]);
    speed_unc = atof(argv[15]);
    theta_unc = atof(argv[16]);

    save_path = std::string(argv[13]) +  "/uncertainty_"+ patch::to_string(xd_l) + "_" + patch::to_string(xd_u) + "_" + patch::to_string(yd_l) + "_" + patch::to_string(yd_u) + "_" +   
    patch::to_string(vd_l) +  "_" + patch::to_string(vd_u)  + "_" +  patch::to_string(thd_l)  + "_" + patch::to_string(thd_u) + "_" + patch::to_string(parameter_uncertainty)+".csv";
    

    // define the subscribers you want 

    // ros::Subscriber sub = n.subscribe("uuv0/delta_hsd", 1, hsd_callback);
    ros::Subscriber sub = n.subscribe("uuv0/hsd_command", 1, hsd_callback);

    res_pub = n.advertise<std_msgs::Float32>("reachability_result", 1);
    float_pub = n.advertise<std_msgs::Float32>("unsafe_grid_value", 1);
    violation_pub = n.advertise<rtreach::interval>("unsafe_hyperrectangle",1);
    violation_index_pub = n.advertise<std_msgs::Float32>("unsafe_hyperrectangle_index",1);
    if(debug)
    {
      vis_pub = n.advertise<visualization_msgs::MarkerArray>( "reach_verify", 10 );
    }
    message_filters::Subscriber<rtreach::reach_tube> obs1(n,"box1/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs2(n,"box2/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs3(n,"box3/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs4(n,"box4/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs5(n,"box5/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs6(n,"box6/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs7(n,"uuv0/bounding_box_interval",10);
    message_filters::Subscriber<rtreach::reach_tube> obs8(n,"obstacles",10);
    message_filters::Subscriber<rtreach::reach_tube> obs9(n,"box9/reach_tube",10);
    message_filters::Subscriber<rtreach::reach_tube> obs10(n,"box10/reach_tube",10);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "uuv0/pose_gt_noisy_ned", 10);
    sub3 = n.subscribe("uuv0/thruster_reallocation",100, degredation_callback);


    typedef sync_policies::ApproximateTime<nav_msgs::Odometry,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube,rtreach::reach_tube> MySyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), odom_sub, obs1,obs2,obs3,obs4,obs5,obs6,obs7,obs8);
    sync.registerCallback(boost::bind(&callback, _1, _2,_3,_4,_5,_6,_7,_8,_9));//,_9,_r,_k));


    while(ros::ok())
    {
      // call service periodically 
      ros::spinOnce();
    }

    double total_periods = safe_periods + unsafe_periods;
    double safety_score = safe_periods / total_periods;
    std::cout << "total periods: " << total_periods << " safety_score: " << safety_score << " safe_periods: " << safe_periods << std::endl;
    std::ofstream outfile(save_path.c_str() , std::ios::app);
    outfile << safety_score << ", " << average_area << std::endl;
    outfile.close();
    return 0; 
}




